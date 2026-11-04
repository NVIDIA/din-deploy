// SPDX-License-Identifier: Apache-2.0
// VMM residual execution and SDPA adapted from the user's tensorrt-playground
// apps/seedvr2_resblock and apps/cudnn_attention implementations.
#include "seedvr_vae.h"
#include "nvtx_helper.h"

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "seedvr_vae_kernels.h"
#include "seedvr_vae_memory.h"
#include <cudnn_frontend.h>
#include <nlohmann/json.hpp>

namespace din::seedvr
{
namespace
{
namespace fe = cudnn_frontend;
using Json = nlohmann::json;
using Attr = std::shared_ptr<fe::graph::Tensor_attributes>;
void check(cudaError_t e, const char* api, const std::source_location location = std::source_location::current())
{
    working_memory::check(e, api, location);
}
void check(cudnnStatus_t e, const char* api, const std::source_location location = std::source_location::current())
{
    if (e != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string(api) + " failed: cuDNN status " + std::to_string(int(e)) + ": " +
                                 cudnnGetErrorString(e) + " at " + location.file_name() + ":" +
                                 std::to_string(location.line()));
}
void check(fe::error_t e, const char* api, const std::source_location location = std::source_location::current())
{
    if (e.is_bad())
        throw std::runtime_error(std::string(api) + " failed: " + e.get_message() + " at " + location.file_name() +
                                 ":" + std::to_string(location.line()));
}
void check(CUresult e, const char* api, const std::source_location location = std::source_location::current())
{
    working_memory::check(e, api, location);
}
#define SEEDVR_VAE_CHECK(expression) check((expression), #expression)
struct Memory
{
    size_t live = 0, peak = 0;
    void add(size_t n)
    {
        live += n;
        peak = std::max(peak, live);
    }
    void sub(size_t n)
    {
        live -= n;
    }
};
struct Buffer
{
    void* p = nullptr;
    size_t bytes = 0;
    Memory* memory = nullptr;
    explicit Buffer(size_t n = 0, Memory* m = nullptr)
        : bytes(n)
        , memory(m)
    {
        if (n)
        {
            try
            {
                SEEDVR_VAE_CHECK(cudaMalloc(&p, n));
            }
            catch (const std::exception& e)
            {
                throw std::runtime_error(std::string(e.what()) + "; requested_bytes=" + std::to_string(n) + "; " +
                                         working_memory::memory_snapshot());
            }
        }
        if (memory)
            memory->add(bytes);
    }
    ~Buffer()
    {
        if (p)
            cudaFree(p);
        if (memory)
            memory->sub(bytes);
    }
    Buffer(const Buffer&) = delete;
};
// Stable graph I/O belongs to the stream's CUDA pool, not the transient VMM arena.
// Allocated once during construction and released only when the session dies.
struct PersistentIO
{
    void* p = nullptr;
    size_t bytes;
    cudaStream_t stream;
    PersistentIO(size_t n, cudaStream_t s)
        : bytes(n)
        , stream(s)
    {
        try
        {
            SEEDVR_VAE_CHECK(cudaMallocAsync(&p, n, stream));
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(std::string(e.what()) + "; persistent_io_bytes=" + std::to_string(n));
        }
    }
    ~PersistentIO()
    {
        if (p)
        {
            const auto status = cudaFreeAsync(p, stream);
            if (status != cudaSuccess)
                std::fprintf(stderr, "[VAE cleanup] cudaFreeAsync(IO) failed: %s (%d)\n", cudaGetErrorName(status),
                             int(status));
        }
    }
    PersistentIO(const PersistentIO&) = delete;
};
// Working buffers reserve VMM addresses during planning; the shared arena
// supplies their physical backing only after both directions are prepared.
struct WorkingBuffer
{
    size_t bytes, allocation_bytes, offset;
    working_memory::View view;
    void* p;
    WorkingBuffer(size_t n, working_memory::Arena& arena, working_memory::Planner& planner)
        : bytes(n)
        , allocation_bytes(working_memory::align_up(n, arena.granularity))
        , offset(planner.acquire(allocation_bytes))
        , view(arena, offset, allocation_bytes)
        , p(view.data())
    {
    }
};
struct Tensor
{
    int c, t, h, w;
    int64_t stride;
    size_t bytes, offset;
    CUdeviceptr base = 0;
    std::unique_ptr<working_memory::View> view;
    Tensor(int channels, int frames, int height, int width, working_memory::Arena& arena,
           working_memory::Planner& planner)
        : c(channels)
        , t(frames)
        , h(height)
        , w(width)
        , stride(int64_t(c) * h * w)
    {
        size_t unit = 0;
        if (c >= 128)
        {
            unit = working_memory::align_up(size_t(stride) * 2, std::lcm(arena.granularity, size_t(w) * c * 2));
            stride = int64_t(unit / 2);
        }
        bytes = working_memory::align_up(size_t(stride) * t * 2, arena.granularity);
        offset = planner.acquire(bytes);
        view = std::make_unique<working_memory::View>(arena, offset, bytes, unit);
        if (unit)
            base = view->base;
    }
    half* data() const
    {
        return static_cast<half*>(view->data());
    }
    void* padded() const
    {
        return reinterpret_cast<void*>(base);
    }
    std::vector<int64_t> dims(int pad = 0) const
    {
        return {1, c, t + pad, h, w};
    }
    std::vector<int64_t> strides(int pad = 0) const
    {
        return {(t + pad) * stride, 1, stride, int64_t(w) * c, c};
    }
};
using TensorPtr = std::shared_ptr<Tensor>;
// A session owns every allocation used by its graph. Local shared_ptr leases
// describe CPU-side lifetimes; reuse is safe because all work uses one stream.
struct Session
{
    working_memory::Planner planner;
    std::vector<std::unique_ptr<Tensor>> tensors;
    std::vector<std::unique_ptr<WorkingBuffer>> buffers;
    std::unique_ptr<PersistentIO> input, output;
    std::unique_ptr<WorkingBuffer> scratch;
    std::vector<std::function<void()>> operations;
    size_t workspace_bytes = 0;
    size_t working_bytes = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    bool frozen = false;
    bool failed = false;
    ~Session()
    {
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
    }
};
struct Plan
{
    fe::graph::Graph graph;
    std::vector<Attr> args;
    bool pending_tune = false;
    int64_t selected = -1;
    size_t required_workspace = 0;
    Json identity;
    std::filesystem::path cache_file;
    std::string baseline_name;
    std::vector<int64_t> candidate_indices;
    Plan()
    {
        graph.set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
    }
    Attr tensor(const std::string& name, const std::vector<int64_t>& d, const std::vector<int64_t>& s)
    {
        auto a = graph.tensor(fe::graph::Tensor_attributes().set_name(name).set_dim(d).set_stride(s));
        args.push_back(a);
        return a;
    }
    void build(cudnnHandle_t handle, size_t limit, bool autotune, const std::filesystem::path& cache,
               int max_candidates)
    {
        if (autotune)
        {
            Json spec;
            graph.serialize(spec);
            // Frontend graph IDs depend on construction order, not semantics.
            spec.erase("gid");
            int device, driver, runtime;
            SEEDVR_VAE_CHECK(cudaGetDevice(&device));
            cudaDeviceProp prop{};
            SEEDVR_VAE_CHECK(cudaGetDeviceProperties(&prop, device));
            SEEDVR_VAE_CHECK(cudaDriverGetVersion(&driver));
            SEEDVR_VAE_CHECK(cudaRuntimeGetVersion(&runtime));
            std::ostringstream uuid;
            for (auto value : prop.uuid.bytes)
                uuid << std::hex << std::setw(2) << std::setfill('0') << int(static_cast<unsigned char>(value));
            identity = {{"schema", 3},
                        {"max_candidates", max_candidates},
                        {"graph", spec},
                        {"gpu_uuid", uuid.str()},
                        {"gpu", prop.name},
                        {"driver", driver},
                        {"cuda_runtime", runtime},
                        {"cudnn", cudnnGetVersion()},
                        {"frontend", CUDNN_FRONTEND_VERSION},
                        {"workspace_limit", limit},
                        {"method", "graph_replay_median_7_probe100ms_10x_v2"}};
            uint64_t hash = 14695981039346656037ull;
            for (unsigned char c : identity.dump())
            {
                hash ^= c;
                hash *= 1099511628211ull;
            }
            std::ostringstream name;
            name << std::hex << hash << ".json";
            if (!cache.empty())
            {
                std::filesystem::create_directories(cache);
                cache_file = cache / name.str();
            }
        }
        SEEDVR_VAE_CHECK(graph.validate());
        SEEDVR_VAE_CHECK(graph.build_operation_graph(handle));
        SEEDVR_VAE_CHECK(graph.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}));
        graph.deselect_workspace_greater_than(limit);
        SEEDVR_VAE_CHECK(graph.check_support(handle));
        // Cache records include the full identity, not just the filename hash.
        // Rebuild and validate the recorded index/tag before trusting it.
        if (autotune && !cache_file.empty())
        {
            std::ifstream file(cache_file);
            if (file)
            {
                try
                {
                    Json record;
                    file >> record;
                    if (record.at("identity") == identity)
                    {
                        const int64_t index = record.at("selected_index");
                        if (index >= 0 && index < graph.get_execution_plan_count() &&
                            !graph.build_plan_at_index(handle, index).is_bad())
                        {
                            std::string name;
                            int64_t bytes = 0;
                            if (!graph.get_plan_name_at_index(index, name).is_bad() &&
                                name == record.at("selected_name").get<std::string>() &&
                                !graph.get_workspace_size_plan_at_index(index, bytes).is_bad() && bytes >= 0 &&
                                size_t(bytes) <= limit)
                            {
                                selected = index;
                                required_workspace = size_t(bytes);
                                std::cout << "VAE autotune cache hit " << cache_file.filename().string() << " " << name
                                          << '\n';
                                return;
                            }
                        }
                    }
                }
                catch (const Json::exception&)
                {
                }  // Invalid cache: remeasure, never change tolerances.
                std::cout << "VAE autotune cache invalid; remeasuring " << cache_file.filename().string() << '\n';
            }
        }
        SEEDVR_VAE_CHECK(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
        SEEDVR_VAE_CHECK(graph.get_plan_name(baseline_name));
        if (autotune)
        {
            // cuDNN supplies heuristic priority order. Build only the first N
            // supported, budget-compliant plans; skip unsupported entries.
            for (int64_t index = 0;
                 index < graph.get_execution_plan_count() && candidate_indices.size() < size_t(max_candidates); ++index)
            {
                if (graph.build_plan_at_index(handle, index).is_bad())
                    continue;
                int64_t bytes = 0;
                if (graph.get_workspace_size_plan_at_index(index, bytes).is_bad() || bytes < 0 || size_t(bytes) > limit)
                    continue;
                candidate_indices.push_back(index);
                required_workspace = std::max(required_workspace, size_t(bytes));
            }
            if (candidate_indices.empty())
                throw std::runtime_error("No supported cuDNN tuning candidates");
            pending_tune = true;
        }
        else
            required_workspace = size_t(graph.get_workspace_size());
        if (required_workspace > limit)
            throw std::runtime_error("cuDNN workspace exceeded configured budget");
    }
    struct Trial
    {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaEvent_t start = nullptr, stop = nullptr;
        ~Trial()
        {
            if (exec)
                cudaGraphExecDestroy(exec);
            if (graph)
                cudaGraphDestroy(graph);
            if (start)
                cudaEventDestroy(start);
            if (stop)
                cudaEventDestroy(stop);
        }
    };
    void tune(cudnnHandle_t handle, std::unordered_map<Attr, void*>& pack, void* workspace)
    {
        cudaStream_t stream;
        SEEDVR_VAE_CHECK(cudnnGetStream(handle, &stream));
        // Tuning only runs before full-VAE capture. Inputs are real intermediate
        // activations; candidate outputs are distinct from all live inputs.
        SEEDVR_VAE_CHECK(cudaStreamSynchronize(stream));
        Json candidates = Json::array();
        float best = std::numeric_limits<float>::infinity();
        for (const auto index : candidate_indices)
        {
            std::string name;
            if (graph.get_plan_name_at_index(index, name).is_bad())
                continue;  // Unsupported/unbuilt plan.
            Json row = {{"index", index}, {"name", name}};
            int64_t bytes = 0;
            auto size_status = graph.get_workspace_size_plan_at_index(index, bytes);
            if (size_status.is_bad() || bytes < 0 || size_t(bytes) > required_workspace)
                continue;
            row["workspace_bytes"] = bytes;
            auto warm = graph.execute_plan_at_index(handle, pack, workspace, index);
            if (warm.is_bad())
            {
                row["error"] = warm.get_message();
                candidates.push_back(row);
                continue;
            }
            SEEDVR_VAE_CHECK(cudaStreamSynchronize(stream));
            Trial trial;
            SEEDVR_VAE_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            auto status = graph.execute_plan_at_index(handle, pack, workspace, index);
            const auto capture_status = cudaStreamEndCapture(stream, &trial.graph);
            if (status.is_bad() || capture_status != cudaSuccess)
            {
                row["error"] = status.is_bad() ? status.get_message() : cudaGetErrorString(capture_status);
                candidates.push_back(row);
                continue;
            }
            SEEDVR_VAE_CHECK(cudaGraphInstantiate(&trial.exec, trial.graph, nullptr, nullptr, 0));
            SEEDVR_VAE_CHECK(cudaEventCreate(&trial.start));
            SEEDVR_VAE_CHECK(cudaEventCreate(&trial.stop));
            SEEDVR_VAE_CHECK(cudaEventRecord(trial.start, stream));
            SEEDVR_VAE_CHECK(cudaGraphLaunch(trial.exec, stream));
            SEEDVR_VAE_CHECK(cudaEventRecord(trial.stop, stream));
            SEEDVR_VAE_CHECK(cudaEventSynchronize(trial.stop));
            float probe_ms;
            SEEDVR_VAE_CHECK(cudaEventElapsedTime(&probe_ms, trial.start, trial.stop));
            row["probe_ms"] = double(probe_ms);
            if (probe_ms > 100.f && probe_ms > best * 10.f)
            {
                row["pruned"] = "Replay exceeds both 100 ms and 10x the current best median";
                candidates.push_back(row);
                continue;
            }
            SEEDVR_VAE_CHECK(cudaGraphLaunch(trial.exec, stream));
            std::vector<float> samples;
            for (int repeat = 0; repeat < 7; ++repeat)
            {
                SEEDVR_VAE_CHECK(cudaEventRecord(trial.start, stream));
                SEEDVR_VAE_CHECK(cudaGraphLaunch(trial.exec, stream));
                SEEDVR_VAE_CHECK(cudaEventRecord(trial.stop, stream));
                SEEDVR_VAE_CHECK(cudaEventSynchronize(trial.stop));
                float ms;
                SEEDVR_VAE_CHECK(cudaEventElapsedTime(&ms, trial.start, trial.stop));
                samples.push_back(ms);
            }
            // cuDNN overrides JSON float serialization with hex strings.
            // Store double values so timing reports contain decimal numbers.
            row["samples_ms"] = std::vector<double>(samples.begin(), samples.end());
            std::sort(samples.begin(), samples.end());
            const float median = samples[samples.size() / 2];
            row["median_ms"] = double(median);
            if (median < best)
            {
                best = median;
                selected = index;
            }
            candidates.push_back(row);
        }
        if (selected < 0)
            throw std::runtime_error("No cuDNN candidate passed graph-replay autotuning");
        pending_tune = false;
        std::string name;
        SEEDVR_VAE_CHECK(graph.get_plan_name_at_index(selected, name));
        Json record = {{"identity", identity},        {"selected_index", selected},      {"selected_name", name},
                       {"selected_ms", double(best)}, {"heuristic_name", baseline_name}, {"candidates", candidates}};
        if (!cache_file.empty())
        {
            auto temp = cache_file;
            temp += ".tmp";
            {
                std::ofstream file(temp);
                file << record.dump(2);
                if (!file)
                    throw std::runtime_error("Cannot write VAE tuning cache");
            }
            std::error_code ec;
            std::filesystem::remove(cache_file, ec);
            std::filesystem::rename(temp, cache_file);
        }
        std::cout << "VAE autotune selected " << name << " " << best << " ms, " << candidates.size() << " candidates\n";
    }
    void run(cudnnHandle_t handle, const std::vector<void*>& ptrs, void* workspace)
    {
        std::unordered_map<Attr, void*> pack;
        for (size_t i = 0; i < args.size(); ++i)
            pack.emplace(args[i], ptrs.at(i));
        if (pending_tune)
            tune(handle, pack, workspace);
        // Re-execute the winner after trials so downstream operations see its
        // output, not the output of the last candidate that was benchmarked.
        if (selected >= 0)
            SEEDVR_VAE_CHECK(graph.execute_plan_at_index(handle, pack, workspace, selected));
        else
            SEEDVR_VAE_CHECK(graph.execute(handle, pack, workspace));
    }
};
}  // namespace

class NativeVae::Impl
{
public:
    cudaStream_t stream;
    cudnnHandle_t handle = nullptr;
    size_t limit;
    bool autotune;
    int tuning_candidates;
    std::filesystem::path tuning_cache;
    working_memory::Arena arena;
    Json manifest;
    std::map<std::string, std::unique_ptr<Buffer>> weights;
    std::map<std::string, std::unique_ptr<Buffer>> folded_weights;
    bool image_mode = false;
    std::map<std::string, std::unique_ptr<Plan>> plans;
    std::unique_ptr<Session> encoder, decoder;
    Session* active = nullptr;
    bool capturing = false;
    Impl(const std::filesystem::path& dir, cudaStream_t s, size_t l, const NativeVae::Shapes& shapes, bool tune,
         const std::filesystem::path& cache, int candidates)
        : stream(s)
        , limit(l)
        , autotune(tune)
        , tuning_candidates(candidates)
        , tuning_cache(cache)
    {
        if (!stream)
            throw std::invalid_argument("Native VAE graph execution requires an explicit CUDA stream");
        if (tuning_candidates < 1)
            throw std::invalid_argument("Native VAE tuning candidate count must be positive");
        if (!shapes.encoder && !shapes.decoder)
            throw std::invalid_argument("Native VAE requires at least one fixed input shape");
        if (shapes.encoder)
            validate_shape(true, *shapes.encoder);
        if (shapes.decoder)
            validate_shape(false, *shapes.decoder);
        std::cerr << "[VAE initialization] loading weights from " << dir.string() << std::endl;
        std::ifstream f(dir / "manifest.json");
        if (!f)
            throw std::runtime_error("Missing native VAE manifest: " + dir.string());
        f >> manifest;
        if (manifest.at("version") != 1)
            throw std::runtime_error("Unsupported native VAE manifest version");
        for (auto& [name, spec] : manifest.at("weights").items())
        {
            auto bytes = spec.at("bytes").get<size_t>();
            std::filesystem::path rel = spec.at("file").get<std::string>();
            if (rel.is_absolute() || rel.has_parent_path())
                throw std::runtime_error("Invalid VAE weight filename");
            std::ifstream data(dir / rel, std::ios::binary | std::ios::ate);
            if (!data || size_t(data.tellg()) != bytes)
                throw std::runtime_error("Invalid native VAE weight: " + name);
            std::vector<char> host(bytes);
            data.seekg(0);
            data.read(host.data(), bytes);
            if (!data)
                throw std::runtime_error("Weight read failed");
            auto b = std::make_unique<Buffer>(bytes);
            SEEDVR_VAE_CHECK(cudaMemcpy(b->p, host.data(), bytes, cudaMemcpyHostToDevice));
            weights.emplace(name, std::move(b));
        }
        SEEDVR_VAE_CHECK(cudnnCreate(&handle));
        try
        {
            SEEDVR_VAE_CHECK(cudnnSetStream(handle, stream));
            if (shapes.encoder)
            {
                std::cerr << "[VAE initialization] preparing encoder input T,H,W=" << shapes.encoder->frames << ','
                          << shapes.encoder->height << ',' << shapes.encoder->width << std::endl;
                encoder = prepare(true, *shapes.encoder);
            }
            if (shapes.decoder)
            {
                std::cerr << "[VAE initialization] preparing decoder latent T,H,W=" << shapes.decoder->frames << ','
                          << shapes.decoder->height << ',' << shapes.decoder->width << std::endl;
                decoder = prepare(false, *shapes.decoder);
            }
            arena.update_capacity();
            size_t weight_bytes = 0;
            for (const auto& [name, weight] : weights)
                weight_bytes += weight->bytes;
            for (const auto& [name, weight] : folded_weights)
                if (weight)
                    weight_bytes += weight->bytes;
            std::cerr << "[VAE memory plan BEFORE commit] encoder_required_bytes="
                      << (encoder ? encoder->working_bytes : 0)
                      << " decoder_required_bytes=" << (decoder ? decoder->working_bytes : 0)
                      << " shared_required_bytes=" << arena.capacity << " (max, not sum)"
                      << " weight_bytes=" << weight_bytes << " workspace_limit_bytes=" << limit
                      << " encoder_persistent_io_bytes="
                      << (encoder ? encoder->input->bytes + encoder->output->bytes : 0)
                      << " decoder_persistent_io_bytes="
                      << (decoder ? decoder->input->bytes + decoder->output->bytes : 0) << std::endl;
            arena.commit();
        }
        catch (...)
        {
            active = nullptr;
            encoder.reset();
            decoder.reset();
            plans.clear();
            cudnnDestroy(handle);
            handle = nullptr;
            throw;
        }
    }
    ~Impl()
    {
        cudaStreamSynchronize(stream);
        encoder.reset();
        decoder.reset();
        plans.clear();
        if (handle)
            cudnnDestroy(handle);
    }
    template <class F, class... Args>
    void enqueue(F fn, Args... args)
    {
        active->operations.emplace_back(
            [fn, values = std::make_tuple(args...)]()
            {
                if constexpr (std::is_same_v<std::invoke_result_t<F, Args...>, cudaError_t>)
                    SEEDVR_VAE_CHECK(std::apply(fn, values));
                else
                    std::apply(fn, values);
            });
    }
    void enqueue_plan(Plan& plan, std::vector<void*> pointers)
    {
        auto* session = active;
        auto* compiled = &plan;
        active->operations.emplace_back(
            [this, session, compiled, pointers = std::move(pointers)]()
            {
                compiled->run(handle, pointers, session->scratch->p);
            });
    }
    TensorPtr tensor(int c, int t, int h, int w)
    {
        if (active->frozen)
            throw std::runtime_error("Cannot allocate after VAE preparation");
        auto owned = std::make_unique<Tensor>(c, t, h, w, arena, active->planner);
        auto* raw = owned.get();
        active->tensors.push_back(std::move(owned));
        auto* planner = &active->planner;
        return TensorPtr(raw,
                         [planner](Tensor* p)
                         {
                             planner->release(p->offset, p->bytes);
                         });
    }
    std::shared_ptr<WorkingBuffer> buffer(size_t bytes)
    {
        if (active->frozen)
            throw std::runtime_error("Cannot allocate after VAE preparation");
        auto owned = std::make_unique<WorkingBuffer>(bytes, arena, active->planner);
        auto* raw = owned.get();
        active->buffers.push_back(std::move(owned));
        auto* planner = &active->planner;
        return std::shared_ptr<WorkingBuffer>(raw,
                                              [planner](WorkingBuffer* p)
                                              {
                                                  planner->release(p->offset, p->allocation_bytes);
                                              });
    }
    half* weight(const Json& op, const char* key)
    {
        return static_cast<half*>(weights.at(op.at(key).get<std::string>())->p);
    }
    void dump(const TensorPtr& x, const std::string& name)
    {
        const char* dir = std::getenv("SEEDVR_VAE_DEBUG_DIR");
        if (!dir || !*dir)
            return;
        auto* tensor = x.get();
        active->operations.emplace_back(
            [this, tensor, name, dir = std::string(dir)]()
            {
                if (capturing)
                    return;  // Host readback is not permitted during capture.
                SEEDVR_VAE_CHECK(cudaStreamSynchronize(stream));
                std::filesystem::create_directories(dir);
                std::vector<half> data(size_t(tensor->c) * tensor->t * tensor->h * tensor->w);
                for (int t = 0; t < tensor->t; ++t)
                    SEEDVR_VAE_CHECK(cudaMemcpy(data.data() + size_t(t) * tensor->c * tensor->h * tensor->w,
                                                tensor->data() + t * tensor->stride,
                                                size_t(tensor->c) * tensor->h * tensor->w * 2, cudaMemcpyDeviceToHost));
                std::ofstream f(std::filesystem::path(dir) / (name + ".bin"), std::ios::binary);
                f.write(reinterpret_cast<const char*>(data.data()), data.size() * 2);
            });
    }
    void workspace(Plan& p)
    {
        active->workspace_bytes = std::max(active->workspace_bytes, p.required_workspace);
    }
    TensorPtr image_conv(const TensorPtr& x, const Json& op, const std::vector<int64_t>& d,
                         const std::vector<int64_t>& st, const std::vector<int64_t>& pads)
    {
        std::string name = op.at("weight");
        auto& fw = folded_weights[name];
        auto& fb = folded_weights[op.at("bias").get<std::string>()];
        if (!fw)
        {
            const int64_t plane = d[1] * d[3] * d[4];
            std::vector<half> original(size_t(d[0] * d[2] * plane));
            SEEDVR_VAE_CHECK(
                cudaMemcpy(original.data(), weight(op, "weight"), original.size() * 2, cudaMemcpyDeviceToHost));
            std::vector<float> folded(size_t(d[0] * plane), 0);
            for (int64_t k = 0; k < d[0]; ++k)
                for (int64_t t = 0; t < d[2]; ++t)
                    for (int64_t i = 0; i < plane; ++i)
                        folded[k * plane + i] += __half2float(original[(k * d[2] + t) * plane + i]);
            fw = std::make_unique<Buffer>(folded.size() * 4);
            SEEDVR_VAE_CHECK(cudaMemcpy(fw->p, folded.data(), folded.size() * 4, cudaMemcpyHostToDevice));
            std::vector<half> bh(d[0]);
            std::vector<float> bf(d[0]);
            SEEDVR_VAE_CHECK(cudaMemcpy(bh.data(), weight(op, "bias"), bh.size() * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < bf.size(); ++i)
                bf[i] = __half2float(bh[i]);
            fb = std::make_unique<Buffer>(bf.size() * 4);
            SEEDVR_VAE_CHECK(cudaMemcpy(fb->p, bf.data(), bf.size() * 4, cudaMemcpyHostToDevice));
        }
        bool down = st[1] == 2;
        int oh = (x->h + int(pads[1] + pads[4]) + (down ? 1 : 0) - int(d[3])) / int(st[1]) + 1;
        int ow = (x->w + int(pads[2] + pads[5]) + (down ? 1 : 0) - int(d[4])) / int(st[2]) + 1;
        auto y = tensor(int(d[0]), 1, oh, ow);
        auto input = buffer(size_t(x->c) * x->h * x->w * 4);
        enqueue(vae::to_float, x->data(), static_cast<float*>(input->p), int64_t(x->c) * x->h * x->w, stream);
        std::string key = "image" + Json({d, st, pads, x->dims(), y->dims()}).dump();
        auto& plan = plans[key];
        if (!plan)
        {
            auto p = std::make_unique<Plan>();
            auto a =
                p->tensor("x", {1, x->c, x->h, x->w}, {int64_t(x->c) * x->h * x->w, 1, int64_t(x->w) * x->c, x->c});
            a->set_data_type(fe::DataType_t::FLOAT);
            auto wa = p->tensor("w", {d[0], d[1], d[3], d[4]}, {d[1] * d[3] * d[4], 1, d[1] * d[4], d[1]});
            wa->set_data_type(fe::DataType_t::FLOAT);
            auto b = p->tensor("b", {1, d[0], 1, 1}, {d[0], 1, d[0], d[0]});
            b->set_data_type(fe::DataType_t::FLOAT);
            auto v = p->graph.conv_fprop(a, wa,
                                         fe::graph::Conv_fprop_attributes()
                                             .set_pre_padding({pads[1], pads[2]})
                                             .set_post_padding({pads[4] + (down ? 1 : 0), pads[5] + (down ? 1 : 0)})
                                             .set_stride({st[1], st[2]})
                                             .set_dilation({1, 1}));
            auto out = p->graph.pointwise(v, b, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
            out->set_output(true)
                .set_data_type(fe::DataType_t::HALF)
                .set_dim({1, d[0], oh, ow})
                .set_stride({d[0] * oh * ow, 1, d[0] * ow, d[0]});
            p->args.push_back(out);
            p->build(handle, limit, autotune, tuning_cache, tuning_candidates);
            plan = std::move(p);
        }
        workspace(*plan);
        enqueue_plan(*plan, {input->p, fw->p, fb->p, y->data()});
        dump(y, op.at("name"));
        return y;
    }
    TensorPtr conv(const TensorPtr& x, const Json& op)
    {
        const bool trace = std::getenv("SEEDVR_VAE_TRACE") != nullptr;
        if (trace)
            std::cerr << "VAE conv " << op.at("name") << " input=" << Json(x->dims()).dump() << std::endl;
        auto d = manifest.at("weights").at(op.at("weight").get<std::string>()).at("shape").get<std::vector<int64_t>>();
        if (d.size() != 5 || d[1] != x->c)
            throw std::runtime_error("Invalid convolution weights: " + op.at("name").get<std::string>());
        auto st = op.at("stride").get<std::vector<int64_t>>();
        auto pads = op.at("pads").get<std::vector<int64_t>>();
        const std::string name = op.at("name");
        if (image_mode && name.find("attentions.") == std::string::npos &&
            name.find("upscale_conv") == std::string::npos)
            return image_conv(x, op, d, st, pads);
        // Downsample has explicit right/bottom zero padding in ONNX before Conv.
        bool down = st[1] == 2;
        int tp = int(d[2] - 1);
        int ot = (x->t + tp - int(d[2])) / int(st[0]) + 1;
        int oh = (x->h + int(pads[1] + pads[4]) + (down ? 1 : 0) - int(d[3])) / int(st[1]) + 1;
        int ow = (x->w + int(pads[2] + pads[5]) + (down ? 1 : 0) - int(d[4])) / int(st[2]) + 1;
        auto y = tensor(int(d[0]), ot, oh, ow);
        const int physical_h = int(x->stride / (int64_t(x->w) * x->c));
        const int output_h = (physical_h + int(pads[1] + pads[4]) + (down ? 1 : 0) - int(d[3])) / int(st[1]) + 1;
        const int64_t output_stride = int64_t(output_h) * ow * d[0];
        const int64_t logical_frame = int64_t(x->h) * x->w * x->c;
        if (x->stride > logical_frame)
            enqueue(cudaMemset2DAsync, x->data() + logical_frame, size_t(x->stride) * 2, 0,
                    size_t(x->stride - logical_frame) * 2, x->t, stream);
        // Different channel counts / stride-2 can require a different row
        // alignment. Compute into a packed temporary only in those cases.
        std::shared_ptr<WorkingBuffer> packed_output;
        if (output_stride != y->stride)
            packed_output = buffer(size_t(output_stride) * ot * 2);
        std::shared_ptr<WorkingBuffer> padded;
        void* input = x->data();
        if (tp)
        {
            if (x->base && tp == 2)
                input = x->padded();
            else
            {
                padded = buffer(size_t(x->stride) * (x->t + tp) * 2);
                enqueue(vae::pad_time, x->data(), static_cast<half*>(padded->p), x->t, x->stride, tp, stream);
                input = padded->p;
            }
        }
        auto input_dims = x->dims(tp);
        input_dims[3] = physical_h;
        std::vector<int64_t> output_dims{1, d[0], ot, output_h, ow};
        std::vector<int64_t> output_strides{ot * output_stride, 1, output_stride, ow * d[0], d[0]};
        std::string key = Json({d, st, pads, input_dims, x->strides(tp), output_dims, output_strides}).dump();
        auto& plan = plans[key];
        if (!plan)
        {
            auto p = std::make_unique<Plan>();
            auto a = p->tensor("x", input_dims, x->strides(tp));
            auto w = p->tensor("w", d, {d[1] * d[2] * d[3] * d[4], 1, d[1] * d[3] * d[4], d[1] * d[4], d[1]});
            auto b = p->tensor("b", {1, d[0], 1, 1, 1}, {d[0], 1, d[0], d[0], d[0]});
            auto v = p->graph.conv_fprop(a, w,
                                         fe::graph::Conv_fprop_attributes()
                                             .set_pre_padding({0, pads[1], pads[2]})
                                             .set_post_padding({0, pads[4] + (down ? 1 : 0), pads[5] + (down ? 1 : 0)})
                                             .set_stride(st)
                                             .set_dilation({1, 1, 1}));
            auto out = p->graph.pointwise(v, b, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
            out->set_output(true).set_data_type(fe::DataType_t::HALF).set_dim(output_dims).set_stride(output_strides);
            p->args.push_back(out);
            try
            {
                p->build(handle, limit, autotune, tuning_cache, tuning_candidates);
            }
            catch (const std::exception& e)
            {
                throw std::runtime_error(op.at("name").get<std::string>() + ": " + e.what());
            }
            plan = std::move(p);
        }
        if (trace)
            std::cerr << "VAE conv executing " << op.at("name") << std::endl;
        workspace(*plan);
        enqueue_plan(*plan,
                     {input, weight(op, "weight"), weight(op, "bias"), packed_output ? packed_output->p : y->data()});
        if (packed_output)
            enqueue(vae::copy, static_cast<const half*>(packed_output->p), y->data(), ot, int(d[0]), oh * ow,
                    output_stride, y->stride, stream);
        dump(y, op.at("name"));
        return y;
    }
    TensorPtr attention(const TensorPtr& x, const Json& op)
    {
        if (std::getenv("SEEDVR_VAE_TRACE"))
            std::cerr << "VAE attention input=" << Json(x->dims()).dump() << std::endl;
        const auto& projections = op.at("projections");
        auto q = conv(x, projections[0]), k = conv(x, projections[1]), v = conv(x, projections[2]);
        auto y = tensor(x->c, x->t, x->h, x->w);
        int64_t seq = int64_t(x->h) * x->w;
        std::string key = "attention" + Json({x->t, x->c, seq, x->stride}).dump();
        auto& p = plans[key];
        if (!p)
        {
            auto plan = std::make_unique<Plan>();
            std::vector<int64_t> dims{x->t, 1, seq, x->c}, strides{x->stride, seq * x->c, x->c, 1};
            auto qa = plan->tensor("q", dims, strides), ka = plan->tensor("k", dims, strides),
                 va = plan->tensor("v", dims, strides);
            auto [out, stats] = plan->graph.sdpa(
                qa, ka, va,
                fe::graph::SDPA_attributes().set_generate_stats(false).set_attn_scale(1.f / std::sqrt(float(x->c))));
            out->set_output(true).set_dim(dims).set_stride(strides);
            plan->args.push_back(out);
            plan->build(handle, limit, autotune, tuning_cache, tuning_candidates);
            p = std::move(plan);
        }
        workspace(*p);
        enqueue_plan(*p, {q->data(), k->data(), v->data(), y->data()});
        q.reset();
        k.reset();
        v.reset();
        return conv(y, projections[3]);
    }
    void prepare_operations(bool encode, const void* input, void* output, int frames, int h, int w)
    {
        if (!input || !output || frames < 1 || frames > INT32_MAX / 128 || h < 1 || w < 1 ||
            int64_t(h) * w > INT32_MAX / (encode ? 32 : 2048))
            throw std::invalid_argument("Invalid native VAE input");
        if (encode && (frames % 4 != 1 || h % 8 || w % 8))
            throw std::invalid_argument("Encoder requires 4n+1 frames and spatial multiples of eight");
        if (!encode && (h > INT32_MAX / 8 || w > INT32_MAX / 8 || frames > INT32_MAX / 4))
            throw std::invalid_argument("Decoder dimensions overflow");
        image_mode = frames == 1;
        auto x = tensor(encode ? 3 : 16, frames, h, w);
        TensorPtr saved;
        enqueue(vae::boundary, static_cast<const half*>(input), x->data(), frames, x->c, h * w, x->stride, true,
                encode ? 0 : 2, stream);
        for (const auto& op : manifest.at("components").at(encode ? "encoder" : "decoder"))
        {
            const std::string type = op.at("op");
            if (type == "save")
                saved = x;
            else if (type == "conv")
                x = conv(x, op);
            else if (type == "norm")
            {
                auto y = x == saved ? tensor(x->c, x->t, x->h, x->w) : x;
                auto stats = buffer(size_t(x->t) * 32 * 2 * sizeof(float));
                enqueue(vae::norm, x->data(), y->data(), static_cast<float*>(stats->p), weight(op, "weight"),
                        weight(op, "bias"), x->t, x->c, x->h * x->w, x->stride, y->stride, op.at("silu").get<bool>(),
                        stream);
                x = y;
            }
            else if (type == "add")
            {
                if (!saved)
                    throw std::runtime_error("Residual without saved input");
                if (!op.at("shortcut").is_null())
                    saved = conv(saved, op.at("shortcut"));
                if (saved->dims() != x->dims())
                    throw std::runtime_error("Native VAE residual shape mismatch");
                enqueue(vae::residual, x->data(), saved->data(), x->t, x->c, x->h * x->w, x->stride, saved->stride,
                        stream);
                x = saved;
                saved.reset();
            }
            else if (type == "attention")
                x = attention(x, op);
            else if (type == "shuffle")
            {
                int rt = op.at("temporal"), c = x->c / (4 * rt);
                auto y = tensor(c, x->t * rt - (rt - 1), x->h * 2, x->w * 2);
                enqueue(vae::shuffle, x->data(), y->data(), x->t, c, x->h, x->w, x->stride, y->stride, rt, stream);
                x = y;
            }
            else
                throw std::runtime_error("Unknown VAE operation: " + type);
        }
        // Encoder conv_out has 32 moments; export selects the first 16 channels.
        if (encode)
        {
            auto selected = tensor(16, x->t, x->h, x->w);
            // cudaMemcpy2D copies the first half of each channels-last pixel.
            for (int t = 0; t < x->t; ++t)
                enqueue(cudaMemcpy2DAsync, selected->data() + t * selected->stride, 16 * 2, x->data() + t * x->stride,
                        32 * 2, 16 * 2, size_t(x->h) * x->w, cudaMemcpyDeviceToDevice, stream);
            x = selected;
        }
        enqueue(vae::boundary, x->data(), static_cast<half*>(output), x->t, x->c, x->h * x->w, x->stride, false,
                encode ? 1 : 3, stream);
    }
    static void validate_shape(bool encode, const NativeVae::Shape& shape)
    {
        const auto [frames, h, w] = shape;
        if (frames < 1 || frames > INT32_MAX / 128 || h < 1 || w < 1 ||
            int64_t(h) * w > INT32_MAX / (encode ? 32 : 2048) || (encode && (frames % 4 != 1 || h % 8 || w % 8)) ||
            (!encode && (h > INT32_MAX / 8 || w > INT32_MAX / 8 || frames > INT32_MAX / 4)))
            throw std::invalid_argument("Invalid fixed native VAE input shape");
    }
    std::unique_ptr<Session> prepare(bool encode, const NativeVae::Shape& shape)
    {
        const auto [frames, h, w] = shape;
        auto prepared = std::make_unique<Session>();
        prepared->input = std::make_unique<PersistentIO>(size_t(encode ? 3 : 16) * frames * h * w * 2, stream);
        const size_t output_elements = encode ? size_t(16) * ((frames + 3) / 4) * (h / 8) * (w / 8)
                                              : size_t(3) * (4 * frames - 3) * (h * 8) * (w * 8);
        prepared->output = std::make_unique<PersistentIO>(output_elements * 2, stream);
        active = prepared.get();
        prepare_operations(encode, active->input->p, active->output->p, frames, h, w);
        prepared->planner.finalize();
        for (auto& tensor : prepared->tensors)
        {
            tensor->offset = prepared->planner.placement(tensor->offset);
            tensor->view->relocate(tensor->offset);
        }
        for (auto& buffer : prepared->buffers)
        {
            if (!buffer->allocation_bytes)
                continue;
            buffer->offset = prepared->planner.placement(buffer->offset);
            buffer->view.relocate(buffer->offset);
        }
        // Scratch is live alongside every activation. Reserve beyond the peak
        // optimized activation extent, so no operation can alias its scratch.
        working_memory::Planner workspace;
        workspace.extent = prepared->planner.extent;
        prepared->scratch = std::make_unique<WorkingBuffer>(prepared->workspace_bytes, arena, workspace);
        prepared->working_bytes = workspace.extent;
        prepared->frozen = true;
        active = nullptr;
        return prepared;
    }
    void run(bool encode, const void* input, void* output)
    {
        auto* session = (encode ? encoder : decoder).get();
        if (!session)
            throw std::invalid_argument("Native VAE direction was not configured at construction");
        if (!input || !output)
            throw std::invalid_argument("Null native VAE I/O");
        if (session->failed)
            throw std::runtime_error("Native VAE session failed; recreate it before running again");
        arena.commit();  // No-op normally; remaps stable views after explicit release.
        active = session;
        try
        {
            SEEDVR_VAE_CHECK(
                cudaMemcpyAsync(active->input->p, input, active->input->bytes, cudaMemcpyDeviceToDevice, stream));
            if (active->executable)
                SEEDVR_VAE_CHECK(cudaGraphLaunch(active->executable, stream));
            else
            {
                // Plans, bindings and allocations already exist. Only library
                // runtime warmup and CUDA Graph capture remain on the first call.
                for (const auto& operation : active->operations)
                    operation();
                SEEDVR_VAE_CHECK(cudaGetLastError());
                SEEDVR_VAE_CHECK(cudaStreamSynchronize(stream));
                SEEDVR_VAE_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                capturing = true;
                try
                {
                    for (const auto& operation : active->operations)
                        operation();
                    SEEDVR_VAE_CHECK(cudaGetLastError());
                }
                catch (...)
                {
                    cudaGraph_t discarded = nullptr;
                    cudaStreamEndCapture(stream, &discarded);
                    if (discarded)
                        cudaGraphDestroy(discarded);
                    throw;
                }
                SEEDVR_VAE_CHECK(cudaStreamEndCapture(stream, &active->graph));
                capturing = false;
                SEEDVR_VAE_CHECK(cudaGraphInstantiate(&active->executable, active->graph, nullptr, nullptr, 0));
            }
            SEEDVR_VAE_CHECK(
                cudaMemcpyAsync(output, active->output->p, active->output->bytes, cudaMemcpyDeviceToDevice, stream));
            active = nullptr;
        }
        catch (...)
        {
            // Keep addresses owned until destruction, but prohibit replay of
            // a failed session. Never lazily rebuild fixed-shape state in Run.
            cudaStreamSynchronize(stream);
            capturing = false;
            session->failed = true;
            active = nullptr;
            throw;
        }
    }
};
NativeVae::NativeVae(const std::filesystem::path& p, cudaStream_t s, const Shapes& shapes, size_t l, bool tune,
                     const std::filesystem::path& cache, int candidates)
    : impl_(std::make_unique<Impl>(p, s, l, shapes, tune, cache, candidates))
{
}
NativeVae::~NativeVae() = default;
void NativeVae::Encode(const void* x, void* y)
{
    [[maybe_unused]] common::nvtx_scoped_range range{"SeedVR2 / VAE encoder / cuDNN"};
    impl_->run(true, x, y);
}
void NativeVae::Decode(const void* x, void* y)
{
    [[maybe_unused]] common::nvtx_scoped_range range{"SeedVR2 / VAE decoder / cuDNN"};
    impl_->run(false, x, y);
}
void NativeVae::ReleaseWorkingMemory()
{
    SEEDVR_VAE_CHECK(cudaStreamSynchronize(impl_->stream));
    impl_->arena.release();
}
void NativeVae::CommitWorkingMemory()
{
    impl_->arena.commit();
}
NativeVae::WorkingMemoryStats NativeVae::GetWorkingMemoryStats() const
{
    return {impl_->encoder ? impl_->encoder->working_bytes : 0,
            impl_->decoder ? impl_->decoder->working_bytes : 0,
            impl_->arena.capacity,
            impl_->arena.resident,
            impl_->encoder ? impl_->encoder->input->bytes + impl_->encoder->output->bytes : 0,
            impl_->decoder ? impl_->decoder->input->bytes + impl_->decoder->output->bytes : 0};
}
size_t NativeVae::PeakWorkingBytes() const
{
    return impl_->arena.peak;
}
}  // namespace din::seedvr

#undef SEEDVR_VAE_CHECK
