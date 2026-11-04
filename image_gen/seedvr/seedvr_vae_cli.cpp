// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "seedvr_vae.h"
namespace
{
void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
struct Device
{
    void* p = nullptr;
    explicit Device(size_t n)
    {
        check(cudaMalloc(&p, n));
    }
    ~Device()
    {
        cudaFree(p);
    }
};
struct Stream
{
    cudaStream_t s = nullptr;
    Stream()
    {
        check(cudaStreamCreate(&s));
    }
    ~Stream()
    {
        cudaStreamDestroy(s);
    }
};
}  // namespace
int main(int argc, char** argv)
try
{
    if (argc < 9)
    {
        std::cerr << "Usage: din_seedvr2_vae_cli WEIGHTS encode|decode T H W input_fp16.bin output_fp16.bin "
                     "report.json [--verify-sharing] [--autotune] [--tuning-cache DIR] [--workspace-mib N] "
                     "[--iterations N]\n";
        return 2;
    }
    bool encode = std::string(argv[2]) == "encode";
    bool verify_sharing = false, autotune = false;
    int iterations = 1, workspace_mib = 256, tuning_candidates = 5;
    std::filesystem::path tuning_cache;
    for (int i = 9; i < argc; ++i)
    {
        const std::string option = argv[i];
        if (option == "--verify-sharing")
            verify_sharing = true;
        else if (option == "--autotune")
            autotune = true;
        else if (option == "--tuning-cache" && i + 1 < argc)
            tuning_cache = argv[++i];
        else if (option == "--workspace-mib" && i + 1 < argc)
            workspace_mib = std::stoi(argv[++i]);
        else if (option == "--iterations" && i + 1 < argc)
            iterations = std::stoi(argv[++i]);
        else if (option == "--tuning-candidates" && i + 1 < argc)
            tuning_candidates = std::stoi(argv[++i]);
        else
            throw std::invalid_argument("Unknown or incomplete option: " + option);
    }
    if (iterations < 1 || workspace_mib < 0 || tuning_candidates < 1)
        throw std::invalid_argument("Invalid iterations or workspace budget");
    if (verify_sharing && !encode)
        throw std::invalid_argument("--verify-sharing requires encode mode");
    if (!encode && std::string(argv[2]) != "decode")
        throw std::invalid_argument("Expected encode or decode");
    int t = std::stoi(argv[3]), h = std::stoi(argv[4]), w = std::stoi(argv[5]);
    if (t < 1 || h < 1 || w < 1 || int64_t(t) * h * w > INT32_MAX)
        throw std::invalid_argument("Invalid dimensions");
    if (encode && ((t - 1) % 4 || h % 8 || w % 8))
        throw std::invalid_argument("Encode requires 4n+1 frames and spatial multiples of eight");
    if (!encode && (t > INT32_MAX / 4 || h > INT32_MAX / 8 || w > INT32_MAX / 8))
        throw std::invalid_argument("Decoder dimensions overflow");
    size_t n = size_t(encode ? 3 : 16) * t * h * w * 2;
    size_t out = encode ? size_t(16) * ((t - 1) / 4 + 1) * (h / 8) * (w / 8) * 2
                        : size_t(3) * ((t - 1) * 4 + 1) * (h * 8) * (w * 8) * 2;
    std::ifstream f(argv[6], std::ios::binary | std::ios::ate);
    if (!f || size_t(f.tellg()) != n)
        throw std::runtime_error("Input file size mismatch");
    std::vector<char> host(n);
    f.seekg(0);
    f.read(host.data(), n);
    if (!f)
        throw std::runtime_error("Input read failed");
    Stream stream;
    Device x(n), y(out);
    check(cudaMemcpy(x.p, host.data(), n, cudaMemcpyHostToDevice));
    din::seedvr::NativeVae::Shapes shapes;
    (encode ? shapes.encoder : shapes.decoder) = din::seedvr::NativeVae::Shape{t, h, w};
    if (verify_sharing)
        shapes.decoder = din::seedvr::NativeVae::Shape{(t - 1) / 4 + 1, h / 8, w / 8};
    din::seedvr::NativeVae vae(argv[1], stream.s, shapes, size_t(workspace_mib) << 20, autotune, tuning_cache,
                               tuning_candidates);
    const auto memory = vae.GetWorkingMemoryStats();
    std::cout << "VAE encoder physical working memory required: " << memory.encoder_bytes << " bytes\n"
              << "VAE decoder physical working memory required: " << memory.decoder_bytes << " bytes\n"
              << "VAE shared physical working memory capacity (max, not sum): " << memory.shared_bytes << " bytes\n"
              << "VAE resident physical working memory: " << memory.resident_bytes << " bytes\n"
              << "VAE encoder persistent I/O (outside VMM): " << memory.encoder_io_bytes << " bytes\n"
              << "VAE decoder persistent I/O (outside VMM): " << memory.decoder_io_bytes << " bytes" << std::endl;
    auto run = [&]()
    {
        if (encode)
            vae.Encode(x.p, y.p);
        else
            vae.Decode(x.p, y.p);
    };
    run();
    cudaEvent_t a, b;
    check(cudaEventCreate(&a));
    check(cudaEventCreate(&b));
    std::vector<float> samples;
    float ms = 0;
    for (int i = 0; i < iterations; ++i)
    {
        check(cudaEventRecord(a, stream.s));
        run();
        check(cudaEventRecord(b, stream.s));
        check(cudaEventSynchronize(b));
        float sample;
        check(cudaEventElapsedTime(&sample, a, b));
        samples.push_back(sample);
        ms += sample;
    }
    ms /= iterations;
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    host.resize(out);
    check(cudaMemcpy(host.data(), y.p, out, cudaMemcpyDeviceToHost));
    if (verify_sharing)
    {
        const auto usage = vae.GetWorkingMemoryStats();
        if (usage.shared_bytes != std::max(usage.encoder_bytes, usage.decoder_bytes))
            throw std::runtime_error("Working memory is not max(encoder, decoder)");
        Device decoded(n), alternate_x(n), alternate_y(out);
        check(cudaMemcpy(alternate_x.p, x.p, n, cudaMemcpyDeviceToDevice));
        vae.Decode(y.p, decoded.p);  // Warm and capture decoder using the same backing.
        check(cudaStreamSynchronize(stream.s));
        std::vector<char> decoder_reference(n), actual(out), decoder_actual(n);
        check(cudaMemcpy(decoder_reference.data(), decoded.p, n, cudaMemcpyDeviceToHost));
        vae.Encode(alternate_x.p, alternate_y.p);  // Different caller addresses.
        check(cudaStreamSynchronize(stream.s));
        check(cudaMemcpy(actual.data(), alternate_y.p, out, cudaMemcpyDeviceToHost));
        if (actual != host)
            throw std::runtime_error("Encoder replay differs after decoder reused backing");
        vae.ReleaseWorkingMemory();
        if (vae.GetWorkingMemoryStats().resident_bytes != 0)
            throw std::runtime_error("Release did not discard physical backing");
        vae.CommitWorkingMemory();
        vae.Decode(alternate_y.p, decoded.p);  // Replay captured graph after remapping.
        vae.Encode(alternate_x.p, y.p);
        check(cudaStreamSynchronize(stream.s));
        check(cudaMemcpy(decoder_actual.data(), decoded.p, n, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(actual.data(), y.p, out, cudaMemcpyDeviceToHost));
        if (actual != host || decoder_actual != decoder_reference)
            throw std::runtime_error("Graph replay differs after VMM recommit");
        std::cout << "Shared backing, changed caller addresses, and graph replay after recommit passed\n";
    }
    std::ofstream output(argv[7], std::ios::binary);
    output.write(host.data(), out);
    if (!output)
        throw std::runtime_error("Output write failed");
    std::ofstream report(argv[8]);
    const auto usage = vae.GetWorkingMemoryStats();
    report << "{\"gpu_elapsed_ms\":" << ms << ",\"working_bytes\":" << vae.PeakWorkingBytes()
           << ",\"input_bytes\":" << n << ",\"output_bytes\":" << out
           << ",\"encoder_working_bytes\":" << usage.encoder_bytes
           << ",\"decoder_working_bytes\":" << usage.decoder_bytes << ",\"shared_working_bytes\":" << usage.shared_bytes
           << ",\"resident_working_bytes\":" << usage.resident_bytes
           << ",\"encoder_persistent_io_bytes\":" << usage.encoder_io_bytes
           << ",\"decoder_persistent_io_bytes\":" << usage.decoder_io_bytes
           << ",\"sharing_recommit_verified\":" << (verify_sharing ? "true" : "false")
           << ",\"autotune\":" << (autotune ? "true" : "false") << ",\"samples_ms\":[";
    for (size_t i = 0; i < samples.size(); ++i)
    {
        if (i)
            report << ',';
        report << samples[i];
    }
    report << "]}\n";
    if (!report)
        throw std::runtime_error("Report write failed");
    std::cout << "Native VAE " << argv[2] << ": " << ms << " ms\n";
    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "Native VAE: " << e.what() << '\n';
    return 1;
}
