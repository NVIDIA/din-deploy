# Shared SAM frame inference

`din_sam_pipeline` provides one `din::sam::Sam` API for SAM2 and SAM3. The existing
SAM2 and SAM3 classes are compatibility adapters; inference and temporal state
updates are abstractes inside these classes providing state handles to the interface.

## Source layout

All SAM runtime implementations and export tools live here. `cpp/` contains the shared
API and family-specific adapters, kernels and tokenizer. `python/` contains both
exporters and minimal verification of the ONNX outputs in python.
Model details are in [SAM2](docs/SAM2.md) and [SAM3](docs/SAM3.md).

## Command-line inference

Build `din_sam_cli` for both model families. The first positional argument is a PNG
file or a frame folder sorted by filename; encoded video must first be decoded into
PNG frames. The second positional argument is the output directory for all results.

```powershell
din_sam_cli input.png masks --model sam2 --model-dir D:/models/sam2.1_base_plus_onnx --prompt prompt.json
din_sam_cli frames masks --model sam3 --model-dir D:/models/sam3_onnx --text person
```

`--model` and `--model-dir` are required.
SAM2 requires `--prompt` with point/box JSON; SAM3 requires `--text`.
`--tracking temporal|none` controls state propagation: `temporal` is the default and
propagates recurrent state across frame folders; `none` segments each frame
independently.
`--max-frames N` limits a folder to its first N frames (0 means all).
`--prompt` and `--text` cannot be combined.

Both families write all results to the required output directory. `run.json` is
always written there, and each valid instance writes one binary mask PNG named
`{frame:06d}_idN.png` by default; `--no-dump-masks` skips the PNGs. SAM2 produces
at most one instance per frame.

`--provider cpu|trt-rtx` and `--device N` configure the shared runtime. CPU inference
requires FP32 models; GPU model I/O supports FP32, FP16 and BF16.
Timing is collected automatically for both families.
`run.json` and console output separate session loading, prompt setup, mean inference latency, and total processing
time/FPS.
Session loading includes the runtime's internal warmup and CUDA graph setup, so the first measured frame is already at
steady state.
Per-frame records contain loading, inference, output snapshot/writing time and writer wait time.
Inference includes upload, mask resize/download and completion, even with `--no-dump-masks`.
PNG writing runs through one `std::async` task at a time with an owned result snapshot,
overlapping the next frame. Write failures propagate, and all writes finish before `run.json`.
Processing FPS includes every PNG decode and frame I/O, excludes setup, output snapshot/submission,
writer waits and report writing, and counts each source frame once. `wall_seconds` and `wall_fps`
include output waits and the final writer drain. Worker write time overlaps processing, so
per-frame `output_seconds` cannot be added to inference time to recover elapsed time.
The initial image size probe belongs to `input_setup_seconds`.
PNG decoding converts into the runtime's reusable ORT host-accessible upload buffer;
the decoder retains internal scratch storage. GPU uploads reuse that buffer without
an additional RGB staging copy. `HostInputBuffer()` waits for prior stream work before reuse.
SAM tensor storage inherits `common::TensorBufferBase`, also used by `common::TensorBuffer<T>`.
It preserves runtime dtypes, reserved workspace capacity and stable bindings, using the EP's
`HOST_ACCESSIBLE` allocation directly on detected UMA devices and `DEFAULT` storage with
host staging on discrete GPUs. CUDA kernels use the resolved execution pointer.
For GPU inference, host mask outputs use the same CUDA resize/conversion kernel as
device outputs. Reusable common tensor storage provides direct host access on UMA
or a readback on discrete GPUs, followed by a copy into the caller's output view.
`sam.Output.ResizeAndWriteHost` runs only for CPU inference; GPU traces separate
`ResizeAndWriteGpu`, `Download`, `WaitForHost` and `CopyToHostView` under `sam.Output`.
The host-output API still waits for GPU completion before returning readable pixels.
These are host measurements, not per-graph GPU timings; the old
SAM3 `--profile` option is replaced by this common report.

With `DIN_ENABLE_NVTX=ON`, `sam.onnx.<model>` ranges identify each ONNX session run
for both families. Separate `sam.Input.*`, `sam.Output.*`, `sam.Setup.*`,
`sam.Warmup` and `sam.cli.*` ranges distinguish transfers, host mask resizing,
initialization, decoding and background PNG writing. Attribute asynchronous CUDA
activity through its launching API call and range, rather than GPU timeline proximity.

SAM3 retains `--threshold` for detection confidence. Mask probabilities use a
fixed 0.5 cutoff for both families.

## Setup and models

Use a configured CMake build directory for your platform and a Python environment
with the family export dependencies. The model paths below are explicit examples;
the CLI has no default model family or model directory.

The native GPU runtime accepts SAM2 and SAM3.1 bundles with `dtype=float32`,
`dtype=float16` or `dtype=bfloat16`. Model tensors and recurrent history retain
that storage precision; CUDA kernels accumulate sensitive arithmetic in FP32.
The public mask/output API retains its existing types. The existing example
bundles are `D:/models/sam2.1_base_plus_onnx` and `D:/models/sam3_onnx`.
CPU inference requires FP32: export with `--dtype float32 --cpu-compatible`
and pass `--provider cpu` to the CLI. FP8 encoder bundles keep BF16 interfaces
and use the same GPU I/O path. TensorRT operator support is separate from I/O:
TensorRT-RTX 1.8.2 currently fails to compile the BF16 transposed convolutions
in the SAM3 vision encoder and SAM2 decoder. The runtime reports that engine
build failure rather than changing graph precision or silently falling back.

Both families require `attention_contract.version=1` and seven fixed spatial
memory inputs. SAM3 additionally requires sixteen pointer histories and
`memory_boundary_version=2`. Pass `--static-memory` when exporting GPU bundles
for this runtime; the exporter flag is off by default. `--cpu-compatible` implies
`--static-memory` and requires `--dtype float32`.

Export final bundles directly with native ONNX `Attention` operators and boolean
history validity inputs:

```powershell
python vision/sam/python/export_sam2_onnx.py --model facebook/sam2.1-hiera-base-plus --device cuda --static-memory --output D:/models/new_sam2_bundle
python vision/sam/python/export_sam3_onnx.py --static-memory --output D:/models/new_sam3_bundle
```

Both exporters accept `--static-memory` and default to BF16 (`--dtype bfloat16`),
the preferred GPU precision. Full BF16 exports use
`D:/models/sam2.1_base_plus_bf16` and `D:/models/sam3.1_bf16`. FP32 is required
for CPU inference and numerical references (`--dtype float32`); FP16 (`--dtype float16`) comes with quality tradeoffs.
Neither exporter inserts
attention-only precision casts into other precisions.

The installed ORT CPU attention implementation requires an explicit query axis
in its mask. `--cpu-compatible` creates a CPU-only compatibility variant using
128-query chunks, rather than expanding a complete Q-by-K mask. It preserves
FP32 arithmetic and uses standard ONNX operations. GPU bundles retain the
unmodified fused attention structure.

## State and lifetime

Construction reserves one state and eight completion slots by default. It loads
all graph sessions, binds persistent tensors, allocates maximum-size source
scratch and pinned staging, warms inference/kernels, and prepares graph replay.
Set maximum source geometry before construction. Prepare prompts during setup.

`AcquireState()` returns a move-only lease without allocating. Its destructor or
`ReleaseState(std::move(lease))` returns the slot; GPU storage remains reserved.
`ReserveStates(n)` explicitly grows capacity without relocating existing state
objects. It may allocate and is a setup operation.

The destination state is explicit:

| Previous state           | Behavior                                                               |
|--------------------------|------------------------------------------------------------------------|
| `nullptr`                | Initialize destination using the supplied `Initialization`             |
| Same as destination      | Propagate in place                                                     |
| A different leased state | Preserve previous state; copy required history and advance destination |

State references expire when their lease is released. `Sam` must outlive its
leases, prepared prompts and completion tokens. Calls on one instance must be
serialized; several video states can be interleaved on its bound stream.

Standard recurrent tensor storage is **7 MiB for SAM2 FP16/BF16** and **14 MiB for
FP32**. SAM3 uses **162.125 MiB for FP16/BF16** or **324.25 MiB for FP32**. It
retains sixteen entries, each with four `256x72x72` tensors and `16x256` object
pointers in the model's precision.
Metadata is additional and small. Model weights, graph outputs, packed attention
workspace and seed-prompt scratch (81 MiB for FP32 or 40.5 MiB for FP16/BF16) are
separate allocations.

SAM3 stores the seed position, sequence step and history ages. Sequential users
do not supply frame indices. Supply the sequence length at initialization; pointer
temporal normalization retains the current `max(1, min(16, length)-1)` denominator.
Empty initial frames advance sequence time and retry detection. While confirmation
is pending, GPU flags select initialization versus propagation candidates without
a host wait. After confirmation, inference submits propagation only. Late arrivals,
association and reconditioning remain outside the current controller's scope.

## GPU video example

The decoder and consumer below represent application code. Their allocations
remain alive and their work is ordered on the same nonblocking CUDA stream.

```cpp
#include "sam.h"
#include <cuda_runtime_api.h>
using namespace din::sam;

SamConfig config;
config.model = ModelFamily::Sam3;
config.model_dir = "D:/models/sam3_onnx";
config.cache_dir = "sam-runtime-cache";
config.device = deviceOrdinal;
config.stream = videoStream; // Borrowed; must outlive Sam.
config.max_source_width = 1920;
config.max_source_height = 1080;
Sam sam(config);

auto prompt = sam.PreparePrompt({.text = "person"});
Initialization init{prompt, videoFrameCount, true};
auto state = sam.AcquireState();

// Allocated by the application once: 16 x 288 x 288 FP32 logits.
MaskOutputView masks{deviceMaskTensor, 288, 288};
masks.device = deviceOrdinal;
masks.validity = deviceValidity;       // Optional float[16].
masks.scores = deviceScores;           // Optional float[16], probabilities.
masks.ids = deviceObjectIds;           // Optional uint32_t[16].
masks.presence_logits = deviceLogits;  // Optional raw float[16].

bool first = true;
while (decoder.HasFrame()) {
    FrameView frame = decoder.DecodeOn(videoStream);
    auto submitted = sam.RunFrame(
        frame, masks, *state,
        first ? nullptr : state.get(), first ? &init : nullptr);
    consumer.Enqueue(frame, masks, videoStream);
    first = false;
    // Releasing this token does not wait. The stream orders buffer reuse.
}
cudaStreamSynchronize(videoStream); // Application's final completion boundary.
```

Device RGB8 is top-down by default and may be pitched. Float RGBA supports an
explicit origin, signed row stride and premultiplied alpha. SAM2 prompts contain
coordinates in its 1024-pixel model coordinate space and exactly the exported
point count; the existing `LoadPrompt` adapter performs source-coordinate scaling.

Output planes are instance-major, with configurable byte strides. Zero strides
mean contiguous storage. Select logits, probabilities, 0/1 binary masks or UInt32
labels. `selected_id=0` writes every instance, while a positive ID writes one
plane. Labels use one plane and choose the highest probability above 0.5, breaking
ties by the lowest ID. Output metadata arrays share the output's memory location.

Raw device calls enqueue preprocessing, inference, state updates and mask output
without host synchronization or frame/mask downloads. Host RGB inputs copy through
preallocated pinned staging; host-output adapters deliberately wait and convert
downloaded native masks. CPU inference is synchronous. These convenience paths
are not the asynchronous device-path performance contract.

Preprocessing writes a stable internal model tensor; caller device image
addresses may vary. Postprocessing writes directly to caller mask allocations.
Graph bindings stay fixed. `cuda_graph=false` selects warmed eager execution for
comparison/debugging.

To retain a checkpoint without copying it immediately:

```cpp
sam.ReserveStates(2); // Setup operation; reserve before the video loop.
auto checkpoint = std::move(state);
state = sam.AcquireState();
auto submitted = sam.RunFrame(nextFrame, masks, *state, checkpoint.get());
// checkpoint still represents the earlier frame; state represents nextFrame.
```

## Completion

`FrameSubmission::completion` leases a timing-disabled CUDA event recorded after
state/output writes. `Ready()` queries it, `WaitOn(otherStream)` inserts a GPU
dependency, and only `Wait()` explicitly blocks the CPU. CPU tokens are already
complete. Producer work on a different stream must be ordered before `Sam::Stream()`
using an event; consumers can use the returned token's `WaitOn` method.

No pageable CPU flag is used for completion. Detected model errors use a separate
pinned asynchronous readback reported by `Ready()`/`Wait()`.
Do not re-record an event while a live token still represents its recording.
Holding all configured tokens reports exhaustion before submission instead of
allocating or waiting. Borrowed input/output allocations must remain alive through
their queued uses; SAM completion does not cover later downstream consumer work.

## Export and verification

See [SAM2](docs/SAM2.md) and [SAM3](docs/SAM3.md) for export commands. The Python
entry points are the two exporters, `prepare_sam3.py` for checkpoint
preparation, `quantize_sam_encoder_fp8.py`, `patch_sam_encoder_fp8_qdq.py`,
and the separate `verify_sam2.py` / `verify_sam3.py` entry points. Exporters produce fixed-capacity GPU or
CPU-compatible bundles directly.
The shared FP8 commands restore SAM2 calibration and patching lost during
consolidation and extend the same dataset-driven ModelOpt workflow to SAM3.
See each variant guide for calibration, bundle patching and verification commands.
Model wrappers, tokenization, antialias resize and ONNX lowering are required
export helpers rather than standalone development tools.

```powershell
python vision/sam/python/verify_sam2.py --onnx-dir models/sam2/ --image input.png --prompts prompt.json --output sam2/verify_image
python vision/sam/python/verify_sam3.py --onnx-dir models/sam3/ --image input.png --text person --output sam3/verify_image
```

Both verifiers retain the image/sequence structure of SAM2's upstream reference
runner. Each pipeline component wraps the same modeling class used by the exporter,
with either an `export` (PyTorch) or `onnx` backend. By default every component in
the candidate runs in ONNX Runtime, compared with the all-PyTorch export baseline.
`--onnx-components image_encoder` (SAM2) or `--onnx-components
vision_encoder` (SAM3) replaces just that component; list multiple names to replace
several. `--onnx-components all` explicitly selects the default; use
`--onnx-components none` for an all-PyTorch candidate.
Only selected ONNX files are required or executed; unselected components need no
ORT session. `--help` lists the component names, and components unused by the
selected image/sequence mode are rejected.

Selected ONNX components use TensorRT RTX EP by default (`--provider cpu` selects
FP32 CPU-compatible graphs). Every component returns the same named tensors for
both backends, so the pipeline and state handling are shared. Fixed-capacity ONNX
attention gets its padding/validity inputs inside the component adapter; PyTorch
uses natural history lengths. The verifier does not call the C++ CLI unless
`--din-sam path/to/din_sam_cli` is supplied. Plugin loading supports the Python EP
package or `--ep-lib`, with optional `--ep-dll-dir` and `--trt-bin` on Windows.
CPU tensor staging supports BF16 graph I/O; these are correctness checks, not benchmarks.

Use `--frames-dir frames --limit-frames 8` instead of `--image` for tracking.
SAM2 accepts the original point/box prompt JSON, defaults `--model` from bundle
metadata, and retains `SAM2ImagePredictor` / `build_sam2_video_predictor_hf` for
the reference. Its current candidate supports single-mask inference and one
tracking object prompted at frame zero. SAM3 uses its pinned image processor
and causal video-session API, with at most sixteen instances.

Each invocation runs the original upstream predictor (`pytorch/`), the component
pipeline entirely in PyTorch (`export/`), and that same pipeline with the requested
ONNX substitutions (`candidate/`). State and text caches are reset between the
last two runs. Optional C++ results go into `native/`.
`comparison.json` includes per-frame agreement with upstream and with the export
baseline, selected backends, component-call counts, and per-output numerical checks
of selected ONNX components against their PyTorch wrappers on the same logical inputs.
Default thresholds are `--min-iou 0.95 --atol 0.002 --rtol 0.002`; reduced precision
can fail strict tensor tolerances even when binary masks agree. Exit code is zero
only when all requested comparisons pass. Missing frames/instances and runtime
errors fail verification; two empty instance sets match. Tracking keeps a stable
instance-ID mapping across frames, so per-frame matching cannot hide ID switches.

The report explicitly lists upstream parity gaps. SAM2 exports omit IoU predictions
and object pointers and use a seven-slot memory ring. SAM3's bounded native
controller omits upstream association/reconditioning; its official video reference
uses BF16 autocast while the two component pipeline runs use bundle precision. Neither difference
is hidden by changing the independent upstream reference. Full controller parity
requires runtime/export work beyond these verifiers. `--din-sam` compares native
masks with the selected component pipeline, also exercising native preprocessing and state handling.
Use a fresh output directory for tracking and native comparisons.

The old `verify_sam_onnx.py` is replaced. For structure-only checking, use either
family entry point with `--validate-only --onnx-dir ... --output ...`. This checks
every graph and external sidecar without changing the bundle; `--graphs name
[name ...]` selects a subset. Fixture generation and comparison are optional:
SAM3 export saves NPZ inputs/outputs only with `--fixtures` (or regenerates them
with `--fixtures-only`); verification reads them only with `--validate-only --fixtures`.
The normal export produces deployment artifacts, and normal image/sequence
verification produces masks for visual inspection without saved export fixtures.
SAM2's exporter does not save those fixtures.
Reports and caches must be outside the model directory.

Generated fixtures, reports, profiling output and other experiment scripts remain
local and untracked.
