# Shared SAM frame inference

`din_sam_pipeline` provides one `din::sam::Sam` API for SAM2 and SAM3. The existing
SAM2 and SAM3 classes are compatibility adapters; inference and temporal state
updates live in this library. OFX keeps its frame scheduling, result caches and
offline mask persistence outside the inference core.

## Source layout

All SAM runtime implementations and export tools live here. `cpp/` contains the shared
API and family-specific adapters, kernels and tokenizer. `python/` contains both
exporters and the distinctly named `sam2_modeling.py` and `sam3_modeling.py`.
Only model exports, verification and their required helpers are published. Local
tests, fixtures, diagnostics and generated reports are excluded from Git. Export details are in [SAM2](docs/SAM2.md) and [SAM3](docs/SAM3.md).
Existing CMake target names and family namespaces are retained. Source-qualified
includes use `vision/sam/cpp/`; the old `vision/sam2` and `vision/sam3` trees have
been removed.

## Setup and models

Use the project's CLion CMake profile and configured `modelopt` Python interpreter.
The prepared Windows bundles are:

- `D:/models/sam2.1_base_plus_onnx`: existing FP16 graph interfaces.
- `D:/models/Sam3`: existing FP32 interfaces and internal BF16 attention.
- Corresponding directories with `_cpu_fp32` appended: synchronous FP32 CPU inference.

Both families require `attention_contract.version=1` and seven fixed spatial
memory inputs. SAM3 additionally requires sixteen pointer histories and
`memory_boundary_version=2`. These are the default canonical bundles. Previous deployments are preserved under
`D:/models/backups`; `D:/models/sam_bundle_promotion.json` records their exact location.

For new source exports, the model preparation helper reuses weights and existing native ONNX `Attention`
operators. It adds boolean history validity at attention normalization; it does
not replace attention with a quadratic score/mask graph or change precision.

```powershell
python vision/sam/python/prepare_static_sam.py sam2 path/to/unmodified_sam2_export D:/models/new_sam2_bundle
python vision/sam/python/prepare_static_sam.py sam3 path/to/corrected_unpadded_sam3_export D:/models/new_sam3_bundle
```

The original exporters also accept `--static-memory`. Keep the existing
BF16-attention conversion step for freshly exported SAM3 GPU attention. A plain
FP32 re-export is not equivalent to the deployed mixed-precision policy.

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

| Previous state | Behavior |
|---|---|
| `nullptr` | Initialize destination using the supplied `Initialization` |
| Same as destination | Propagate in place |
| A different leased state | Preserve previous state; copy required history and advance destination |

State references expire when their lease is released. `Sam` must outlive its
leases, prepared prompts and completion tokens. Calls on one instance must be
serialized; several video states can be interleaved on its bound stream.

Standard recurrent tensor storage is **7 MiB for SAM2 FP16**, **14 MiB for its
FP32 CPU reference**, and **324.25 MiB for SAM3**. SAM3 retains sixteen complete
entries, each with four `256x72x72` FP32 tensors and `16x256` FP32 object pointers.
Metadata is additional and small. Model weights, graph outputs, packed attention
workspace and the 81 MiB seed-prompt scratch are separate allocations.

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
config.model_dir = "D:/models/Sam3";
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

No pageable CPU flag is used for completion. Small diagnostics use separate
pinned asynchronous readback; `Ready()`/`Wait()` report detected model errors.
Do not re-record an event while a live token still represents its recording.
Holding all configured tokens reports exhaustion before submission instead of
allocating or waiting. Borrowed input/output allocations must remain alive through
their queued uses; SAM completion does not cover later downstream consumer work.

## Export and verification

See [SAM2](docs/SAM2.md) and [SAM3](docs/SAM3.md) for export commands. The Python
entry points are the two exporters and `verify_sam_onnx.py`; source preparation,
fixed-capacity conversion and attention casting support those export workflows.
Model wrappers, tokenization, antialias resize and ONNX lowering are required
export helpers rather than standalone development tools.

```powershell
python vision/sam/python/verify_sam_onnx.py D:/models/sam2.1_base_plus_onnx
python vision/sam/python/verify_sam_onnx.py D:/models/Sam3
```

The verifier checks every graph and external weight sidecar without changing the
bundle. For an FP32 CPU bundle with saved export fixtures, add `--fixtures` to
compare ONNX Runtime outputs with the saved PyTorch outputs. This is a separate
numerical check; structural verification alone does not establish model parity.
An optional `--report artifacts/sam/verification.json` saves the results.

Unit/integration tests, experiment scripts, profiling tools and validation
bookkeeping stay local and untracked. They are not required to export, verify
or build the published runtime. Existing preprocessing, precision-placement and
controller differences from native PyTorch remain separate investigations.
