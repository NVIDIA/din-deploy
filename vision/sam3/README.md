# SAM3.1 export and native inference

The current default bundle is the **memory-alias-corrected mixed-BF16** set in
`artifacts/sam3/onnx-bf16`, mirrored to `D:/models/Sam3`. Both contain the same nine
runtime graphs. The corrected default reproduced the validated 81-frame diagnostic
run; deployed runtime assets were SHA-256 verified. Remaining resize, seed-memory,
and controller limitations described below still apply.

Superseded SAM3 FP32/FP16 exports, diagnostic ONNX copies, unused auxiliary ONNX,
and generated ONNX cache contexts were removed at the user's request. Historical
reports and fixtures remain. Commands below referencing those old model directories
require regenerating the exports first. Promotion hashes and the cleanup audit are
in `artifacts/sam3/promotion/`.

This module provides pinned SAM3.1 reference/export tools and a native text-prompted
image segmentation pipeline and experimental causal multiplex tracking. **The
requested native video implementation is not complete.** Sequence reset, sequential
host/device frames, seed decoding, memory attention, temporal positions and bounded
GPU memory history are implemented. Detection association, suppression, periodic
reconditioning and object lifecycle remain incomplete. Detections currently seed
the first nonempty frame; subsequent detections do not update track state. OFX has
not been changed.

The CLI reuses SAM2's `ResolveInputFrames` for ordered frame-folder input. Both
pipelines use the common ONNX Runtime/TensorRT RTX sessions, device tensor buffers
and CUDA stream infrastructure. SAM2's model graphs and single-object memory state
are incompatible with SAM3's 256-channel, 16-object multiplex state.

```powershell
out/build/windows-x64/bin/Release/din_sam3_cli.exe V1-0022_frames artifacts/sam3/V1-0022_hair_cpp_tracking --text hair --track
```

`--framewise` instead runs independent image inference on every frame. It does not
preserve object identity. `--track` requires `temporal_constants.json` from the
exporter, prohibits repeated inference within a sequence, and is explicitly
experimental. `--max-frames` limits either folder mode. `ResetSequence` takes the
concept and total frame count because upstream pointer positions depend on clip
length. Image dimensions must remain fixed during a sequence.

The native default is now `artifacts/sam3/onnx-bf16`, selected by the user on
2026-10-30. This is mixed BF16: text/vision encoders use BF16 with FP32 exceptions;
detection, tracking and memory graphs remain FP32. Select `--model-dir
artifacts/sam3/onnx-fp32` after regenerating that previous baseline. Official PyTorch BF16 video
output is the accuracy reference; native FP32 is only a diagnostic comparison.
Selecting BF16 as the default does not mean upstream video parity has passed.

The native FP32 tracking CLI processed all 95 `V1-0022_frames` with the concept
`hair`, returning one instance with ID 1 throughout. Cached initialization took
2.616 s and mean inference took 0.699 s/frame. Mask/label export consistency passed.
Against the official BF16 video reference, mean matched IoU was 0.91676 and minimum
IoU was 0.63434; only one frame met 0.99. **Video parity therefore fails.** These are
different precision paths and the native controller lacks reconditioning, so the
comparison does not isolate a single cause. Native two-frame bicycle reset/replay,
empty sequence and sequence-boundary rejection tests pass.

## Setup and provenance

Run commands from the repository root with the project's configured Python
interpreter and CMake profile. Development configure/build/run commands on Windows
were executed through CLion MCP using `D:\repos\din-deploy`.

`upstream.json` pins both the official source commit and the checkpoint revision.
`prepare_sam3.py` uses `huggingface_hub` authentication and its configured global
cache. On this machine that is `D:\hf\hub`; no checkpoint is copied into the repo.
The source archive and generated artifacts live under ignored `artifacts/sam3/`.

The `sam3` Python extra in the root `pyproject.toml` lists the additional reference
dependencies. The existing interpreter needs PyTorch with CUDA, torchvision,
ONNX, ONNX Script and ONNX Runtime as well. Optional FP16 conversion uses NVIDIA
ModelOpt. `setuptools<81` supplies `pkg_resources`, which this upstream revision
imports. The tested environment uses PyTorch `2.14.0.dev20260625+cu132` and an RTX
PRO 4000 Blackwell GPU. Export metadata records the installed package versions.
The native tokenizer links Windows ICU; other platforms require ICU development
libraries discoverable by CMake.

```powershell
python vision/sam3/python/prepare_sam3.py
python vision/sam3/python/export_sam3_onnx.py --output artifacts/sam3/onnx-fp32
```

The exporter defaults to ONNX opset 23 with Dynamo, retaining native `Attention`
operators. It explicitly expands broadcast attention masks along the query axis
for ONNX Runtime CPU compatibility. Use `--opset 20` to reproduce decomposed
attention exports. Export into a new directory when comparing against existing
bundles; changing the exporter does not update deployed model files.

TensorRT RTX 1.7.1 may reject native FP32 attention when no dedicated kernel is
available. To evaluate BF16 attention separately, preserving graph interfaces and
the precision of surrounding operators, use a separate output directory:

```powershell
python vision/sam3/python/cast_sam3_attention.py path/to/opset23-bundle path/to/attention-bf16-bundle
```

This changes attention precision and requires numerical validation; performance
differences are not attributable to the opset change alone.

For gated access errors, obtain access on the
[official model page](https://huggingface.co/facebook/sam3.1) and run `hf auth login`.
The configured account successfully downloaded the pinned 3.50 GB checkpoint.

## Reference and export contracts

```powershell
python vision/sam3/python/run_sam3_reference.py vision/sam2/bike.png --text bicycle --output artifacts/sam3/reference_image
python vision/sam3/python/run_sam3_reference.py path/to/jpeg_frames --video --text bicycle --output artifacts/sam3/reference_video
```

Video reference inference uses the official multiplex orchestrator, forward
propagation, a grounding batch size of one, and hotstart delay zero. The upstream
orchestrator uses BF16 memory/autocast; its metadata explicitly records BF16.
The image reference and tensor graph exports use FP32 with TF32 disabled.

The graph set separates image preprocessing, text encoding, the shared vision
encoder, concept detection, seed decoding, multiplex tracking, memory encoding,
and memory attention. Seed and propagation memory have separate wrappers because
their mask inputs differ. Vision input is 1008×1008, stride 14; seed masks are
1152×1152 and tracking masks are 288×288. There is one 16-slot multiplex bucket.
`metadata.json` records actual ONNX interfaces, fixture dimensions, source and
checkpoint revisions, preprocessing, tokenizer assets, precision, and upstream
tracking parameters. Exported tracking boundaries do not establish controller
parity.

Adapters are scoped to the pinned implementation: they avoid double checkpoint
loading, check learned-weight coverage, retain standard attention/MLP operations,
handle the unsupported state-offload argument, lower constant sequence accesses,
and replace fixed antialiased resizes with separable tensor filters. Original
upstream source files are not patched. Native GPU image preprocessing uses the
equivalent CUDA filter because TensorRT RTX does not accept antialiased ONNX Resize.

## Native image CLI and API

```powershell
cmake --build out/build/windows-x64 --config Release --target din_sam3_cli din_sam3_pipeline_tests din_sam3_graph_tests din_sam3_tokenizer_tests
out/build/windows-x64/bin/Release/din_sam3_cli.exe vision/sam2/bike.png artifacts/sam3/result --text bicycle --model-dir artifacts/sam3/onnx-fp32 --warmup 2 --repeat 5
```

Use `--provider cpu` for the ONNX Runtime CPU comparison. The default provider is
`trt-rtx`, with CPU fallback disabled by the common runtime. `--threshold` defaults
to 0.5. More than 16 accepted detections is an explicit error; results are not
silently truncated. `--ep-cache` selects the runtime cache directory.

`Sam3Pipeline::SegmentImage` accepts host RGB8 data. `SegmentDeviceFrame` accepts a
borrowed pitched CUDA RGB8 allocation plus its producer stream; the allocation must
remain valid until the method returns. It synchronizes the producer with the
pipeline stream. Graph connections stay on the GPU. Only small detection metadata
and selected final masks are downloaded. The pipeline caches the last concept's
text embeddings. A pipeline instance is intended for one calling thread.

Each result contains IDs, scores, pixel-coordinate XYXY boxes, and full-resolution
mask probabilities. IDs start at one for each image; they are **not video IDs**.
Outputs are:

- `000000_combined.png`: binary union of instance masks.
- `000000_id<ID>.png`: individual binary masks, thresholded at probability >0.5.
- `000000_labels.npy`: little-endian uint32 labels, with zero for background.
- `000000.json`: instance IDs, scores, boxes, and mask filenames.
- `run.json`: initialization time, per-run inference times, warmup count and settings.

Overlapping labels select the greatest mask probability. Exact ties select the
lowest object ID. Saving files is excluded from the CLI's inference timing.

## Validation

For timing attribution, add `--profile` to the native CLI. `run.json` then includes
`stage_wall_ms` for each inference call. `graph.*` includes ORT binding and host
enqueue time; it no longer waits for GPU completion. `segment_total` includes its vision/detector graphs;
`memory_preparation` includes allocation, spatial packing and pointer preparation.
These nested totals must not be summed together. PNG decode/encode is outside the
CLI's inference timer. Profiling does not enable CUDA graphs. On GPU it also
records `gpu_stream.graph.*` with CUDA events, read after the frame's necessary
CPU readback has completed, without per-session event waits. Stream elapsed time includes any launch gaps;
it is not the sum of individual kernel durations. `bind.*`, `run.*`, and `sync.*`
are nested diagnostic portions of each graph wall time (the former `sync.*`
session measurements are no longer emitted for GPU execution).

GPU sessions share a compute stream and disable ORT's end-of-run synchronization.
Dependent kernels and sessions enqueue in stream order. Pinned readbacks wait on
completion events only where CPU scores, probabilities or export rasters are
needed. Detector scores/boxes/presence share one readback batch. Tracking keeps
detector masks on device, avoiding full-resolution detector-mask downloads.
The CLI enables CUDA rasterization for individual binary masks, combined mattes,
and probability/ID-ranked label maps. PNG encoding and file writes stay on CPU.
Tracking mask bounds and probability validation also run on CUDA.
The host API still returns probability vectors; `Labels()` and `CombinedMask()`
remain CPU utilities for arbitrary caller-owned results. GPU export rasters are
available separately through `LastRasterizedMasks()` until the next inference.

The native CLI retains graph output allocations, preprocessing/mask scratch,
and maximum-capacity temporal input buffers. Historical features and memory are
copied device-to-device into a bounded ring before graph outputs are reused.
Necessary score and final-mask downloads use reusable pinned host staging.
The two unconsumed inputs `detector_position_0` and `detector_position_1` in the
pinned detector export are deliberately not bound: ORT otherwise downloads
106,168,320 bytes of unused positions per frame. Both native SAM3 paths enable
the TensorRT RTX provider's `nv_persistent_context_memory` option to retain
execution workspace. Memory-attention shapes still grow during sequence startup;
persistent addresses do not make those different shapes equivalent.

For Nsight captures, use `--trace=cuda,nvtx --cuda-memory-usage=true
--cuda-event-trace=true --sample=process-tree --cpuctxsw=process-tree`.
On Windows, run Nsight elevated and inspect its log: without administrator
privileges it silently disables requested CPU sampling after a warning.
`--cuda-graph-trace=node` additionally exposes kernels inside CUDA graphs.
Profiled timings include instrumentation overhead; benchmark separately.

With `DIN_ENABLE_NVTX=ON`, SAM3 emits `sam3.*` NVTX ranges even without
`--profile`. `sam3.frame.<index>` groups CLI inference and result export;
`sam3.graph.<model>` identifies each model session, and `sam3.kernel.<name>`
marks each custom CUDA kernel launch. There are no nested memory-operation,
binding, synchronization, file-I/O, or postprocessing-phase ranges. The device
API uses `sam3.gpu.frame.<index>` for submitted frames. These are host ranges:
asynchronous work may finish after a range ends. Correlate them with the CUDA
timeline; instrumentation does not add synchronization or change graph capture.

`--diagnostic-dir` also writes `memory_attention_bindings.jsonl`, recording
input/output shapes and buffer addresses for each attention call. Use this in a
separate diagnostic run to distinguish shape changes from pointer churn without
adding disk I/O to the profiled session ranges.

`python/benchmark_sam3_rtx.py` benchmarks the FP32 graph set serially with an
explicit `--executable` path. It saves the exact commands, input fixtures, logs,
timing JSON and layer profiles. Memory-attention cases cover one spatial memory
with 16 pointer tokens and seven spatial memories with 256 pointer tokens, using
the native pipeline's dynamic shape bounds. Benchmarks disable CUDA graphs except
for separately named comparison cases. They exclude DMA and use eager shape
specialization after warmup; they are steady-state component timings, not frame
latencies or an accuracy validation.

```powershell
ctest --test-dir out/build/windows-x64 -C Release -R din_sam3 --output-on-failure
out/build/windows-x64/bin/Release/din_sam3_pipeline_tests.exe artifacts/sam3/onnx-fp32 vision/sam2/bike.png
python vision/sam3/python/verify_sam3_native_graphs.py out/build/windows-x64/bin/Release/din_sam3_graph_tests.exe artifacts/sam3/onnx-fp32 --provider cpu
python vision/sam3/python/verify_sam3_native_graphs.py out/build/windows-x64/bin/Release/din_sam3_graph_tests.exe artifacts/sam3/onnx-fp32 --provider trt-rtx
python vision/sam3/python/compare_sam3_masks.py artifacts/sam3/reference_image artifacts/sam3/result --min-iou .99 --report artifacts/sam3/mask_parity.json
```

The native graph harness uses the same C++ ONNX Runtime/TensorRT RTX integration as
the pipeline and saved PyTorch fixtures. It preserves diagnostic logs and numerical
error reports. Its elementwise tolerance is `atol=0.002, rtol=0.002`; a successful
end-to-end mask check does not imply every graph satisfies this stricter diagnostic.
The optional Python graph checker uses the interpreter's separate ONNX Runtime
installation. Regenerate realistic fixtures with `export_sam3_onnx.py
--fixtures-only --fixture-image <image>` before mixed-precision conversion.

Observed checks:

- Three bicycle instances pass native TensorRT RTX versus the FP32 image reference
  at IoU 0.99922–0.99980. The CPU image comparison also exceeds 0.99.
- Mixed-FP16 vision/text encoders with an FP32 detector pass the same fixture at
  matched IoU 0.99232–0.99901. Warm mean inference is 0.194 s; the first initialization
  with a new engine cache took 49.73 s. This validates one image fixture, not video.
- CUDA pitched device input and host input agree, including on a separate producer
  stream. Image replay, empty-concept output, and capacity overflow checks pass.
- With an existing engine cache, initialization measured 2.28 s and mean warm
  inference 0.335 s over five runs after two warmups on the three-object fixture.
  These measurements do not cover the requested 1/8/16-object video benchmarks.
- Result tests cover mask union, label ties, order independence and invalid results.
- Tokenizer fixtures cover Unicode normalization, casing, emoji, HTML entities,
  special tokens and truncation. They do not establish full parity with every ftfy
  encoding-repair heuristic; that remains an open requirement.
- The mixed artifact set passes native GPU device/replay/empty/overflow checks.
  CPU execution of that set currently fails an ONNX Runtime inserted-cast type
  check. Use the FP32 artifact set for the CPU baseline.
- A three-frame official causal video reference runs successfully in BF16.
- SAM2 regression targets build. Their four CTest runs fail before inference because
  the configured inputs/references under `S:/din_deploy_artifacts/sam2` are absent.

Mixed FP16 conversion preserves FP32 graph interfaces and sensitive operations.
By default it converts only the text and vision encoders. The installed ModelOpt
version emits invalid detector tensor types, so the detector remains FP32. Converted
encoders undergo type/shape validation after stale intermediate annotations are removed:

```powershell
python vision/sam3/python/convert_sam3_fp16.py artifacts/sam3/onnx-fp32 artifacts/sam3/onnx-fp16
```

The same converter accepts `--precision bf16` for a mixed-BF16 candidate, with
FP32 inputs/outputs and calibrated FP32 exceptions. The default conversion scope
is still the text and vision encoders; the artifact metadata records each graph's
precision. Conversion alone does not authorize changing the runtime default.
TensorRT RTX 1.7.1 cannot compile this vision encoder's BF16 transposed
convolutions. BF16 conversion therefore excludes `ConvTranspose`, retaining its
FP32 computation and weights. This is separate from the calibrated numerical
exceptions selected by ModelOpt.

```powershell
python vision/sam3/python/convert_sam3_fp16.py artifacts/sam3/onnx-fp32 artifacts/sam3/onnx-bf16 --precision bf16
python vision/sam3/python/validate_sam3_precision.py artifacts/sam3/onnx-fp32 artifacts/sam3/onnx-bf16 artifacts/sam3/bf16_validation --executable out/build/windows-x64/bin/Release/din_sam3_cli.exe --image vision/sam2/bike.png --video V1-0022_frames --reference-image artifacts/sam3/reference_fp32
```

The precision validator runs native baseline and candidate processes serially,
checks three bicycle instances and an absent concept, and compares the full hair
sequence. It checks instance counts, output consistency and matched-mask IoU at
0.99. The optional official image reference adds a PyTorch comparison. Passing
native video precision equivalence does not establish upstream temporal-controller
parity; that remains a separate requirement.

The 2026-10-30 BF16 trial passed the three-bicycle image comparison (minimum IoU
0.99321 against official FP32 PyTorch) and empty/device/reset/replay/overflow
checks. It failed full-video precision equivalence: 14 of 95 hair-frame masks were
below 0.99, with mean IoU 0.98803 and minimum 0.76079 against native FP32. Image
inference improved from 364 to 195 ms and video from 718 to 557 ms, but **FP32
remained the default at that stage**. BF16 was subsequently selected explicitly
by the user; the historical native-FP32 comparison remains a diagnostic result.
The candidate metadata records the failed video gate;
`artifacts/sam3/bf16-validation-summary.md` contains the detailed results.

To isolate precision failures, `python/diagnose_sam3_precision.py` runs separate
text-only and vision-only substitutions against an existing native FP32 video
result. It reuses exported graphs, runs cases serially, and records per-frame IoU.
The output directories contain diagnostic models; do not edit their linked ONNX
files because those files share storage with the source exports. External tensor
data is copied because ONNX's external-data loader rejects hard links.

`python/instrument_sam3_decoder.py SOURCE OUTPUT` creates a diagnostic model with
the tracking decoder's three mask-selection scores and selected index exposed.
Pass its `tracking_decoder.onnx` to the native CLI with
`--diagnostic-decoder PATH --diagnostic-dir DIRECTORY`. All other graphs still
come from `--model-dir`, so existing encoder caches are reused. The diagnostic
directory contains one JSON per zero-based sequence frame; the seed frame has
object logits but no tracking selector. This mode introduces GPU readbacks and
file writes and must not be used for performance measurement. Compare instrumented
and ordinary mask outputs before attributing differences to precision.
`python/summarize_sam3_decoder_diagnostics.py` correlates the traces with a mask
comparison report. These tools diagnose the experimental CPU-result CLI controller;
they do not establish official-controller parity or change the default precision.

The 2026-10-30 encoder ablation isolated the large video failures to vision:
BF16 text alone passed all 95 frames (minimum IoU 0.99850), while BF16 vision alone
reproduced all 14 failures (minimum 0.76087). Decoder traces preserved all original
masks exactly and showed mask-selection flips at source frames 15, 23 and 55,
the three largest IoU drops. Eleven other failing frames retained the same mask
candidate, so recurrent drift and within-candidate mask changes still require
separate localization. See `artifacts/sam3/bf16_diagnosis/findings.md` for evidence.

A subsequent fresh official PyTorch BF16 comparison changes the interpretation
of those flips: at source frame 23, native BF16 is closer to PyTorch (IoU 0.95058)
than native FP32 (0.78670). At frames 15 and 55 it is farther. Across 95 frames,
BF16 has mean IoU 0.91438 versus FP32's 0.91676 against PyTorch; both fail the
0.99 gate on 94 frames. BF16 is closer on 42 frames and farther on 53. Both fresh
PyTorch and native-default runs reproduce their earlier masks exactly. BF16 is
the user-selected default, but the incomplete native controller and differing
precision policies still prevent upstream parity. Full results and a per-frame
plot are in `artifacts/sam3/bf16_pytorch_validation/findings.md`.

The subsequent per-graph BF16 comparison identified a concrete memory export bug:
the export fixture aliased `valid` and `conditioning`, so `tracking_memory.onnx`
used conditioning to mask valid objects. Ordinary tracking frames therefore lost
their mask-memory input. The exporter now uses distinct tensors and rejects a
memory export with an unused `valid` input. Previously exported bundles require
regeneration or the validated edge correction; changing the export script alone
does not repair them. The current default bundle includes the validated correction.

Diagnostic edge corrections improved PyTorch-reference IoU on source frames
15/23/55/77/81 from 0.880/0.951/0.855/0.636/0.659 to
0.995/0.993/0.985/0.993/0.963, with BF16 retained. Separate 1008-versus-1152 memory
resize and seed-consolidation mismatches remain. These results supersede any
inference that the overall reference gap was solely a controller or BF16 issue.
See `artifacts/sam3/bf16_graph_isolation_v3/findings.md` for all graph comparisons.

The diagnostic workflow is:

```powershell
python vision/sam3/python/capture_sam3_bf16_graphs.py --video V1-0022_frames --output artifacts/sam3/graph_capture_new --frames 15 23 55 77 81
out/build/windows-x64/bin/Release/din_sam3_replay_graphs.exe artifacts/sam3/onnx-bf16 artifacts/sam3/graph_capture_new
python vision/sam3/python/analyze_sam3_graph_replay.py artifacts/sam3/graph_capture_new
python vision/sam3/python/isolate_sam3_graph_impact.py artifacts/sam3/graph_capture_new
```

Capture propagates upstream context but saves only the requested frames and seed
context. The native replay never feeds one ONNX graph's outputs into another graph;
each receives captured PyTorch inputs. Static-interface mismatches are reported,
not silently resized. `replay_graphs MODEL_DIR FIXTURES cpu GRAPH_NAME` repeats a
graph on ORT CPU. `--diagnostic-graph-dir DIRECTORY` on the regular CLI overrides
only ONNX files present in that directory, preserving other model paths and caches.

Use an empty output directory and rerun graph and mask comparisons before accepting
that artifact set. Conversion does not change the FP32 baseline or certify parity.
Quantization is not implemented.

Remaining completion work includes completing the native causal sequence controller,
exact tokenizer cleanup parity, FP32 video reference parity, broader FP16
validation, late arrivals/disappearance/occlusion/crossing/reset/replay fixtures,
1/8/16-object latency and peak-VRAM measurements, and SAM2 regression execution with
available fixtures. Interactive prompts, backward propagation, persistent tracking
caches remain future native-controller work.

## OpenFX device-output integration

The separate `Sam3GpuPipeline` API in `cpp/sam3_gpu.h` provides persistent CUDA
buffers, device mask composition and asynchronous producer/consumer event handoff.
It is used by **DIN SAM3** in the existing [OpenFX bundle](../../ofx_plugin/README.md).
The CPU-result `Sam3Pipeline` methods above keep their existing behavior. Both
SAM models share the common ONNX Runtime environment so they can coexist in one
process; the plugin builds them against its manually initialized private runtime.

The device-output tracker ports the experimental initial-object controller. It
does not add association, late arrivals, reconditioning or official-reference parity.
It fixes the object set at the seed (an empty seed stays empty), retains live memory
and caches 12 GPU masks; backward cache misses replay from the seed. The native
controller's 0.917 official-reference IoU remains a limitation, irrespective of
agreement between the native and device-output implementations.
