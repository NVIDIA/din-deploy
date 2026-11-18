# SAM3.1 export and verification

The shared [SAM API](../README.md) loads fixed-attention contract v1 and corrected
memory-boundary v2. The preferred BF16 export directory is `D:/models/sam3.1_bf16`.
The CPU example below exports to `artifacts/sam3/cpu_fp32`. The SAM3 source commit
is pinned in the `sam3` extra in [pyproject.toml](../../../pyproject.toml).

Install dependencies with `uv pip install -e ".[sam3]"`. Obtain access to the
checkpoint on Hugging Face and authenticate before downloading its weights:

```powershell
python vision/sam/python/prepare_sam3.py
```

Preparation, export, verification and SAM3 FP8 calibration share these checkpoint
arguments and defaults: `--checkpoint-repo facebook/sam3.1`,
`--checkpoint-file sam3.1_multiplex.pt`, and
`--checkpoint-revision daa63191845a41281374e725f4c9e51c7a824460`.
Pass the same overrides to each command when using a different checkpoint.
Preparation downloads weights into the Hugging Face cache; source installation
is handled by the Python dependency. Export metadata and calibration provenance
record the source pin and chosen checkpoint. Verification checks those values
against the bundle before loading its PyTorch reference.

## Export

```powershell
python vision/sam/python/export_sam3_onnx.py --static-memory --output D:/models/sam3.1_bf16
python vision/sam/python/verify_sam3.py --onnx-dir D:/models/sam3.1_bf16 --image input.png --text bicycle --output artifacts/sam3/verify_image
```

The exporter uses opset 23 native `Attention`, sixteen object slots, seven
spatial memories and sixteen pointer histories. It also exports tokenizer assets,
temporal constants and metadata. Inspect the verifier's masks under `pytorch/`,
`export/`, and `candidate/` to compare the original predictor and exported pipeline.
Saved graph fixtures are optional and are not needed for this workflow. ONNX wrappers,
tokenizer serialization, antialias resize and sequence lowering are required
parts of the exporter.

Every graph is optimized and constant-folded, including text, vision and detection.
Folding includes weight transposes and static positional/shape calculations;
input-dependent computations remain dynamic. Native attention and FP8 Q/DQ
boundaries are retained. Per-graph node counts are recorded in `metadata.json`.

To create separate folded BF16 and calibrated FP8 encoder bundles:

```powershell
python vision/sam/python/export_sam3_onnx.py --static-memory --output D:/models/sam3.1_bf16_folded
python vision/sam/python/quantize_sam_encoder_fp8.py --model-family sam3 --dtype bfloat16 --dataset TNILab/pascal_voc_seg_train_val --output artifacts/sam3/fp8_calibration
python vision/sam/python/patch_sam_encoder_fp8_qdq.py --model-family sam3 --input D:/models/sam3.1_bf16_folded --calibration artifacts/sam3/fp8_calibration/calibration.json --output D:/models/sam3.1_fp8_folded
```

The FP8 bundle uses selected encoder layers in E4M3FN with BF16 interfaces and
dense surrounding graphs. Constant weights are stored directly as FP8 initializers;
their quantization is folded offline while activation Q/DQ stays in the graph.

Exporter defaults are `--dtype bfloat16`, `--opset 23`, and
`--output artifacts/sam3/onnx-bf16`. All ten graphs are exported by default;
`--graphs` takes space-separated graph names. `--static-memory`,
`--cpu-compatible`, `--fixtures`, and `--fixtures-only` are off by default. The runtime
requires `--static-memory` for GPU bundles. `--cpu-compatible` requires
`--dtype float32` and enables static memory automatically. Export uses CUDA,
including when generating an FP32 CPU bundle; there is no `--device` option.
Without `--fixture-image`, the image reference input is zero-filled. Existing
output metadata must match the selected precision, source and checkpoint.

**BF16 is the default and preferred export precision** (`--dtype bfloat16`).
Model weights and reduced-precision execution use BF16 across all graphs.
FP32 is required for CPU inference and numerical references (`--dtype float32`).
FP16 (`--dtype float16`) comes with quality tradeoffs. The exporter preserves
native attention at the selected precision without inserting attention-only
casts. Preprocessing and numerically sensitive upstream operations can retain
FP32 arithmetic.

The native GPU runtime accepts FP16/BF16/FP32 interfaces, including the BF16
interfaces of FP8 encoder bundles. Recurrent features remain in model precision;
public mask outputs keep their existing types. CPU inference requires FP32.
TensorRT-RTX 1.8.2 currently fails to compile a BF16 transposed convolution in
the vision encoder (`node_convolution_8`). The runtime accepts the I/O contract
and reports that separate engine build failure. The existing
`D:/models/sam3_onnx` FP32-interface bundle is used in the native GPU example below.

Vision input remains 1008 square, decoder logits 288, binary seed prompts 1152,
ordinary consolidated seed-memory logits 1008, and tracking-memory resizing
288 to 1152.

## Optional encoder FP8 quantization

Install ModelOpt with `uv pip install -e ".[sam3,quantization]"` and prepare the
pinned source/checkpoint as above. Use the same dtype for calibration and export.
This example keeps native-compatible FP32 interfaces and dense surrounding graphs:

```powershell
python vision/sam/python/export_sam3_onnx.py --dtype float32 --static-memory --output artifacts/sam3/dense_fp32
python vision/sam/python/quantize_sam_encoder_fp8.py --model-family sam3 --dtype float32 --dataset TNILab/pascal_voc_seg_train_val --split train --output artifacts/sam3/fp8_calibration
python vision/sam/python/patch_sam_encoder_fp8_qdq.py --model-family sam3 --input artifacts/sam3/dense_fp32 --calibration artifacts/sam3/fp8_calibration/calibration.json --output artifacts/sam3/fp8_qdq
python vision/sam/python/verify_sam3.py --onnx-dir artifacts/sam3/fp8_qdq --validate-only --output artifacts/sam3/fp8_structure
```

The shared command mirrors SAM2's dataset-driven ModelOpt static per-tensor E4M3
calibration and conservative exclusions. SAM3 uses its own 1008-pixel preprocessor,
pinned checkpoint and vision trunk linears; patch embedding, necks, text, detector
and temporal graphs remain dense. SAM3 calibration requires CUDA; `--model` is
SAM2-only. BF16 is the default dtype, but full BF16 bundles remain subject to the
TensorRT operator limitation above.

Defaults are 128 calibration rows and the next 64 scoring rows from the dataset's
`image` column. Select representative images with `--dataset`/`--split`, adjust
counts with `--calib-samples`/`--score-samples`, and ensure scoring images do not
duplicate calibration images. Calibration reports encoder feature MSE and saves
`encoder_fp8_modelopt.pt`, `quant_summary.txt` and `calibration.json` with exact
maxima and selected layers. Exclusion controls are listed in `--help`.
`--dataset` also accepts a local image folder, optionally containing a subfolder
named by `--split`; images are read in filename order. Local folders avoid the
`datasets`/PyArrow dependency when no Windows ARM64 wheel is available.

The patcher copies to a new bundle, changes only `vision_encoder.onnx`, preserves
interfaces and surrounding precision, and supports `--weight-only`. It uses exact
calibration values rather than rounded printed summaries. Unlike the Downloads
study/patch scripts, it needs neither local image lists nor a multi-variant study
report, makes no fixed 128-linear assumption, and does not insert BF16 casts.
Study, reference and benchmark diagnostics remain local.

FP8 can affect segmentation and tracking quality. Structural verification and
encoder feature MSE do not establish downstream parity or faster kernels. Validate
the resulting bundle with TensorRT-RTX and representative image/video prompts;
CPU FP8 execution and FP8 speedup are not established by these commands.
Use the original PyTorch checkpoint as the quality reference, with matching
precision, images and prompts. Label ModelOpt simulation and exported ONNX
execution results separately.

## CPU bundle and numerical verification

```powershell
python vision/sam/python/export_sam3_onnx.py --dtype float32 --cpu-compatible --fixtures --output artifacts/sam3/cpu_fp32
python vision/sam/python/verify_sam3.py --onnx-dir artifacts/sam3/cpu_fp32 --provider cpu --validate-only --fixtures --output artifacts/sam3/verification
din_sam_cli input.png output --model sam3 --model-dir artifacts/sam3/cpu_fp32 --provider cpu --text bicycle
```

The CPU export chunks attention queries for ORT's mask handling
and preserves FP32 arithmetic. Numerical fixture checks are an optional diagnostic:
export with `--fixtures` to save `.inputs.npz`, `.outputs.npz`, and tokenizer test
cases, then verify with `--validate-only --fixtures`. To regenerate fixtures for
existing graphs, export with `--fixtures-only`. No raw `.bin` copies are written.
Without `--fixtures`, `--validate-only` checks ONNX structure and sidecars only.
Normal image/sequence verification uses the supplied images and original checkpoint;
it does not read export fixtures. Existing fixture files are left untouched when
exporting into an existing directory without fixture options.

## Python reference and ORT verification

```powershell
python vision/sam/python/verify_sam3.py --onnx-dir artifacts/sam3/bundle --image input.png --text bicycle --onnx-components vision_encoder --output artifacts/sam3/verify_image
python vision/sam/python/verify_sam3.py --onnx-dir artifacts/sam3/bundle --frames-dir frames --text bicycle --limit-frames 8 --onnx-components all --output artifacts/sam3/verify_tracking
```

The script follows SAM2's separate image/sequence reference-runner structure,
using `Sam3Processor` for images and the pinned upstream causal video-session API.
The component pipeline reuses `sam3_modeling.py`. The baseline runs every component's
export (PyTorch) path; the candidate uses ONNX for every component by default.
`--onnx-components` selects `image_preprocess`,
`text_encoder`, `vision_encoder`, `concept_detector`, or any combination. Sequence
mode additionally supports `seed_decoder`, `seed_memory`, `tracking_decoder`,
`tracking_memory`, and `memory_attention`. `all` is the default and selects every
component used by the mode; `none` selects an all-PyTorch candidate.
The raw `memory_encoder` export is an alternative packed-mask boundary
inside the seed/tracking memory components; it is checked separately with
`--validate-only` / `--fixtures`, rather than executing the same network twice.
An explicitly selected ONNX component never silently falls back to PyTorch.

Outputs retain per-instance masks/probabilities, scores, label maps and combined
masks under `pytorch/` (full upstream), `export/` (all-PyTorch component baseline),
and `candidate/` (selected backend combination). The baseline and candidate share
one pipeline and reset their temporal state/text cache between runs.
`comparison.json` reports instance-count checks, one-to-one IoU matching against
both references, component backends/calls, numerical graph errors and parity gaps. Instance ordering
may differ between independent runs. Use `--din-sam path/to/din_sam_cli` for an
additional native-mask comparison; tracking accepts image folders, not encoded video.
Tracking retains a stable ID mapping across frames to detect identity switches.
The pinned session wrapper forwards default initialization arguments that its
multiplex model does not accept; a scoped adapter handles only the default
offload/loader arguments and restores the original method after session creation.

The existing sixteen-object contract is preserved. The native controller's
association/reconditioning gaps remain visible against the full upstream reference.
The video reference retains official BF16 autocast; network-wrapper comparisons
use the bundle dtype. The pinned predictor requires CUDA even when testing a CPU
ORT bundle. Structure/fixture checks do not require constructing that predictor.
See [verification options](../README.md#export-and-verification) for details.

## Native inference

```powershell
cmake --build out/build/windows-arm64 --config Release --target din_sam_cli
din_sam_cli input.png output --model sam3 --model-dir D:/models/sam3_onnx --text bicycle
```

The build command assumes the `windows-arm64` configure preset has already
been configured. Use `out/build/windows-x64` for the `windows-x64` preset.
The CLI defaults to `--provider trt-rtx --device 0 --threshold 0.5`;
`--threshold` controls detection confidence. The mask probability cutoff is
fixed at 0.5. The CLI requires both `--model` and `--model-dir`.

Use a frame directory for causal video inference (temporal tracking is the
default) or `--tracking none` for independent segmentation. Cache options and timing reports are shared with
SAM2; see [CLI usage](../README.md#command-line-inference). The current
controller initializes on the first qualifying detection and retains bounded
history; association, reconditioning and full native PyTorch video parity remain
separate work.
