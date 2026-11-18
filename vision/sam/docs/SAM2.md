# SAM2 export and verification

The shared [SAM API](../README.md) loads fixed-attention contract v1. The preferred
BF16 GPU bundle is `D:/models/sam2.1_base_plus_bf16`. Export dependencies are
installed with `uv pip install -e ".[sam2]"`.

BF16 is the default and preferred precision (`--dtype bfloat16`). FP32 is required
for CPU inference and numerical references (`--dtype float32`). FP16 (`--dtype float16`) comes with quality tradeoffs.
Attention retains the selected
model precision without attention-only casts.

The native GPU runtime accepts FP16/BF16/FP32 interfaces and stores history in
the selected precision. CPU inference requires FP32. TensorRT-RTX 1.8.2 currently
fails to compile the decoder's BF16 transposed convolution; this is an engine
operator limitation after I/O validation, rather than a rejected precision contract.

## GPU export

```powershell
python vision/sam/python/export_sam2_onnx.py --model facebook/sam2.1-hiera-base-plus --device cuda --static-memory --output D:/models/sam2.1_base_plus_bf16
python vision/sam/python/verify_sam2.py --onnx-dir D:/models/sam2.1_base_plus_bf16 --image input.png --prompts prompt.json --output artifacts/sam2/verify_image
```

This exports seven spatial-memory slots with validity masking. Keep the 1024
image size and match the exported point count when preparing prompts. The
exporter's ONNX wrappers and shared attention helper are required source files.
Every graph is automatically constant-folded after export and memory specialization,
including preprocessing and mask postprocessing. Folding precomputes static weights,
positions and shapes while preserving graph interfaces and native attention.
No numerical accuracy checks or fixtures are generated during export.
Use the verifier's `pytorch/`, `export/`, and `candidate/` masks for visual inspection;
normal image/sequence verification does not require saved export fixtures.

Exporter defaults are `--model facebook/sam2.1-hiera-large`, `--device cpu`,
`--dtype bfloat16`, `--output artifacts/sam2/onnx`, `--opset 23`,
`--image-size 1024`, `--max-points 3`, `--max-memory-frames 7`, and
`--max-input-size 4096`. The example explicitly selects base-plus and CUDA.
All seven graphs are exported by default. `--graphs` takes a comma-separated
list. `--static-memory` and `--cpu-compatible` are off by default;
`--cpu-compatible` requires FP32 and enables static memory automatically.
The image size must remain 1024 for this exporter.

`--clean` is on by default and removes an existing output directory before
export. `--no-clean` preserves existing files and checks the existing metadata's
model and precision when present.

## Optional encoder FP8 quantization

Install ModelOpt with `uv pip install -e ".[sam2,quantization]"`. Calibrate the
same model and dtype used by the dense export. This example uses native-compatible
FP16; FP16 and FP8 can change segmentation quality.

```powershell
python vision/sam/python/export_sam2_onnx.py --model facebook/sam2.1-hiera-base-plus --dtype float16 --device cuda --static-memory --output artifacts/sam2/dense_fp16
python vision/sam/python/quantize_sam_encoder_fp8.py --model-family sam2 --model facebook/sam2.1-hiera-base-plus --dtype float16 --dataset TNILab/pascal_voc_seg_train_val --split train --output artifacts/sam2/fp8_calibration
python vision/sam/python/patch_sam_encoder_fp8_qdq.py --model-family sam2 --input artifacts/sam2/dense_fp16 --calibration artifacts/sam2/fp8_calibration/calibration.json --output artifacts/sam2/fp8_qdq
python vision/sam/python/verify_sam2.py --onnx-dir artifacts/sam2/fp8_qdq --validate-only --output artifacts/sam2/fp8_structure
```

Calibration uses ModelOpt static per-tensor E4M3 activation/weight quantization,
128 calibration rows and the next 64 rows for encoder feature-MSE scoring.
Choose a representative image dataset with an `image` column using `--dataset`
and `--split`; counts are adjustable with `--calib-samples` and `--score-samples`.
`--dataset` also accepts a local image folder (or a folder containing the selected
split subfolder). Images are read in filename order. This path needs Pillow and
ModelOpt, but not Hugging Face `datasets`/PyArrow, whose Windows ARM64 wheel may
be unavailable.
Separate rows are used for scoring; ensure the dataset itself has no duplicate
images across those subsets. Sensitive layers remain dense. Exclusion controls
are listed in `--help`. BF16 is the calibration default, matching the exporters.

Calibration saves `encoder_fp8_modelopt.pt`, `quant_summary.txt` and
`calibration.json` with exact maxima, selected layers and provenance. It does not
export ONNX. The patcher requires a new output directory, copies the dense bundle
and inserts FP8 Q/DQ only in `image_encoder.onnx`. `--weight-only` skips activation
quantization. Other graphs, interfaces and bundle dtype metadata are preserved.

This restores the calibration/patching workflow from `main` in the unified SAM
directory. Printed summary rounding and the old patcher's fixed FP16 scale type
are avoided. Graph validation and feature MSE do not establish mask quality,
tracking parity or FP8 speedup. Test the patched bundle with TensorRT-RTX on
representative images/video before deployment. CPU FP8 execution is not covered;
full BF16 bundles remain subject to the TensorRT operator limitation described above.
Use the original PyTorch checkpoint as the quality reference, with matching
precision, images and prompts. Report dense ONNX export drift and FP8 drift
against that same PyTorch reference separately.

## CPU export

Export the final FP32 CPU bundle directly:

```powershell
python vision/sam/python/export_sam2_onnx.py --model facebook/sam2.1-hiera-base-plus --dtype float32 --cpu-compatible --output artifacts/sam2/cpu_fp32
python vision/sam/python/verify_sam2.py --onnx-dir artifacts/sam2/cpu_fp32 --validate-only --output artifacts/sam2/cpu_structure
din_sam_cli input.png output --model sam2 --prompt prompt.json --model-dir artifacts/sam2/cpu_fp32 --provider cpu
```

## Native inference

```powershell
cmake --build out/build/windows-arm64 --config Release --target din_sam_cli din_sam2_verify_masks
din_sam_cli input.png output --model sam2 --prompt prompt.json --model-dir D:/models/sam2.1_base_plus_onnx
```

The build command assumes the `windows-arm64` configure preset has already
been configured. Use `out/build/windows-x64` for the `windows-x64` preset.
The CLI defaults to `--provider trt-rtx --device 0` and requires an explicit
`--model` and `--model-dir`; its remaining defaults are in [CLI usage](../README.md#command-line-inference).

The prompt JSON contains source-image coordinates and exactly the exported
number of labels; use `-1` for unused points. For example:

```json
{
  "points": [
    [
      384,
      384
    ],
    [
      0,
      0
    ],
    [
      0,
      0
    ]
  ],
  "labels": [
    1,
    -1,
    -1
  ]
}
```

Use a frame directory for video inference (temporal tracking is the default) or
`--tracking none` for independent segmentation. The shared CLI uses the same cache layout, output names
and timing report for both families; see [CLI usage](../README.md#command-line-inference).
The standalone
`din_sam2_verify_masks` utility compares exported masks with a supplied reference
directory; it does not depend on local unit tests.

```powershell
din_sam2_verify_masks --candidate output --reference reference_masks --min-iou 0.95
```

Candidate PNG filenames must match those in the reference directory. The default
minimum IoU is 0.95 for each image.

## Python reference and ORT verification

```powershell
python vision/sam/python/verify_sam2.py --onnx-dir artifacts/sam2/bundle --image input.png --prompts prompt.json --onnx-components image_encoder --output artifacts/sam2/verify_image
python vision/sam/python/verify_sam2.py --onnx-dir artifacts/sam2/bundle --frames-dir frames --prompts prompt.json --limit-frames 8 --onnx-components all --output artifacts/sam2/verify_tracking
```

The reference retains `SAM2ImagePredictor` and `build_sam2_video_predictor_hf`,
including their prompt and sequence handling. The component pipeline directly reuses
`sam2_modeling.py`: the baseline runs through its PyTorch export wrappers, and the
candidate uses ONNX for every component by default.
`--onnx-components` selects any combination of `image_preprocess`, `image_encoder`,
`sam2_decoder`, `sam2_decoder_propagate`, `memory_attention`, and `memory_encoder`;
the last three require sequence mode. `all` is the default and selects every
component used by the mode; `none` selects an all-PyTorch candidate.
Only selected components require ONNX files or ORT sessions. Mask postprocessing
remains shared Python/PyTorch code; the standalone `mask_postprocess` graph can be
checked using `--validate-only` / `--fixtures` when fixtures are supplied.
The all-PyTorch export baseline and mixed candidate share the same seven-slot
memory controller, and selected ONNX outputs are compared against their wrapper
on the same logical inputs. Fixed-history padding belongs to the ONNX adapter only.
Use `--din-sam path/to/din_sam_cli` to compare native masks with the candidate.

Current exports support `multimask_output=False`; the tracking candidate supports
one object prompted at frame zero. Missing IoU predictions, object pointers and
upstream memory selection are explicit gaps, not fabricated adapter outputs.
Prompts may use fewer than `max_points`; unused candidate points are padded with
label `-1`, with box corners counted against the capacity. Sequence frames are
converted to real JPEGs for the upstream loader, and both Python paths consume
those same pixels. Native comparison stages the same pixels as PNGs.

`pytorch/` contains the full upstream reference, `export/` the all-PyTorch component
baseline, and `candidate/` the selected backend combination. `comparison.json`
records per-frame upstream IoU, `export_iou`, component backends/calls, numerical
errors and parity gaps. The baseline and candidate use the bundle precision.
See [verification options](../README.md#export-and-verification) for provider,
precision and threshold details. To check only structure, pass `--validate-only`.
SAM2's exporter does not write the fixtures required by `--fixtures`.
