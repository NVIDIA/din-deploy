# SAM2 export and verification

The shared [SAM API](../README.md) loads fixed-attention contract v1. The preferred
BF16 GPU bundle is `D:/models/sam2.1_base_plus_bf16`. Export dependencies are
installed with `uv pip install -e ".[sam2]"`.

BF16 is the default and preferred precision (`--dtype bfloat16`). FP32 is required
for CPU inference and numerical references (`--dtype float32`). FP16 (`--dtype float16`) comes with quality tradeoffs.
Attention retains the selected
model precision without attention-only casts.

The current native runtime accepts SAM2 FP16/FP32 bundles; full BF16 bundles
can be exported and structurally verified but are not yet accepted by the runtime.

## GPU export

```powershell
python vision/sam/python/export_sam2_onnx.py --model facebook/sam2.1-hiera-base-plus --device cuda --static-memory --output D:/models/sam2.1_base_plus_bf16
python vision/sam/python/verify_sam_onnx.py D:/models/sam2.1_base_plus_bf16
```

This exports seven spatial-memory slots with validity masking. Keep the 1024
image size and match the exported point count when preparing prompts. The
exporter's ONNX wrappers and shared attention helper are required source files.

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

## CPU export

Export the final FP32 CPU bundle directly:

```powershell
python vision/sam/python/export_sam2_onnx.py --model facebook/sam2.1-hiera-base-plus --dtype float32 --cpu-compatible --output artifacts/sam2/cpu_fp32
python vision/sam/python/verify_sam_onnx.py artifacts/sam2/cpu_fp32
din_sam_cli input.png output --model sam2 --prompt prompt.json --model-dir artifacts/sam2/cpu_fp32 --provider cpu --dump-masks
```

## Native inference

```powershell
cmake --build out/build/windows-arm64 --config Release --target din_sam_cli din_sam2_verify_masks
din_sam_cli input.png output --model sam2 --prompt prompt.json --model-dir D:/models/sam2.1_base_plus_onnx --dump-masks
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

Use `--track` with a frame directory for video inference, or `--framewise` for
independent segmentation. The shared CLI uses the same cache layout, output names
and timing report for both families; see [CLI usage](../README.md#command-line-inference).
The standalone
`din_sam2_verify_masks` utility compares exported masks with a supplied reference
directory; it does not depend on local unit tests.

```powershell
din_sam2_verify_masks --candidate output --reference reference_masks --min-iou 0.95
```

Candidate PNG filenames must match those in the reference directory. The default
minimum IoU is 0.95 for each image. `verify_sam_onnx.py` instead checks ONNX
structure; SAM2's exporter does not write the input/output fixtures required by
its optional `--fixtures` comparison mode.
