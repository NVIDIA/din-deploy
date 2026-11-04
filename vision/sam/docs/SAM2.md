# SAM2 export and verification

The shared [SAM API](../README.md) loads fixed-attention contract v1. The default
GPU bundle is `D:/models/sam2.1_base_plus_onnx`; its FP32 CPU counterpart appends
`_cpu_fp32`. Export dependencies are installed with `uv pip install -e ".[sam2]"`.

## GPU export

```powershell
python vision/sam/python/export_sam2_onnx.py --model facebook/sam2.1-hiera-base-plus --dtype float16 --static-memory --output artifacts/sam2/onnx
python vision/sam/python/verify_sam_onnx.py artifacts/sam2/onnx
```

This exports seven spatial-memory slots with validity masking. Keep the 1024
image size and match the exported point count when preparing prompts. The
exporter's ONNX wrappers and shared attention helper are required source files.

## CPU export

Export FP32 first, then add the fixed contract and CPU mask compatibility:

```powershell
python vision/sam/python/export_sam2_onnx.py --model facebook/sam2.1-hiera-base-plus --dtype float32 --output artifacts/sam2/raw_fp32
python vision/sam/python/prepare_static_sam.py sam2 artifacts/sam2/raw_fp32 artifacts/sam2/cpu_fp32 --cpu-compatible
python vision/sam/python/verify_sam_onnx.py artifacts/sam2/cpu_fp32
```

## Native inference

```powershell
cmake --build out/build/windows-x64 --config Release --target din_sam2_cli din_sam2_verify_masks
din_sam2_cli input.png prompt.json output --model-dir D:/models/sam2.1_base_plus_onnx --dump-masks
```

The prompt JSON contains source-image coordinates and exactly the exported
number of labels; use `-1` for unused points. For example:

```json
{"points": [[384, 384], [0, 0], [0, 0]], "labels": [1, -1, -1]}
```

Use `--propagate` with a frame directory for video inference. The standalone
`din_sam2_verify_masks` utility compares exported masks with a supplied reference
directory; it does not depend on local unit tests.
