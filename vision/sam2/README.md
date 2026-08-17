# SAM2

## Setup

Install the SAM2 Python/export dependencies:

```bash
uv pip install -e ".[sam2]"
```

## Export ONNX Artifacts

Export the split SAM2 graphs:

```bash
python vision/sam2/python/export_sam2_onnx.py \
  --model facebook/sam2.1-hiera-large \
  --output artifacts/sam2/onnx \
  --device cuda \
  --dtype float16 \
  --max-points 3
```

This writes the ONNX files plus `metadata.json` under `artifacts/sam2/onnx`.

## Build

Configure and build the CLI:

```bash
cmake --preset linux-x64
cmake --build out/build/linux-x64 --config Release --target din_sam2
```

## Run

Create a prompt JSON. The exported decoder expects exactly `--max-points` labels; use `-1` for unused points.

```json
{
  "points": [[384, 384], [0, 0], [0, 0]],
  "labels": [1, -1, -1]
}
```

Run image segmentation:

```bash
din_sam2_cli \
  --model-dir artifacts/sam2/onnx \
  --provider trt-rtx \
  --dump-masks \
  input.png prompt.json out/sam2_cli/masks
```

Run video/frame-directory propagation:

```bash
din_sam2_cli \
  --model-dir artifacts/sam2/onnx \
  --provider trt-rtx \
  --propagate \
  --dump-masks \
  frames_dir prompt.json out/sam2_video/masks
```

