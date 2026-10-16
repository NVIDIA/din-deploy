# PixelDiT

CLI image generation with ONNX Runtime.

## Supported models

| Model | Hugging Face ID | Checkpoint | ONNX export |
| --- | --- | --- | --- |
| PixelDiT-1300M-1024px | `nvidia/PixelDiT-1300M-1024px` | FP32 (runs in BF16) | BF16 weights, FP32 I/O |
| Gemma-2-2B-IT (text encoder) | `Efficient-Large-Model/gemma-2-2b-it` | BF16 | BF16 weights, FP32 I/O |

PixelDiT generates pixels directly, so there is no VAE. The DiT model code comes from the `pixeldit` pip package, installed from [NVlabs/PixelDiT](https://github.com/NVlabs/PixelDiT).

## Pipeline

| Stage | Where | Details |
| --- | --- | --- |
| Tokenize | Host | Gemma tokenizer. The positive prompt is prefixed with a fixed instruction prompt and padded to 506 tokens; the negative prompt is padded to 506 too. |
| Text encoder | `text_encoder/model.onnx` | Gemma-2 decoder, then `select_index` picks the 300 rows the DiT uses. |
| DiT | `transformer/model.onnx` | Batch of 2: negative and positive prompt for classifier-free guidance. Returns velocity. |
| Sampling | Host or CUDA (`--processing`) | CFG and flow DPM-Solver++ (2nd order), 50 steps, CFG scale 2.75, flow shift 4.0 by default. |

Sampling defaults and the token layout are written to `pipeline_config.json`. The C++ runtime reads it at startup and falls back to built-in defaults (matching the 1024x1024 export) for missing fields.

## Backends

| Mode | CLI | Inference and sampling |
| --- | --- | --- |
| TensorRT RTX + CPU | `--provider trt-rtx --processing cpu` | TensorRT RTX inference, sampling on the CPU. |
| TensorRT RTX + CUDA | `--provider trt-rtx --processing cuda` | TensorRT RTX inference, sampling in CUDA kernels. |

Both modes run the models through ONNX Runtime with the TensorRT RTX execution provider.

CUDA sampling keeps the sample on the GPU, so the DiT calls run back to back with one host sync per image. Both modes produce identical images for the same seed.

DirectX and Vulkan processing are not implemented yet. `--provider cpu` is not supported.

## Export

Install the export dependencies first, from the repository root. The `pixeldit` extra adds `pixeldit`, upstream's PixelDiT model package, to the common exporter dependencies (see [Requirements](../../README.md#requirements)). Export and verify run on the GPU, so install a CUDA build of PyTorch first (see [pytorch.org](https://pytorch.org/get-started/locally/)).

```bash
python -m pip install -e ".[pixeldit]"
cd image_gen/pixeldit/model_export
python export_pixeldit.py --output ./models/PixelDiT-1300M-1024px-onnx
```

The DiT is upstream's `PixDiT_T2I` (imported as `pixdit_core`; the local `model_export/pixeldit/` folder holds din-deploy's export helpers). `export_pixeldit.py` patches it only while exporting: complex-valued RoPE has no ONNX lowering, and TensorRT RTX rejects the `Col2Im` op that `F.fold` exports to. See `model_export/pixeldit/upstream_patch.py` for the details. `test_upstream_patch.py` checks the patches on a small random model.

An export contains `text_encoder/model.onnx`, `transformer/model.onnx`, `tokenizer/tokenizer.json`, and `pipeline_config.json`.

Shapes are static. `--height` and `--width` (default 1024, multiples of 16) set the image size; re-export to change it. Add `--compile_trt --trt_root <TensorRT-RTX>` to check that each model builds with the TensorRT RTX CLI.

## Verify

```bash
python verify_pixeldit.py --onnx_dir ./models/PixelDiT-1300M-1024px-onnx --provider trt-rtx
```

Compares each ONNX model running on TensorRT RTX with the upstream PyTorch model from `pixeldit` (unpatched), then generates one image with each and saves `pytorch_output.png`, `ort_output.png`, and `comparison.png`. Pass `--validate_only` to only check the graphs.

Per-model checks decide pass or fail. The image check only fails on broken output: 50-step BF16 sampling amplifies tiny rounding differences, so correct runs can differ in fine detail, or with some seeds in composition. Compare the images visually.

## Build

```powershell
cmake --preset windows-x64
cmake --build --preset windows-x64-release --target din_pixeldit_cli
```

## Run

```powershell
out\build\windows-x64\bin\Release\din_pixeldit_cli.exe --model-dir D:\models\PixelDiT-1300M-1024px-onnx --provider trt-rtx --processing cuda --prompt "a red fox" --output out
```

`--model-dir` defaults to `./models/PixelDiT-1300M-1024px-onnx`, relative to the working directory.

Optional: `--negative-prompt`, `--seed`, `--num-images`, `--steps`, `--cfg-scale`, `--flow-shift` (defaults come from `pipeline_config.json`). The C++ noise generator differs from PyTorch's, so the same seed gives a different image than the Python scripts.
