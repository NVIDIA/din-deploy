# PixelDiT

ONNX export and verification for PixelDiT text-to-image with ONNX Runtime. There is no C++ runtime yet.

## Supported models

| Model | Hugging Face ID | Checkpoint | ONNX export |
| --- | --- | --- | --- |
| PixelDiT-1300M-1024px | `nvidia/PixelDiT-1300M-1024px` | FP32 (runs in BF16) | BF16 weights, FP32 I/O |
| Gemma-2-2B-IT (text encoder) | `Efficient-Large-Model/gemma-2-2b-it` | BF16 | BF16 weights, FP32 I/O |

PixelDiT generates pixels directly, so there is no VAE. The model code is vendored in `model_export/pixeldit/`, so the [NVlabs/PixelDiT](https://github.com/NVlabs/PixelDiT) repo is not needed to export or verify.

## Pipeline

| Stage | Where | Details |
| --- | --- | --- |
| Tokenize | Host | Gemma tokenizer. The positive prompt is prefixed with a fixed instruction prompt and padded to 506 tokens; the negative prompt is padded to 506 too. |
| Text encoder | `text_encoder/model.onnx` | Gemma-2 decoder, then `select_index` picks the 300 rows the DiT uses. |
| DiT | `transformer/model.onnx` | Batch of 2: negative and positive prompt for classifier-free guidance. Returns velocity. |
| Sampling | Host | CFG and flow DPM-Solver++ (2nd order), 50 steps, CFG scale 2.75, flow shift 4.0 by default. |

Sampling defaults and the token layout are written to `pipeline_config.json`.

## Export

```bash
cd model_export
python export_pixeldit.py --output ./models/PixelDiT-1300M-1024px-onnx
```

An export contains `text_encoder/model.onnx`, `transformer/model.onnx`, `tokenizer/tokenizer.json`, and `pipeline_config.json`.

Shapes are static. `--height` and `--width` (default 1024, multiples of 16) set the image size; re-export to change it. Add `--compile_trt --trt_root <TensorRT-RTX>` to check that each model builds with the TensorRT RTX CLI.

## Verify

```bash
python verify_pixeldit.py --onnx_dir ./models/PixelDiT-1300M-1024px-onnx --provider trt-rtx
```

Compares each ONNX model running on TensorRT RTX with the vendored PyTorch model, then generates one image with each and saves `pytorch_output.png`, `ort_output.png`, and `comparison.png`. Pass `--validate_only` to only check the graphs.

Per-model checks decide pass or fail. The image check only fails on broken output: 50-step BF16 sampling amplifies tiny rounding differences, so correct runs can differ in fine detail, or with some seeds in composition. Compare the images visually.
