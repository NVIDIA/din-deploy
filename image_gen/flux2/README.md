# FLUX.2-klein

Interactive image generation and a CLI for FLUX.2-klein-4B using ONNX Runtime and TensorRT RTX.

The interactive studio was used to produce the results in
[The Weight Is Over – Interactive Diffusion on Consumer GPUs](https://arxiv.org/abs/2609.21849)
(Frieder Ganz and Maximilian Müller). See the paper for the small text-encoder translator,
interactive workflow, and speed/quality/memory trade-offs implemented here.

## Supported models

| Model                 | Hugging Face ID                           | Checkpoint | ONNX export |
|-----------------------|-------------------------------------------|------------|-------------|
| FLUX.2-klein-4B       | `black-forest-labs/FLUX.2-klein-4B`       | BF16       | FP32 I/O    |
| FLUX.2-klein-4B FP8   | `black-forest-labs/FLUX.2-klein-4b-fp8`   | FP8        | FP32 I/O    |
| FLUX.2-klein-4B NVFP4 | `black-forest-labs/FLUX.2-klein-4b-nvfp4` | NVFP4      | FP32 I/O    |

The backend supports CPU or TensorRT RTX execution with CPU, CUDA, DirectX, DirectX CIG, and Vulkan processing where
available.

## Export

Run from the repository root:

```powershell
python ./model_export/export_flux2.py --model_name black-forest-labs/FLUX.2-klein-4b --output models/flux_full --model all --transformer-precision bf16
python ./model_export/quantize_onnx.py --input models/flux_full/transformer_bf16 --quant fp8 --output models/flux_full/transformer_fp8
```

Use `--quant nvfp4` with an output directory of `transformer_nvfp4` for NVFP4.
All precisions share the text encoder, VAE, and tokenizer under one model root.

Optionally export Qwen3-0.6B + `gafr1/translator` as the small text encoder:

```powershell
python image_gen/flux2/model_export/export_flux2.py --model_name models/flux_full --output models/flux_full --model text_encoder_translator
python image_gen/flux2/model_export/verify_flux2.py --onnx_dir models/flux_full --encoder translator --encoder-only
```

Use `verify_flux2.py --help` for full-pipeline verification options.

## Build and run

```powershell
cmake --preset windows-x64
cmake --build --preset windows-x64-release --target din_flux2_ui din_flux2_cli
```

From the repo root you can then run:

```
out/build/windows-x64/bin/Release/din_flux2_ui.exe --model-dir <path to models>/flux_full
```

The studio provides prompt editing, encoder/precision/backend selection, seed sweeps,
weight-streaming controls, and an image gallery with optional PNG output. It selects the
translator when exported; override with `--encoder 4b|translator`.
Runtime and compiled-model caching are always enabled; clear affected caches after re-exporting models in place.

The UI uses GLFW, Dear ImGui, and OpenGL. For Debian/Ubuntu, install
`libgl1-mesa-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev`.
Use `-DDIN_BUILD_FLUX2_UI=OFF` for headless builds.

For pure CLI you can use:

```powershell
out/build/windows-x64/bin/Debug/din_flux2_cli.exe --model-dir models/flux_full --precision fp8 --provider trt-rtx --processing cuda --prompt "a red fox" --output out
```

Use `--help` for available processing backends and options, including `--encoder`,
`--steps`, and startup-only `--ws`. CPU, CUDA, DirectX, and Vulkan processing are available where built. 
