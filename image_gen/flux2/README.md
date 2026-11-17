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
python image_gen/flux2/model_export/export_flux2.py --model_name black-forest-labs/FLUX.2-klein-4b --output models/flux_full --model all --transformer-precision bf16
python image_gen/flux2/model_export/quantize_onnx.py --input models/flux_full/transformer_bf16 --quant fp8 --output models/flux_full/transformer_fp8
```

Use `--quant nvfp4` with an output directory of `transformer_nvfp4` for NVFP4.
All precisions share the text encoder, VAE, and tokenizer under one model root.

For BF16 weights and computation with dynamic image height/width, use a separate
output directory:

```powershell
python image_gen/flux2/model_export/export_flux2.py --model_name black-forest-labs/FLUX.2-klein-4b --output models/flux_dynamic_bf16 --model all --transformer-precision bf16 --dynamic-shapes
```

Quantize that dynamic transformer while sharing its text encoder and VAE:

```powershell
python image_gen/flux2/model_export/quantize_onnx.py --input models/flux_dynamic_bf16/transformer_bf16 --output models/flux_dynamic_bf16/transformer_fp8 --quant fp8 --compute-dtype bf16 --fold-weight-quantizers
python image_gen/flux2/model_export/quantize_onnx.py --input models/flux_dynamic_bf16/transformer_bf16 --output models/flux_dynamic_bf16/transformer_nvfp4 --quant nvfp4 --compute-dtype bf16
```

Quantization preserves the graph's dynamic dimensions and FP32 inputs/outputs.
It uses the published quantized checkpoints' scales and NVFP4 payloads rather
than running new calibration. FP8 weight folding stores FP8 weights on disk;
NVFP4 weights are stored packed. Other operations retain BF16 computation.

`--image_size` selects the square example used during export; dynamic graphs can
accept rectangular images with height and width divisible by 16. The transformer
shares a dynamic image-token dimension between `hidden_states` and `img_ids`;
the VAE encoder and decoder have dynamic spatial dimensions. Batch size stays 1
and text length stays 512. Floating-point graph inputs/outputs stay FP32 to match
the C++ buffers. The studio has Width and Height controls and accepts `--width`
and `--height` at startup. Each dimension must be divisible by 16, with width
32–3840 and height 32–2160; the default is 1024×1024. Select a model root exported
with `--dynamic-shapes` to use other resolutions.

The runtime compiles a shared TensorRT profile with minimum 32×32, optimum
1024×1024, and maximum 3840×2160. Resolution changes recreate the pipeline buffers
and sessions while reusing the compiled engine cache. The CLI defaults to
1024×1024. Export-time `--compile_trt` still takes profiles through `--trt_arg`.

Optionally export Qwen3-0.6B + `gafr1/translator` as the small text encoder:

```powershell
python image_gen/flux2/model_export/export_flux2.py --model_name models/flux_full --output models/flux_full --model text_encoder_translator
python image_gen/flux2/model_export/verify_flux2.py --onnx_dir models/flux_full --encoder translator --encoder-only
```

Use `verify_flux2.py --help` for full-pipeline verification options.

Optionally export the [FLUX.2 small decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder)
alongside the full decoder. This loads only the VAE checkpoint and writes
`vae_decoder_small/model.onnx`, external BF16 weights, and its own `latent_stats.json`:

```powershell
python image_gen/flux2/model_export/export_flux2.py --output D:/models/flux_dynamic_bf16 --model vae_decoder_small --opset 24 --dynamic-shapes
python image_gen/flux2/model_export/export_flux2.py --output D:/models/flux_dynamic_bf16_opset24 --model vae_decoder_small --opset 24 --dynamic-shapes
```

The small decoder defaults to opset 24 and requires a native ONNX `Attention` node.
Select **Small** in the studio's **VAE decoder** menu, or launch the UI/CLI with
`--decoder small` (default: `full`). The UI enables Small when its ONNX file exists
in the selected model root. Switching decoders reloads the VAE stage and uses
separate engine caches. The encoder and transformer are reused.

Known runtime limitation: TensorRT RTX 1.8.2 rejects this checkpoint's native
Attention node (one head with 384 channels) during engine compilation on the
tested Windows ARM64 system. The opset-24 export passes ONNX validation, but
generation with this decoder remains blocked by that backend limitation.

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
The has configuration checks per stage and does partial reload of impacted models upon changes.
Changing RNG, denoising steps, or the runtime weight-streaming budget reuses all
loaded model sessions. For a full execution backend change all models are reloaded.

The UI uses GLFW, Dear ImGui, and OpenGL. For Debian/Ubuntu, install
`libgl1-mesa-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev`.
Use `-DDIN_BUILD_FLUX2_UI=OFF` for headless builds.

For pure CLI you can use:

```powershell
out/build/windows-x64/bin/Release/din_flux2_cli.exe --model-dir models/flux_full --precision fp8 --provider trt-rtx --processing cuda --prompt "a red fox" --output out
```

Use `--help` for available processing backends and options, including `--encoder`,
`--steps`, and startup-only `--ws`. CPU, CUDA, DirectX, and Vulkan processing are available where built. 
