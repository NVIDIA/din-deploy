# SeedVR2

Image super-resolution with ONNX Runtime, TensorRT RTX, and CUDA.

## Supported models

| Model      | Hugging Face ID     | Checkpoint | ONNX export                    |
|------------|---------------------|------------|--------------------------------|
| SeedVR2 3B | `Comfy-Org/SeedVR2` | FP16       | FP16, dynamic height and width |

The pipeline supports one image, one video frame, and the 3B checkpoint. Image
dimensions are padded to multiples of 16 before VAE encoding. The C++ target
uses no Vulkan or DirectX components.

## Export

Use an existing ComfyUI checkout and run the exporter from the repository root:

```powershell
& .\din_venv\Scripts\python.exe image_gen\seedvr\model_export\export_seedvr.py `
  --comfy-root D:\repos\ComfyUI `
  --snapshot D:\hf\hub\models--Comfy-Org--SeedVR2\snapshots\673340c8a66db62b84e4099def7d01d337ae12dc `
  --output out\seedvr\onnx\seedvr2_3b_fp16_dit.onnx
```

The default export produces:

- `seedvr2_3b_fp16_dit.onnx` and its external data
- `seedvr2_ema_vae_encoder.onnx`
- `seedvr2_ema_vae_decoder.onnx`
- `seedvr2_3b_fp16_positive_context.bin`

The VAE image axes and DiT latent axes are dynamic. Batch size, temporal
length, channels, and the `58x5120` text context remain fixed. The DiT receives
normal and shifted window/RoPE metadata as additional dynamic-length inputs.

Select individual artifacts with `--components dit context vae_encoder
vae_decoder`.

## Verify

Compare the full ComfyUI PyTorch pipeline with the three ONNX components
running through TensorRT RTX:

```powershell
& .\din_venv\Scripts\python.exe image_gen\seedvr\model_export\verify_seedvr.py `
  --comfy-root D:\repos\ComfyUI `
  --onnx-dir out\seedvr\onnx `
  --input assets\tj.png `
  --output-dir out\seedvr\verify
```

The verifier writes `pytorch_output.png`, `ort_output.png`, and
`comparison.png`, then reports MSE, MAE, maximum pixel error, and PSNR. CPU
execution-provider fallback is disabled for the ONNX path.

## Build

```powershell
cmake --preset windows-arm64
cmake --build --preset windows-arm64-release --target din_seedvr2_cli
```

## Run

```powershell
.\out\build\windows-arm64\bin\Release\din_seedvr2_cli.exe assets\tj.png `
  --model-dir out\seedvr\onnx `
  --output out\seedvr\tj-seedvr-2x-cpp.png `
  --scale 2 `
  --seed 0
```

Image resize and window/RoPE metadata generation run on the CPU. The VAE
encoder, DiT, and VAE decoder run through TensorRT RTX; condition packing and
the Euler update run as CUDA kernels on the same ONNX Runtime compute stream.
TensorRT RTX runtime caches are stored under `out/seedvr/trt_rtx_cache` by
default.
