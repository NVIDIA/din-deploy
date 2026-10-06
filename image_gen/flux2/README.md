# FLUX.2-klein

CLI image generation with ONNX Runtime.

## Supported models

| Model | Hugging Face ID | Checkpoint | ONNX export |
| --- | --- | --- | --- |
| FLUX.2-klein-4B | `black-forest-labs/FLUX.2-klein-4B` | BF16 | FP32 I/O |
| FLUX.2-klein-4B FP8 | `black-forest-labs/FLUX.2-klein-4b-fp8` | FP8 | FP32 I/O |
| FLUX.2-klein-4B NVFP4 | `black-forest-labs/FLUX.2-klein-4b-nvfp4` | NVFP4 | FP32 I/O |

The CLI supports CPU or TensorRT RTX execution with CPU, CUDA, DirectX, DirectX CIG, and Vulkan processing where available.

## Backends

`CiG` means compute-in-graphics: inference runs in the graphics GPU context so it can coexist with rendering workloads.

| Mode | CLI | Inference and sampling |
| --- | --- | --- |
| Direct TensorRT RTX | `--provider trt-rtx --processing cuda` | TensorRT RTX inference with CUDA sampling. |
| Direct TensorRT RTX | `--provider trt-rtx --processing cpu` | TensorRT RTX inference with CPU sampling. |
| Vulkan interop | `--provider trt-rtx --processing vk` | TensorRT RTX inference with Vulkan sampling; shared buffers and timeline semaphores keep intermediate data on the GPU. |
| DirectX CiG | `--provider trt-rtx --processing dx-cig` | TensorRT RTX inference in a DirectX CiG context with DirectX sampling. |

CiG is useful when inference must share the graphics GPU with rendering. It can reduce standalone inference performance because of graphics-concurrency limits. TensorRT RTX currently documents DirectX CiG support; Vulkan CiG support is planned. See [Simultaneous Compute and Graphics](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/compute-graphics.html).

## Export

```powershell
cd model_export
python export_flux2.py --model_name black-forest-labs/FLUX.2-klein-4b --output D:\models\flux_full --model all --transformer-precision bf16
python quantize_onnx.py --input D:\models\flux_full\transformer_bf16 --quant fp8 --output D:\models\flux_full\transformer_fp8
python quantize_onnx.py --input D:\models\flux_full\transformer_bf16 --quant nvfp4 --output D:\models\flux_full\transformer_nvfp4
```

All precisions share one root directory. The export contains `text_encoder/model.onnx`, `vae_decoder/model.onnx`, and `tokenizer/tokenizer.json` once, plus one transformer per precision: `transformer_bf16/model.onnx`, `transformer_fp8/model.onnx`, and `transformer_nvfp4/model.onnx`.

## Verify

```powershell
python verify_flux2.py --model_name black-forest-labs/FLUX.2-klein-4b --onnx_dir D:\models\flux_full --precision fp8 --output_dir flux2-fp8 --provider trt-rtx
```

## Build

```powershell
cmake --preset windows-x64
cmake --build --preset windows-x64-debug --target din_flux2_cli
```

## Run

```powershell
out\build\windows-x64\bin\Debug\din_flux2_cli.exe --model-dir D:\models\flux_full --precision nvfp4 --provider trt-rtx --processing cuda --prompt "a red fox" --output out
```

`--precision` selects `transformer_<precision>` and defaults to `bf16`. Only the transformer TensorRT RTX runtime-cache and EP-context paths are precision-qualified, so switching precisions never reuses an incompatible compiled transformer engine while the shared text encoder and VAE reuse their existing caches.
