# SeedVR2

Image super-resolution with ONNX Runtime, TensorRT RTX, and CUDA.

## Supported models

| Model      | Hugging Face ID     | Checkpoint | ONNX export                    |
|------------|---------------------|------------|--------------------------------|
| SeedVR2 3B | `Comfy-Org/SeedVR2` | FP16       | FP16, dynamic height and width |
| SeedVR2 3B FP8 | `Comfy-Org/SeedVR2` | FP8 | Weight-only FP8 DiT, FP16 activations |
| SeedVR2 3B NVFP4 | `Comfy-Org/SeedVR2` | NVFP4 | NVFP4 DiT with TensorRT dynamic activation quantization |

The pipeline supports one image, one video frame, and the 3B checkpoint. Image
dimensions are padded to multiples of 16 before VAE encoding. The C++ target
uses no Vulkan or DirectX components.

## Export

Run all commands from the repository root. Paths are relative and use forward
slashes. Use a Python environment with the export dependencies installed.
Replace `input.png` in the examples with your input image.

Check out ComfyUI at the path used by the commands below:

```powershell
git clone https://github.com/comfyanonymous/ComfyUI.git out/seedvr/ComfyUI
```

Then run the exporter:

```powershell
python image_gen/seedvr/model_export/export_seedvr.py --comfy-root out/seedvr/ComfyUI --output models/SeedVR2-3B-onnx/seedvr2_3b_fp16_dit.onnx --decompose-vae-attention
```

You can specify a local Hugging Face snapshot of `Comfy-Org/SeedVR2` using
`--snapshot`, or omit it to download the checkpoints into the global cache.

The default export writes ONNX opset 23 models to
`models/SeedVR2-3B-onnx`:

- `seedvr2_3b_fp16_dit.onnx` and its external data
- `seedvr2_ema_vae_encoder.onnx`
- `seedvr2_ema_vae_decoder.onnx`
- `seedvr2_3b_fp16_positive_context.bin`

The VAE image axes and DiT latent axes are dynamic. Batch size, temporal
length, channels, and the `58x5120` text context remain fixed. The DiT receives
normal and shifted window/RoPE metadata as additional dynamic-length inputs.
For one-frame VAE inputs, the exporter folds repeated causal-convolution
frames into precomputed spatial kernels. This removes the expanded temporal
activations and changes FP16 accumulation slightly.
The default VAE export retains fused ONNX `Attention`. To export attention as
matmul and softmax for the ONNX Runtime TensorRT RTX execution provider, pass
`--decompose-vae-attention`. The standalone TensorRT RTX CLI accepts native
encoder `Attention`, but the ONNX Runtime provider failed to build it at
128×128 and 512×512 in the encoder patch audit.

Select individual artifacts with `--components dit context vae_encoder
vae_decoder`.

## Quantize the DiT

Convert the existing FP16 DiT with the Comfy-Org FP8 or NVFP4 checkpoint:

```powershell
python image_gen/seedvr/model_export/quantize_onnx.py --input models/SeedVR2-3B-onnx/seedvr2_3b_fp16_dit.onnx --output models/SeedVR2-3B-onnx/seedvr2_3b_fp8_wo_dit.onnx --quant fp8_wo

python image_gen/seedvr/model_export/quantize_onnx.py --input models/SeedVR2-3B-onnx/seedvr2_3b_fp16_dit.onnx --output models/SeedVR2-3B-onnx/seedvr2_3b_nvfp4_dit.onnx --quant nvfp4
```

The script downloads only the selected checkpoint into the global Hugging Face
cache, using a pinned revision. Pass `--checkpoint` for a local safetensors file
or `--revision` to select a different repository revision. Dependencies are
`torch`, `onnx>=1.18`, `safetensors`, and `huggingface_hub`, plus `comfy-kitchen`
for NVFP4 packing and scale-layout conversion.

FP8 is weight-only: it uses the checkpoint's linear weights with dequantization
scale 1.0 and leaves activations unchanged, with no activation Q/DQ nodes.
NVFP4 preserves the checkpoint's FP8 block scales and
FP32 global weight scales, unpacks their cuBLAS scale layout, and converts the
FP4 nibble order to ONNX. Since this checkpoint has no activation scales, the
script uses a global activation scale of 1.0. TensorRT RTX 1.7.1 requires that
scale to be a constant initializer; block activation scales are still computed
by `TRT_FP4DynamicQuantize`. This differs from ComfyUI's runtime global-scale
calculation. If a checkpoint supplies `input_scale`, the converter preserves it.

The converter targets named `Gemm(transB=1)` weights from `export_seedvr.py`.
Shared weights are stored/dequantized once. FP8 replaces 216 linear weights
used by 322 calls; NVFP4 replaces 210 weights used by 316 calls, keeping six
input/output/embedding projections in FP16. Biases, normalization/modulation
parameters, conditioning, and VAE models retain their baseline values. This
is linear-layer quantization, not a full reproduction of ComfyUI's FP8 model,
which also rounds nonlinear parameters to FP8.

Each output has a separate `<output>.data` file. The script streams external
data, checks the staged ONNX model, and refuses to overwrite existing output
files. Keep the ONNX file and its data file together. Export, verification,
quantization, and the C++ CLI default to `models/SeedVR2-3B-onnx`, relative to
the working directory. Quantization defaults its output beside the input;
with the default paths, only `--quant fp8_wo` or `--quant nvfp4` is needed.
Use `--output` for export, `--input`/`--output` for quantization, `--onnx-dir`
for verification, or `--model-dir` for the CLI to override model locations.
No machine-specific directory is required.

The CLI selects the DiT with `--transformer-precision`:

| Value | DiT filename |
|-------|--------------|
| `fp16` (default) | `seedvr2_3b_fp16_dit.onnx` |
| `fp8_wo` | `seedvr2_3b_fp8_wo_dit.onnx` |
| `nvfp4` | `seedvr2_3b_nvfp4_dit.onnx` |

All three share the VAE and positive-context files in the model directory.
DiT runtime caches include the precision to prevent cross-precision reuse.

The earlier FP8 activation-Q/DQ variant and NVFP4 compile and deserialize with TensorRT RTX 1.7.1.116 using explicit
min/opt/max shapes for all 20 inputs at a 64x64 latent resolution (512x512
image). Compilation used `--skipInference`; the saved engines have fixed
profiles for that resolution. Reproduction commands, exact metadata shapes,
and build logs are under `out/seedvr/quantized/compile`.

The C++ pipeline was also tested on two center-cropped/downsampled 256x256
inputs (the bicycle image and video frame 0001), with 2x upscale, seed 0,
`cpu-std` RNG, no VAE tiling, and `--color-correction none`. All variants used
the same VAE/context files and separate caches. The earlier FP8 activation-Q/DQ
variant produced entirely black outputs on both inputs despite successful
execution. The current weight-only FP8 graph was retested with the same C++
binary, inputs, VAE/context, and settings in a fresh cache. It produces normal
images on both samples: PSNR versus FP16 is 27.23 dB (bicycle) and 34.06 dB
(frame), with mean absolute pixel errors of 5.19 and 2.77 on a 0-255 scale.
Removing activation Q/DQ eliminates the black-output failure on these samples.
The weight-only results and comparison are saved as `metrics_weight_only.json`
and `comparison_weight_only.png` in the quality directory below.
All three models were also rerun with `--color-correction lab`, keeping the
other settings identical. Against LAB-corrected FP16, weight-only FP8 gives
27.85/34.35 dB PSNR (bicycle/frame), and NVFP4 gives 21.29/26.90 dB.
LAB results, commands, logs, and comparisons are in the `lab` subdirectory.
NVFP4 produced recognizable images with
visible differences from FP16: PSNR versus FP16 was 20.62 dB (bicycle) and
26.61 dB (frame); mean absolute pixel errors were 11.95 and 6.54 on a 0-255
scale. These measure baseline agreement, not ground-truth reconstruction
quality, and do not establish quality across a broader dataset.

NVFP4 custom-op outputs have explicit FP4/FP8 type annotations because ONNX
Runtime cannot infer them from the custom schema. The comparison image,
metrics, input provenance, commands, and logs are saved under
`out/seedvr/quantized/quality`. Performance has not been benchmarked. NVFP4
uses a TensorRT custom operator and cannot run as-is in a generic ONNX CPU
runtime.

## Verify

To run one frame through the ComfyUI PyTorch pipeline only, using checkpoints
from the global Hugging Face cache:

```powershell
python image_gen/seedvr/model_export/verify_seedvr.py --comfy-root out/seedvr/ComfyUI --input input.png --scale 2 --seed 0 --pytorch-only --output-dir out/seedvr/pytorch
```

The result is `out/seedvr/pytorch/pytorch_output.png`.

Compare the full ComfyUI PyTorch pipeline with the three FP16 ONNX components
running through TensorRT RTX:

```powershell
python image_gen/seedvr/model_export/verify_seedvr.py --comfy-root out/seedvr/ComfyUI --onnx-dir models/SeedVR2-3B-onnx --input input.png --output-dir out/seedvr/verify
```

The verifier writes `pytorch_output.png`, `ort_output.png`, and
`comparison.png`, then reports MSE, MAE, maximum pixel error, and PSNR. CPU
execution-provider fallback is disabled for the ONNX path.

## Build

```powershell
cmake --preset windows-x64
cmake --build --preset windows-x64-release --target din_seedvr2_cli
```

## Run

For example, select weight-only FP8 with LAB correction (the default):

```powershell
out/build/windows-x64/bin/Release/din_seedvr2_cli.exe input.png --model-dir models/SeedVR2-3B-onnx --transformer-precision fp8_wo --output out/seedvr/fp8_wo.png
```

For FP16 with 2x upscale:

```powershell
out/build/windows-x64/bin/Release/din_seedvr2_cli.exe input.png --model-dir models/SeedVR2-3B-onnx --transformer-precision fp16 --output out/seedvr/seedvr-2x-cpp.png --scale 2 --seed 0
```

Pass a directory in place of the PNG to process its top-level PNG files in
filename order. For directory input, `--output` names an output directory;
each result is saved as `<input-stem>_seedvr.png`.
The default `--color-correction lab` matches ComfyUI's SeedVR postprocessing:
CUDA blends the generated high-frequency detail with the resized input's
low-frequency color, then matches CIELAB channel histograms. Use
`--color-correction none` for the raw decoded output. LAB runs after the final
decoder clamp and tile blending, and uses additional GPU scratch memory.
The pipeline reuses its encoder, DiT, and decoder sessions while consecutive
images have the same padded dimensions. A later size change raises an error;
start a separate CLI process for a different padded shape.

Image resize and window/RoPE metadata generation run on the CPU. The VAE
encoder, DiT, and VAE decoder run through TensorRT RTX; condition packing and
the Euler update run as CUDA kernels on the same ONNX Runtime compute stream.
TensorRT RTX runtime caches are stored under `out/seedvr/trt_rtx_cache_cpp` by
default, and compiled-model contexts under `out/seedvr/ep_context_cpp`.
Clear affected caches and contexts after re-exporting models in place.

Use `--ws 50%` to pass a weight-streaming budget to the DiT's TensorRT RTX
provider as `nv_weight_streaming_budget`. The value is forwarded as a string;
`--ws off` (the default) omits the option.
On a 24 GB GPU, a 3840x1632 padded image required `--ws 50%` to leave enough
memory for VAE encoder attention.

## CUDA VAE tiling

Encoder and decoder tiling are independently opt-in. All CLI sizes are in
**upscaled image pixels**, including decoder sizes. For example, a decoder
tile size of 512 uses a 64x64 latent tile. Sizes and overlaps must be multiples
of 16; enabled sizes must be at least 32 and overlap must be smaller than size.
Size 0 retains the full-frame path. Default overlap is 64.

`seedvr2_ema_vae_decoder.onnx` always exports unclamped output. Both tiled and
full-frame decoding use this same model. A CUDA kernel clamps the final image to
`[0,1]` after decoding and any tile blending. Optional LAB correction runs
before copying the result to the CPU.
Re-export older decoders with `--components vae_decoder --decompose-vae-attention`;
clipping inside an old graph cannot be undone by the pipeline. The decoder uses
the `vae_decoder_unclamped_` runtime cache namespace to avoid old full-frame caches.

```powershell
out/build/windows-x64/bin/Release/din_seedvr2_cli.exe input.png --model-dir models/SeedVR2-3B-onnx --output out/seedvr/tiled.png --scale 2 --seed 0 --ws 50% --vae-encoder-tile-size 512 --vae-encoder-overlap 64 --vae-decoder-tile-size 512 --vae-decoder-overlap 64
```

CUDA gathers contiguous tiles, blends them with separable cosine weights into
FP32 device accumulators, and normalizes the result to FP16. Half-sample cosine
ramps keep every covered pixel's weight positive. Tiles run sequentially on the
ORT compute stream. The last tile shifts back to the image boundary, so its
actual overlap can exceed the requested overlap. Images smaller than a tile use
their padded dimensions; a single tile bypasses stitching. This differs from
ComfyUI's smaller edge tiles and endpoint-inclusive ramps.

VAE scratch buffers are released between stages. Model sessions remain resident;
the DiT still processes the complete latent image, and existing resolution limits
still apply. Full input/output tensors and FP32 stitching canvases remain on the
GPU, but VAE intermediate activations are limited to one tile. Spatial attention
and normalization change with tile context, so tiled output can differ in color
and detail from full-frame output even with seamless stitching.

Build `din_seedvr2_tiling_test` and run
`ctest --test-dir out/build/windows-x64 -C Release -R din_seedvr2_tiling --output-on-failure`
to check CUDA gathering, reconstruction, weighting against a CPU reference,
clipping, edge coverage, latent alignment, and invalid settings. A CUDA GPU is required.
