# SeedVR2

Image and video super-resolution using ONNX Runtime, TensorRT RTX, and CUDA.
Checkpoint source: `Comfy-Org/SeedVR2` (3B). Commands run from the repository root.

> [!WARNING]
> **Dynamic runtime shapes are not supported. All video frames must have identical
> dimensions.** The first frame fixes session profiles (`min = opt = max`); a later
> padded-resolution change raises an error. Use a separate process per resolution.
> ONNX spatial axes are dynamic, but TRT engines/contexts are shape-specific:
> a single cached engine cannot serve arbitrary resolutions. Matching-shape caching
> and session reuse remain supported. Clear affected caches after replacing models.

## Export

Use the `modelopt` environment with export dependencies and a ComfyUI source checkout:

```powershell
git clone https://github.com/comfyanonymous/ComfyUI.git out/seedvr/ComfyUI
conda run -n modelopt python image_gen/seedvr/model_export/export_seedvr.py --comfy-root out/seedvr/ComfyUI --output models/SeedVR2-3B-onnx/seedvr2_3b_fp16_dit.onnx
```

Dynamo exports opset 23 with external weights. Checkpoints use the global Hugging
Face cache; `--snapshot` accepts a local snapshot. `--components dit context
vae_encoder vae_decoder` selects artifacts.

| Artifact | Image export |
|---|---|
| DiT | `seedvr2_3b_fp16_dit.onnx` |
| Encoder | `seedvr2_ema_vae_encoder.onnx` |
| Decoder | `seedvr2_ema_vae_decoder.onnx` |
| Positive context | `seedvr2_3b_fp16_positive_context.bin` |

Keep ONNX and external-data files together. Batch, channels, temporal length, and
`58x5120` text context are fixed; spatial axes and window/RoPE metadata lengths are
symbolic. The image VAE folds repeated causal-convolution frames into spatial kernels.

VAE export defaults to fused ONNX `Attention` for the current RTX 1.7.1 C++ runtime.
`--decompose-vae-attention` is an opt-in compatibility workaround for older providers;
Python ORT packages bundling RTX 1.5 can reject fused attention even with fixed profiles.
The decoder exports unclamped values; the runtime clamps after tile blending.

## Build and run

```powershell
cmake --preset windows-x64
cmake --build --preset windows-x64-release --target din_seedvr2_cli
out/build/windows-x64/bin/Release/din_seedvr2_cli.exe input.png --model-dir models/SeedVR2-3B-onnx --output out/seedvr/output.png --scale 2 --seed 0
```

The CPU resizes input and generates window/RoPE metadata. Images are padded to
multiples of 16 before encoding. Encoder, DiT, and decoder execute through RTX;
condition packing, Euler update, tile blending, and postprocessing use CUDA on the
ORT compute stream. Sessions remain resident and are reused.

| Option | Behavior |
|---|---|
| `--transformer-precision fp16\|fp8_wo\|nvfp4` | Image DiT selection; default `fp16` |
| `--ws 50%` | Forward string to DiT EP as `nv_weight_streaming_budget`; default `off` omits it |
| `--color-correction lab\|none` | Default `lab`: combine generated detail with input low-frequency color, then match CIELAB histograms |
| `--ep-cache PATH` | Runtime cache; default `out/seedvr/trt_rtx_cache_cpp` |
| `--ep-context-dir PATH` | Compiled contexts; default `out/seedvr/ep_context_cpp` |

Weight streaming applies only to the DiT. LAB correction follows final decoder
clamping to `[0,1]` and requires additional GPU scratch space. Cache namespaces
separate DiT precisions.

## VAE tiling

```powershell
out/build/windows-x64/bin/Release/din_seedvr2_cli.exe input.png --output out/seedvr/tiled.png --scale 2 --ws 50% --vae-encoder-tile-size 512 --vae-encoder-overlap 64 --vae-decoder-tile-size 512 --vae-decoder-overlap 64
```

Encoder and decoder tiling are independently opt-in (`size=0` disables). All sizes
are **upscaled image pixels**, including decoder sizes: 512 pixels map to 64 latent
pixels. Sizes/overlaps must be multiples of 16; enabled sizes must be at least 32,
with overlap smaller than size (default overlap: 64).

Tiles execute sequentially. CUDA gathers tiles, blends separable half-sample cosine
weights into FP32 accumulators, and normalizes to FP16. Boundary tiles shift inward;
actual overlap may exceed the requested value. A single tile bypasses stitching.
VAE scratch is released between stages; full input/output tensors and stitching
canvases remain resident. The DiT always processes the complete latent tensor.
Tiling changes attention/normalization context and can change color and detail;
its boundaries and blending differ from ComfyUI.

## Video

Export a fixed clip length `N = 4n+1` into a separate directory:

```powershell
conda run -n modelopt python image_gen/seedvr/model_export/export_seedvr.py --video-frames 5 --output models/SeedVR2-3B-video/seedvr2_video_dit.onnx
out/build/windows-x64/bin/Release/din_seedvr2_cli.exe path/to/frames --model-dir models/SeedVR2-3B-video --output out/seedvr/video --scale 1 --ws 50% --vae-encoder-tile-size 512 --vae-decoder-tile-size 512
```

Artifacts: `seedvr2_video_{vae_encoder,dit,vae_decoder}.onnx`, external weights,
positive context, and `seedvr2_video_frames.txt`. Use one directory per clip length.
Default `--video-frames 1` produces image models. Video supports FP16 DiT only.

| Stage | Tensor shape |
|---|---|
| Encoder input | `[1,3,N,H,W]` |
| Latent / DiT | `[1,16,T,H/8,W/8]`, `T = (N-1)/4+1` |
| Decoder output | `[1,3,N,H,W]` |

Folder input processes top-level PNGs in filename order; use zero-padded names.
Outputs are `<input-stem>_seedvr.png`. The manifest determines clip length. Clips
advance by `N-1` frames, retaining the first prediction of the shared boundary
frame. The final clip repeats its last frame to fill `N`; padded outputs are discarded.
A one-PNG folder still uses video mode.

The native causal VAE includes temporal convolutions. Spatial VAE tiles span the
full clip; the DiT is neither spatially nor temporally tiled. There is no persistent
VAE state or cross-clip blending. DiT temporal window width is one for `N <= 13`;
`N >= 17` enables attention windows spanning multiple latent time steps. Longer
clips increase memory consumption.

## DiT quantization

Convert an FP16 image DiT using the corresponding Comfy-Org checkpoint:

```powershell
conda run -n modelopt python image_gen/seedvr/model_export/quantize_onnx.py --quant fp8_wo
conda run -n modelopt python image_gen/seedvr/model_export/quantize_onnx.py --quant nvfp4
```

Defaults read/write `models/SeedVR2-3B-onnx`; override with `--input`/`--output`.
Checkpoints download at a pinned revision into the global cache; use `--checkpoint`
or `--revision` to override. Existing outputs are not overwritten. Dependencies:
`torch`, `onnx>=1.18`, `safetensors`, `huggingface_hub`; NVFP4 additionally needs
`comfy-kitchen`.

| CLI precision | DiT filename | Representation |
|---|---|---|
| `fp16` | `seedvr2_3b_fp16_dit.onnx` | Baseline |
| `fp8_wo` | `seedvr2_3b_fp8_wo_dit.onnx` | FP8 weights, scale 1.0, FP16 activations; no activation Q/DQ |
| `nvfp4` | `seedvr2_3b_nvfp4_dit.onnx` | FP4 weights, FP8 block scales, FP32 global scales; dynamic block activation quantization |

Quantization targets named `Gemm(transB=1)` weights: FP8 replaces 216 shared
weights/322 calls; NVFP4 replaces 210/316, retaining six boundary/embedding
projections in FP16. VAE, context, biases, and nonlinear parameters are unchanged.
This does not reproduce ComfyUI's nonlinear-parameter FP8 rounding.

NVFP4 converts checkpoint scale layout and nibble order to ONNX. RTX 1.7.1 requires
a constant global activation scale (checkpoint `input_scale`, otherwise 1.0),
unlike ComfyUI's runtime calculation. Block scales use `TRT_FP4DynamicQuantize`;
explicit FP4/FP8 output annotations support ORT type inference. This custom operator
requires TensorRT. Quantized outputs can differ substantially from FP16; benchmark
quality and performance for the target workload.

## Verification

```powershell
conda run -n modelopt python image_gen/seedvr/model_export/verify_seedvr.py --comfy-root out/seedvr/ComfyUI --onnx-dir models/SeedVR2-3B-onnx --input input.png --output-dir out/seedvr/verify
```

Compares native ComfyUI/PyTorch against FP16 ONNX/RTX with CPU fallback disabled;
reports MSE, MAE, maximum error, and PSNR, and writes `pytorch_output.png`,
`ort_output.png`, and `comparison.png`. Add `--pytorch-only` for the reference alone.
Folder input accepts up to one exported video clip, repeats the final frame as
needed, and compares every real output frame. Check the Python provider's bundled
RTX version when verifying fused attention.
