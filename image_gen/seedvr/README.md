# SeedVR2

Image and video super-resolution using ONNX Runtime, TensorRT RTX, and CUDA.
Checkpoint source: `Comfy-Org/SeedVR2` (3B and 7B). Commands run from the repository root.

Start by [exporting models](#export), then [build and run the CLI](#build-and-run).
The same pipeline supports single images and folders of video frames.

> [!IMPORTANT]
> Runtime shapes are fixed at pipeline construction. All frames in a video folder
> must have identical dimensions. The CLI reads dimensions and clip length before
> creating the pipeline. Create a new pipeline when changing dimensions or clip
> length. ONNX spatial axes are symbolic, but runtime profiles and caches are
> specialized to the chosen shape.

## Export

The ONNX export for SeedVR2 requires a ComfyUI source checkout and its dependencies. From the repository root:

```powershell
git clone https://github.com/comfyanonymous/ComfyUI.git external/ComfyUI
python -m pip install -r external/ComfyUI/requirements.txt
python -m pip install -e .
```

ComfyUI supplies the checkpoint-backed model implementation; its UI does not need
to be running.
The exporter downloads the DiT and VAE checkpoints from `Comfy-Org/SeedVR2`
into the shared Hugging Face cache. `--model REPO` accepts a Hugging Face slug;
`--variant 3b|7b` selects its checkpoint (default 3b). Use `--snapshot DIR` for a
local snapshot, or `--model PATH` and `--vae PATH` for explicit checkpoint files.
Export uses **PyTorch Dynamo (`dynamo=True`) and ONNX opset 24 by default for all
models**. Keeping scaled dot-product attention intact allows the exporter to
emit native ONNX `Attention` operators. `--opset` overrides the version; the
exporter requires at least opset 23.

### Export image models

The default is **BF16 DiT weights/computation and FP16 VAE encoder/decoder**,
matching native PyTorch's component precision. DiT inputs/outputs and positive
context remain FP16 at the pipeline boundary; the exporter inserts casts around
the BF16 DiT automatically. RoPE metadata and normalization/attention
accumulation keep their required FP32 precision. `--native-dit-io` is an opt-in
for native-dtype graph I/O when using a separate numerical-testing tool.

```powershell
python image_gen/seedvr/model_export/export_seedvr.py --comfy-root external/ComfyUI --output models/SeedVR2-3B-onnx/seedvr2_3b_bf16_dit.onnx
```

`--components dit context vae_encoder vae_decoder` selects which artifacts to create.
The default exports the DiT and both VAE components; exporting the DiT also writes positive context.
This allows export of separate VAE or context artifacts for use with other existing parts.

#### 7B image precision

For the 7B DiT BF16 computation is even more important than on the 3B variant since we have seen numerical overflow on FP16.
Select the non-sharp 7B checkpoint from the same Hugging Face repository:

```powershell
python image_gen/seedvr/model_export/export_seedvr.py --comfy-root external/ComfyUI --model Comfy-Org/SeedVR2 --variant 7b --component dit --output models/SeedVR2-7B-onnx/seedvr2_7b_bf16_dit.onnx
```

`--component dit` exports only the DiT and its 7B positive context, leaving VAE
files untouched. The VAE is shared with 3B: keep the existing VAE files in the
7B output directory, or copy the encoder/decoder ONNX files and their external
data from the 3B image export once. For video, reuse the video VAE pair instead.
Omit `--component dit` when exporting all components into a new directory.

### Export video models

Use the image export command above with `--video-frames 5`. This selects the
capture example; the exported graphs support dynamic windows of 5, 9, ..., 33
frames. Use a separate video output directory if specifying `--output`, and keep
each ONNX file with its adjacent external-data files.

### Check numerical correctness

Use `verify_seedvr.py` to compare the exported BF16-DiT/FP16-VAE ONNX/RTX pipeline with the
ComfyUI/PyTorch reference on the same input, scale, and seed:

```powershell
python image_gen/seedvr/model_export/verify_seedvr.py --comfy-root external/ComfyUI --onnx-dir models/SeedVR2-3B-onnx --input input.png --scale 2 --seed 0 --output-dir out/seedvr/verify
```

The script reports MSE, MAE, maximum error, and PSNR, and saves reference, runtime,
and comparison images. It uses the TensorRT RTX provider through ONNX Runtime. 
This is mostly a sanity check for the export; with default export arguments we have tested quality.
For video, pass a folder containing a series of frames as PNG.

Note: add `--check-torch-parity` during export to compare the exported DiT ONNX against native PyTorch. 
This optional check is off by default; captured-program and ONNX structural checks remain enabled.

### Optional image DiT quantization

The quantization scripts currently use an FP16 source DiT. Export that variant
explicitly, check it numerically, then create the desired quantized variant:

```powershell
python image_gen/seedvr/model_export/export_seedvr.py --comfy-root external/ComfyUI --dit-dtype fp16 --components dit context --output models/SeedVR2-3B-onnx/seedvr2_3b_fp16_dit.onnx
python image_gen/seedvr/model_export/quantize_onnx.py --quant fp8_wo
python image_gen/seedvr/model_export/quantize_onnx.py --quant nvfp4
```

Provide the exisinting exported high precision ONNX files directory using `--input` and `--output`.
The corresponding checkpoints download into the Hugging Face cache.
Outputs are not overwritten but written with as a separate ONNX file. 
NVFP4 additionally requires `comfy-kitchen`, installed with the ComfyUI dependencies above.

| CLI precision | DiT filename | Representation |
|---|---|---|
| `bf16` (default) | `seedvr2_3b_bf16_dit.onnx` | BF16 weights/computation, FP16 pipeline boundaries |
| `fp16` | `seedvr2_3b_fp16_dit.onnx` | FP16 weights/computation, |
| `fp8_wo` | `seedvr2_3b_fp8_wo_dit.onnx` | FP8 weights and BF16 activations |
| `nvfp4` | `seedvr2_3b_nvfp4_dit.onnx` | FP4 weights with block scales and dynamic block activation quantization |

VAE and conditioning stay FP16. Quantized variants require compatible hardware and provider support. 
NVFP4 uses a TensorRT-specific quantization operator.

## Build and run

After installing the [C++ build prerequisites](../../README.md#requirements),
configure and build the CLI using the matching repository preset. For Windows:

```powershell
cmake --preset windows-x64
cmake --build --preset windows-x64-release --target din_seedvr2_cli
out/build/windows-x64/bin/Release/din_seedvr2_cli.exe input.png --model-dir models/SeedVR2-3B-onnx --output out/seedvr/output --scale 2 --seed 0
```
`--input` can either point to a single image or a folder of video frames.
`--output` should always point to a directory for video and image where each result is saved as `<input-stem>_seedvr.png`.

For video inference we will require chunk based processing, add `--video-frames 9` (or other supported lengths between 5 and 33).
To additionally save memory you can set a VAE encoder and decoder tile size or configure weight streaming for the DiT. See [VAE tiling](#vae-tiling) and [Runtime options](#runtime-options) for details.

The pipeline fixes temporal/spatial dimensions and attention metadata at session creation, 
with  separate VAE contexts for each window size. 
Changing the execution window requires a new pipeline instance and forces recompilation of the EP context files.

### Runtime options

| Option | Behavior |
|---|---|
| `--scale FLOAT` | Output scaling factor; default `2` |
| `--seed N` | Starting-noise seed; default `0` |
| `--rng cpu-std\|cuda-philox\|cpu-philox` | Noise generator; default `cpu-std` |
| `--transformer-precision bf16\|fp16\|fp8_wo\|nvfp4` | DiT selection; default `bf16`; quantized variants are image-only |
| `--video-frames N` | Fix a video window at load time; default five; accepts `5,9,...,33` |
| `--precompile-only` | Prepare video contexts without inference; DiT always uses the built-in dynamic profile |
| `--ws 50%` | DiT weight-streaming budget; default `off` |
| `--color-correction lab\|none` | Default `lab`: retain generated detail while correcting low-frequency color and matching CIELAB histograms |
| `--latent-noise-scale FLOAT` | Conditioning-latent noise in `[0,1]`; default `0` |
| `--ep-cache PATH` | TensorRT RTX runtime-cache directory |
| `--ep-context-dir PATH` | Compiled-context directory |

Weight streaming applies to the DiT. Cached engines/contexts are specific to model
and execution shape. Use separate caches or clear affected entries when replacing
model files. LAB correction requires additional scratch memory.

## VAE tiling

Enable encoder and decoder tiling independently to reduce VAE execution memory:

```powershell
out/build/windows-x64/bin/Release/din_seedvr2_cli.exe input.png --model-dir models/SeedVR2-3B-onnx --output out/seedvr/tiled --scale 2 --vae-encoder-tile-size 512 --vae-encoder-overlap 64 --vae-decoder-tile-size 512 --vae-decoder-overlap 64
```

Sizes and overlaps use **upscaled image pixels**, including decoder settings.
A size of `0` disables tiling. Enabled sizes must be multiples of 16 and at least
32; overlap must be a nonnegative multiple of 16 smaller than the tile size.
Tiles run sequentially and blend into FP32 accumulators before conversion to FP16.
Scratch and stitching buffers are allocated at construction and reused.
Tiling changes the VAE's attention/normalization context and can affect output.
The DiT remains untiled.

## Quality tuning

For original-size restoration, start with `--scale 1 --color-correction lab`.
If generated detail is too strong, compare conditioning-noise scales such as
`0`, `0.05`, `0.10`, and `0.15` with a fixed seed, RNG, and color-correction setting:

```powershell
out/build/windows-x64/bin/Release/din_seedvr2_cli.exe input.png --model-dir models/SeedVR2-3B-onnx --output out/seedvr/restored --scale 1 --seed 0 --latent-noise-scale 0.10
```

This augments the encoded conditioning latent. It leaves the sampler's starting
noise and one-step timestep unchanged and applies to image/video models.
More noise can remove useful detail or change facial features;
inspect images at their output resolution and video in motion. Color correction
can reduce color shifts while retaining generated texture.

## C++ tensor API

`ImagePipeline` and `VideoPipeline` are thin wrappers over a shared implementation.
Their constructors receive fixed source/target dimensions and model settings,
initialize models, allocate internal tensors/bindings, and upload constant context
and attention metadata. Backend initialization can still occur on the first run.

```cpp
din::seedvr::ModelConfig models;
models.model_dir = "models/SeedVR2-3B-onnx";
din::seedvr::ImagePipeline pipeline(models, {512, 768, 1024, 1536});
auto input = pipeline.CreateImage(din::seedvr::ImageRole::Input);
auto output = pipeline.CreateImage(din::seedvr::ImageRole::Output);
// Fill input with RGB float32 CHW values in [0,1].
pipeline.RunImage(input, output);

din::seedvr::DynamicOptionsUpdate update;
update.seed = 42;
update.latent_noise_scale = 0.1f;
pipeline.UpdateOptions(update);
pipeline.RunImage(input, output);
```

