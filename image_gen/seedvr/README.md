# SeedVR2

Image and video super-resolution using ONNX Runtime, TensorRT RTX, and CUDA.
Checkpoint source: `Comfy-Org/SeedVR2` (3B). Commands run from the repository root.

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
into the shared Hugging Face cache. Use `--snapshot DIR` for a local snapshot,
or `--model PATH` and `--vae PATH` for explicit checkpoint files.
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

`--components dit context vae_encoder vae_decoder` selects which artifacts to
create. The default exports the DiT and both VAE components; exporting the DiT
also writes positive context. Select `context` alone to extract conditioning
without exporting the DiT, or select VAE components to export them independently:

```powershell
python image_gen/seedvr/model_export/export_seedvr.py --comfy-root external/ComfyUI --components vae_encoder vae_decoder --output models/SeedVR2-3B-onnx/seedvr2_3b_bf16_dit.onnx
```

#### 7B image precision

The standard non-sharp 7B checkpoint requires the default BF16 DiT computation for good results.
FP16 DiT execution overflows in the text branch and can produce a blurred-looking result after color correction.
Use an explicit 7B `--model` checkpoint and output filename:

```powershell
python image_gen/seedvr/model_export/export_seedvr.py --comfy-root external/ComfyUI --model path/to/seedvr2_7b_fp16.safetensors --output models/SeedVR2-7B-onnx/seedvr2_7b_bf16_dit.onnx
```

### Export video models

Choose a fixed clip length `T = 4n+1`, with `T >= 5`, and a separate model directory
for each clip length:

```powershell
python image_gen/seedvr/model_export/export_seedvr.py --comfy-root external/ComfyUI --video-frames 5 --output models/SeedVR2-3B-video/seedvr2_video_dit.onnx
```

`--video-frames 1` is the default image mode. Video exports include a frame-count
manifest used by the CLI. BF16 is also the default for video DiTs; FP16 remains
available with `--dit-dtype fp16` when exporting and
`--transformer-precision fp16` when running or verifying older FP16 exports.

| Artifact | Image export | Video export |
|---|---|---|
| DiT | `seedvr2_3b_bf16_dit.onnx` | `seedvr2_video_dit.onnx` |
| Encoder | `seedvr2_ema_vae_encoder.onnx` | `seedvr2_video_vae_encoder.onnx` |
| Decoder | `seedvr2_ema_vae_decoder.onnx` | `seedvr2_video_vae_decoder.onnx` |
| Positive context | `seedvr2_3b_fp16_positive_context.bin` | `seedvr2_3b_fp16_positive_context.bin` |
| Clip length | One image | `seedvr2_video_frames.txt` |

Keep each ONNX file with its adjacent external-data files. Batch, channels,
temporal length, and `58x5120` text context are fixed; spatial axes and attention
metadata lengths are symbolic. `--latent-height` / `--latent-width` and
`--image-height` / `--image-width` select export example shapes, rather than
fixing the runtime resolution. The decoder exports unclamped values; the runtime
clamps after tile blending.

### Check numerical correctness

Use `verify_seedvr.py` to compare the exported BF16-DiT/FP16-VAE ONNX/RTX pipeline with the
ComfyUI/PyTorch reference on the same input, scale, and seed:

```powershell
python image_gen/seedvr/model_export/verify_seedvr.py --comfy-root external/ComfyUI --onnx-dir models/SeedVR2-3B-onnx --input input.png --scale 2 --seed 0 --output-dir out/seedvr/verify
```

The script reports MSE, MAE, maximum error, and PSNR, and saves reference, runtime,
and comparison images. It uses the TensorRT RTX provider with CPU fallback
disabled. Compare representative inputs at the resolutions and clip lengths you
intend to use, and inspect the reported errors and images. A successful export
alone does not establish numerical correctness in the target runtime.

For video, pass a folder containing up to one exported clip and select its video
model directory. The script repeats the final frame when necessary and compares
all real frames. `--pytorch-only` saves only the reference result.

### Optional image DiT quantization

The quantization scripts currently use an FP16 source DiT. Export that variant
explicitly, check it numerically, then create the desired quantized variant:

```powershell
python image_gen/seedvr/model_export/export_seedvr.py --comfy-root external/ComfyUI --dit-dtype fp16 --components dit context --output models/SeedVR2-3B-onnx/seedvr2_3b_fp16_dit.onnx
python image_gen/seedvr/model_export/quantize_onnx.py --quant fp8_wo
python image_gen/seedvr/model_export/quantize_onnx.py --quant nvfp4
```

Defaults read/write `models/SeedVR2-3B-onnx`; override with `--input` and `--output`.
The corresponding checkpoints download into the Hugging Face cache at the script's
pinned revision. Use `--checkpoint` for a local file or `--revision` to override.
Outputs are not overwritten. NVFP4 additionally requires `comfy-kitchen`, installed
with the ComfyUI dependencies above.

| CLI precision | DiT filename | Representation |
|---|---|---|
| `bf16` (default) | `seedvr2_3b_bf16_dit.onnx` | BF16 weights/computation, FP16 pipeline boundaries |
| `fp16` | `seedvr2_3b_fp16_dit.onnx` | FP16 reference |
| `fp8_wo` | `seedvr2_3b_fp8_wo_dit.onnx` | FP8 weights and FP16 activations |
| `nvfp4` | `seedvr2_3b_nvfp4_dit.onnx` | FP4 weights with block scales and dynamic block activation quantization |

VAE and conditioning stay FP16. Quantized variants require compatible hardware
and provider support. NVFP4 uses a TensorRT-specific quantization operator.
Compare quality and performance against FP16 for your workload.

## Build and run

After installing the [C++ build prerequisites](../../README.md#requirements),
configure and build the CLI using the matching repository preset. For Windows:

```powershell
cmake --preset windows-x64
cmake --build --preset windows-x64-release --target din_seedvr2_cli
out/build/windows-x64/bin/Release/din_seedvr2_cli.exe input.png --model-dir models/SeedVR2-3B-onnx --output out/seedvr/output --scale 2 --seed 0
```

`--output` always selects a directory (default `out/seedvr/seedvr-cpp`). Each result
is saved as `<input-stem>_seedvr.png`, including single-image inputs.

For a filename-sorted folder of video frames:

```powershell
out/build/windows-x64/bin/Release/din_seedvr2_cli.exe path/to/frames --model-dir models/SeedVR2-3B-video --output out/seedvr/video --scale 1 --seed 0
```

The pipeline resizes input with CUDA, pads spatial dimensions to multiples of 16,
then runs **VAE encoder → one DiT pass → Euler update → VAE decoder**. CUDA handles
conditioning, tile blending, and color correction. Models and internal buffers
remain resident and are reused.

The padded target must be at least `32x32` and fit within `3840x2160` in landscape
or `2160x3840` in portrait. Padding is cropped afterward, and final odd dimensions
are rounded down to even values.

### Video chunking

Folder input reads top-level PNGs in filename order; use zero-padded filenames.
Outputs are `<input-stem>_seedvr.png`. Consecutive clips overlap by one frame,
keeping its first prediction. The CLI repeats the final input to complete a short
last clip and discards padded predictions. A one-PNG folder still uses video mode.

For `T` input frames, the VAE latent has `(T-1)/4+1` time steps. Spatial VAE tiles
span the full clip. The DiT processes the complete latent; there is no persistent
VAE state or cross-clip blending. Longer clips require more memory.

### Runtime options

| Option | Behavior |
|---|---|
| `--scale FLOAT` | Output scaling factor; default `2` |
| `--seed N` | Starting-noise seed; default `0` |
| `--rng cpu-std\|cuda-philox\|cpu-philox` | Noise generator; default `cpu-std` |
| `--transformer-precision bf16\|fp16\|fp8_wo\|nvfp4` | DiT selection; default `bf16`; quantized variants are image-only |
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

`CreateImage(role, ImageDevice::Cuda)` allocates a frame on the pipeline's CUDA
device. Caller-supplied dense float32 `[3,H,W]` `Ort::Value` tensors are also
accepted on CPU or that device. Inputs use source dimensions; outputs use target dimensions rounded
down to even values. Runs fill output storage without replacing it. CUDA inputs
must be ready on entry; resizing and output conversion run on CUDA.

`VideoPipeline(models, {image_shape, T})` requires `T=4n+1`, `T>=5`.
`RunChunk(inputs, outputs)` takes exactly `T` input and output frame tensors.
The image class exposes only `RunImage()`; the video class exposes only `RunChunk()`.
File handling, temporal padding, and overlapping chunk selection belong to callers.

`UpdateOptions()` changes only supplied fields: seed, RNG, conditioning noise,
color correction, and weight-streaming budget. Shapes, tiling, model selection,
and backend settings are fixed. Budget updates require streaming enabled at
construction and can recreate provider execution contexts.

Runs are synchronous and instances are not thread-safe. Factory-created tensors
must be destroyed before their pipeline. Pipeline-owned tensor storage is allocated
at construction; caller-created images and backend workspace requests are separate.
Construction selects the CUDA device matching the selected ORT device once. Keep
that device current on the calling thread until the pipeline and its factory tensors
are destroyed.

## C++ checks

With `DIN_BUILD_TESTING=ON`, build and run the API and conditioning-noise checks:

```powershell
cmake --build --preset windows-x64-release --target din_seedvr2_pipeline_tests din_seedvr2_noise_tests
ctest --test-dir out/build/windows-x64 -C Release -R din_seedvr2_ --output-on-failure
```

For model-backed API checks, pass image and video model directories to
`din_seedvr2_pipeline_tests`. This covers repeated calls, CPU/CUDA tensors,
option updates, shape validation, and VAE tiling.

## Temporary workaround: native cuDNN VAE

The native cuDNN VAE is a temporary workaround until TensorRT's VAE execution
catches up. It replaces the VAE encoder and decoder; the DiT continues to run
through ONNX Runtime and TensorRT RTX. The standard workflow uses ONNX for all
three models.

This section contains the preparation, build, execution, and validation steps for
the workaround. It requires the repository's C++/CUDA toolchain with cuDNN, a
Python environment with the exporter dependencies, and a ComfyUI source checkout
at `external/ComfyUI`.

### Prepare models and weights

Export a complete five-frame video model directory, then extract the VAE weights
and architecture into a new directory:

```powershell
python image_gen/seedvr/model_export/export_seedvr.py --comfy-root external/ComfyUI --video-frames 5 --output models/SeedVR2-3B-video/seedvr2_video_dit.onnx
python image_gen/seedvr/model_export/prepare_native_vae.py --source models/SeedVR2-3B-video --output models/SeedVR2-native-vae
```

The preparation script requires the exported **video VAE pair**, creates its
output directory, and refuses to overwrite it. The extracted weights support
both image and video execution. Image runs still need an image DiT and positive
context in their runtime model directory.

| Script | Purpose |
|---|---|
| [`prepare_native_vae.py`](model_export/prepare_native_vae.py) | Extract weights and architecture from the video ONNX VAE encoder/decoder into the native backend's directory format. |
| [`verify_native_vae.py`](model_export/verify_native_vae.py) | Compare native encoder/decoder outputs numerically with source ONNX outputs using FP16 inputs and numerical tolerances. `--image` selects image-mode checks against image ONNX exports. |

### Build and run

Enable `DIN_SEEDVR_NATIVE_VAE` explicitly and build the pipeline and standalone
VAE CLIs. For Windows:

```powershell
cmake --preset windows-x64 -DDIN_SEEDVR_NATIVE_VAE=ON
cmake --build --preset windows-x64-release --target din_seedvr2_cli din_seedvr2_vae_cli
out/build/windows-x64/bin/Release/din_seedvr2_cli.exe path/to/frames --model-dir models/SeedVR2-3B-video --output out/seedvr/native-video --scale 1 --vae-backend cudnn --vae-weights models/SeedVR2-native-vae
```

Select the workaround with `--vae-backend cudnn --vae-weights DIR`. For an image,
pass its input path and image model directory with the same backend options.

### Verify numerical correctness

Compare the standalone native VAE with the source ONNX encoder and decoder:

```powershell
python image_gen/seedvr/model_export/verify_native_vae.py --source models/SeedVR2-3B-video --weights models/SeedVR2-native-vae --exe out/build/windows-x64/bin/Release/din_seedvr2_vae_cli.exe --output out/seedvr/native-verify
```

The script reports finite-value checks, maximum and mean absolute error, and
elements outside its numerical tolerances. Add `--image` and supply an image
ONNX model directory for image-mode checks. For pipeline API coverage, supply
native weights as the third argument to `din_seedvr2_pipeline_tests`, after image
and video model directories; an optional fourth argument selects a CUDA ordinal.

### Tuning and memory

`--vae-autotune on` benchmarks supported cuDNN plans; tuning is off by default.
`--vae-tuning-candidates N` controls the search size (default 5),
`--vae-workspace-mib N` limits each plan's workspace (default 256), and
`--vae-tuning-cache DIR` stores measured choices. Tune while the device is idle.
Encoder and decoder share working memory according to their peak requirements;
weights, persistent I/O, and allocator overhead consume additional memory.
