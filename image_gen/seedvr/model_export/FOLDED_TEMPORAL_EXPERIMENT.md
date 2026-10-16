# Single-frame VAE temporal-convolution experiment

`export_seedvr_folded_temporal.py` is an experimental copy of `export_seedvr.py`.
It retains Dynamo ONNX export and the regular FP32 attention path. For a
one-frame causal convolution, it precomputes the sum of the temporal kernel
slices and exports a spatial convolution. The baseline exporter is unchanged.

Run both exporters from the repository root with `--comfy-root out/seedvr/ComfyUI`
and `--components vae_encoder`, using 32x32 example images and distinct
`--vae-encoder-output` paths. The checkpoint comes from the global Hugging Face cache.
`compare_folded_temporal.py` compares the resulting ONNX files with ONNX Runtime
on CPU at three image sizes.

| 1080p TensorRT RTX 1.7.1.116 | Baseline | Folded |
| --- | ---: | ---: |
| Full-resolution three-frame layer outputs | 5 | 0 |
| Engine size | 209.78 MiB | 75.28 MiB |
| Execution-context device memory | 8262.16 MiB | 8325.44 MiB |
| TensorRT-managed GPU memory after context creation | 8464 MiB | 8394 MiB |
| Short-run mean latency | 540.40 ms | 239.46 ms |

The context memory **increases** by 63.28 MiB. Total TensorRT-managed GPU memory
after context creation falls by approximately 70 MiB because the folded engine
holds fewer weights. The FP32 attention matrices remain, and both exports fail
to build at 3840x1632. The folded export's 4K build requested 68,854,439,040
bytes before reporting out of memory.

CPU ONNX Runtime comparison of the same deterministic image gave maximum
absolute differences of 0.001953 at 32x32, 0.002045 at 128x128, and 0.003906
at 512x512. At 512x512, 2 of 65,536 outputs exceeded
`atol=0.002, rtol=0.002`. The export's 32x32 eager parity and ONNX structural
validation passed. This experiment is not yet an exact numerical replacement.

## Verify script accuracy check

`verify_seedvr.py` was run on `out/seedvr/input.png` at scale 1, seed 0, and
without color correction. Only the VAE encoder ONNX path differed between the
two ONNX pipelines; both used the same DiT and decoder. The PyTorch reference
images were byte-identical. The folded run repeated with the same result.

| 128x128 final image vs. PyTorch | Unfolded | Folded |
| --- | ---: | ---: |
| MSE (8-bit RGB) | 32.0618 | 41.9898 |
| MAE (8-bit RGB) | 2.4152 | 2.6635 |
| PSNR | 33.07 dB | 31.90 dB |

The verify script's encoder-only mode also compared latents directly with the
ComfyUI VAE. The folded encoder was closer at both tested shapes:

| Image shape | Unfolded latent MAE | Folded latent MAE |
| --- | ---: | ---: |
| 128x128 | 0.00420434 | 0.00135356 |
| 512x512 | 0.00541649 | 0.00266245 |

Thus, improved encoder latent MAE did not improve this final image's PSNR after
denoising. The full pipeline test covers one small image; it does not establish
accuracy across other images or resolutions.
