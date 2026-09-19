# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# !/usr/bin/env python3
"""Compare the ComfyUI SeedVR2 pipeline with its ONNX Runtime export."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from export_seedvr import (
    CONTEXT_TOKENS,
    CONTEXT_WIDTH,
    DEFAULT_SNAPSHOT,
    DIT_INPUT_NAMES,
    get_seedvr_metadata,
)
from PIL import Image

_REGISTERED_EP_NAME: str | None = None


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def add_comfyui_to_path(comfy_root: str | None) -> Path:
    candidate = comfy_root or os.environ.get("COMFYUI_ROOT")
    if not candidate:
        raise RuntimeError("Pass --comfy-root or set COMFYUI_ROOT; this script does not install ComfyUI.")
    root = Path(candidate).expanduser().resolve()
    if not (root / "comfy_extras" / "nodes_seedvr.py").is_file():
        raise FileNotFoundError(f"{root} does not contain comfy_extras/nodes_seedvr.py")
    sys.path.insert(0, str(root))
    return root


def load_image(path: Path) -> torch.Tensor:
    """Load one image in ComfyUI IMAGE layout: frames, height, width, channels."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0)


def resize_image(images: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0:
        raise ValueError(f"--scale must be positive, got {scale}.")
    _, height, width, _ = images.shape
    target_height = max(2, round(height * scale))
    target_width = max(2, round(width * scale))
    return F.interpolate(
        images.movedim(-1, 1),
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
    ).movedim(1, -1)


def to_pil_image(images: torch.Tensor) -> Image.Image:
    if images.ndim == 5:
        images = images.reshape(-1, *images.shape[-3:])
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError(f"Expected one or more HWC output frames, got {tuple(images.shape)}.")
    array = images[0].detach().cpu().clamp(0.0, 1.0).mul(255).round().to(torch.uint8).numpy()
    return Image.fromarray(array, mode="RGB")


def _numpy(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    tensor = tensor.detach()
    if dtype == np.float16:
        tensor = tensor.to(torch.float16)
    elif dtype == np.float32:
        tensor = tensor.to(torch.float32)
    elif dtype == np.int64:
        tensor = tensor.to(torch.int64)
    return np.ascontiguousarray(tensor.to("cpu").numpy().astype(dtype, copy=False))


def _numpy_dtype(ort_type: str) -> np.dtype:
    return {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(int64)": np.int64,
    }[ort_type]


def _profile_string(shapes: dict[str, tuple[int, ...]]) -> str:
    return ",".join(f"{name}:{'x'.join(map(str, shape))}" for name, shape in shapes.items())


def create_trt_rtx_session(
        onnx_path: Path,
        cache_path: Path,
        input_shapes: dict[str, tuple[int, ...]],
):
    global _REGISTERED_EP_NAME
    import onnxruntime as ort
    import onnxruntime_ep_nv_tensorrt_rtx as ep

    if not onnx_path.is_file():
        raise FileNotFoundError(f"SeedVR2 ONNX model not found: {onnx_path}")

    ep_name = ep.get_ep_name()
    if _REGISTERED_EP_NAME is None:
        ort.register_execution_provider_library(ep_name, ep.get_library_path())
        _REGISTERED_EP_NAME = ep_name
    devices = [device for device in ort.get_ep_devices() if device.ep_name == ep_name]
    if not devices:
        raise RuntimeError(f"TensorRT RTX EP {ep_name!r} registered but exposed no GPU device.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    profile = _profile_string(input_shapes)
    provider_options = {
        "nv_profile_min_shapes": profile,
        "nv_profile_opt_shapes": profile,
        "nv_profile_max_shapes": profile,
        "nv_runtime_cache_path": str(cache_path),
    }
    session_options = ort.SessionOptions()
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session_options.add_provider_for_devices(devices, provider_options)
    print(f"Building/loading TensorRT RTX session: {onnx_path}")
    print(f"TensorRT RTX profile: {profile}")
    return ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=None)


def run_vae_encoder(session, images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 5:
        if images.shape[0] * images.shape[1] != 1:
            raise ValueError(f"The initial TensorRT RTX sample supports one image, got {tuple(images.shape)}.")
        images = images.reshape(-1, *images.shape[2:])
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected a BHWC RGB image for VAE encoding, got {tuple(images.shape)}.")
    input_info = session.get_inputs()[0]
    image_nchw = images.movedim(-1, 1)
    output = session.run(None, {input_info.name: _numpy(image_nchw, _numpy_dtype(input_info.type))})[0]
    latent = torch.from_numpy(output).unsqueeze(2)
    print(f"TensorRT RTX VAE encoder: image={tuple(image_nchw.shape)} -> latent={tuple(latent.shape)}")
    return latent


def run_vae_decoder(session, latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 5 or latent.shape[2] != 1:
        raise ValueError(f"Expected a one-frame BCTHW latent for VAE decoding, got {tuple(latent.shape)}.")
    input_info = session.get_inputs()[0]
    latent_nchw = latent.squeeze(2)
    output = session.run(None, {input_info.name: _numpy(latent_nchw, _numpy_dtype(input_info.type))})[0]
    images = torch.from_numpy(output).float().movedim(1, -1)
    print(f"TensorRT RTX VAE decoder: latent={tuple(latent_nchw.shape)} -> image={tuple(images.shape)}")
    return images


class SeedVRTrtRtxDenoiser(torch.nn.Module):
    """ComfyUI NaDiT-compatible wrapper around a TensorRT RTX ORT session."""

    def __init__(self, session, diffusion_model: torch.nn.Module):
        super().__init__()
        self.session = session
        self.diffusion_model = diffusion_model
        self.input_dtypes = {item.name: _numpy_dtype(item.type) for item in session.get_inputs()}
        self.dtype = next(diffusion_model.parameters()).dtype
        self.calls = 0

    def forward(
            self,
            latent: torch.Tensor,
            timestep: torch.Tensor,
            context: torch.Tensor,
            disable_cache: bool = False,
            **kwargs,
    ) -> torch.Tensor:
        del disable_cache
        condition = kwargs.get("condition")
        if condition is None:
            raise ValueError("SeedVR2 TensorRT RTX inference requires the conditioning latent.")
        if latent.shape[0] != 1 or latent.shape[2] != 1:
            raise ValueError(f"The initial TensorRT RTX sample supports B=T=1, got {tuple(latent.shape)}.")
        if context.shape != (1, CONTEXT_TOKENS, CONTEXT_WIDTH):
            raise ValueError(f"Expected context [1, 58, 5120], got {tuple(context.shape)}.")

        height, width = latent.shape[3:5]
        device = next(self.diffusion_model.parameters()).device
        normal, shifted = get_seedvr_metadata(self.diffusion_model, height, width, CONTEXT_TOKENS, device, self.dtype)
        tensors = (latent, timestep, context, condition, *normal.tensors(), *shifted.tensors())
        feed = {
            name: _numpy(tensor, self.input_dtypes[name]) for name, tensor in zip(DIT_INPUT_NAMES, tensors, strict=True)
        }
        self.calls += 1
        print(f"TensorRT RTX DiT call {self.calls}: latent={tuple(latent.shape)}")
        output = self.session.run(["denoised_latent"], feed)[0]
        return torch.from_numpy(output).to(device=latent.device, dtype=latent.dtype)


def _checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    model_path = args.model or args.snapshot / "diffusion_models" / "seedvr2_3b_fp16.safetensors"
    vae_path = args.vae or args.snapshot / "vae" / "seedvr2_ema_vae_fp16.safetensors"
    return model_path.expanduser().absolute(), vae_path.expanduser().absolute()


def run_pytorch_pipeline(args: argparse.Namespace) -> Image.Image:
    """Run the unmodified ComfyUI implementation as the reference."""
    import comfy.model_management
    import comfy.sd
    import comfy.utils
    import nodes
    from comfy_extras.nodes_seedvr import SeedVR2Conditioning, SeedVR2PostProcessing, SeedVR2Preprocess

    model_path, vae_path = _checkpoint_paths(args)
    input_path = args.input.expanduser().absolute()
    for path, label in ((model_path, "diffusion model"), (vae_path, "VAE"), (input_path, "input image")):
        if not path.is_file():
            raise FileNotFoundError(f"SeedVR2 {label} not found: {path}")

    print(f"[PyTorch] Loading diffusion model: {model_path}")
    model = comfy.sd.load_diffusion_model(str(model_path))
    print(f"[PyTorch] Loading VAE: {vae_path}")
    vae_state_dict, vae_metadata = comfy.utils.load_torch_file(str(vae_path), return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_state_dict, metadata=vae_metadata)
    vae.throw_exception_if_invalid()

    resized_images = resize_image(load_image(input_path), args.scale)
    prepared_images = SeedVR2Preprocess.execute(resized_images)[0]
    encoded = vae.encode(prepared_images)
    latent = {"samples": encoded}
    positive, negative = SeedVR2Conditioning.execute(model, latent)
    sampled = nodes.common_ksampler(
        model,
        seed=args.seed,
        steps=1,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=positive,
        negative=negative,
        latent=latent,
        denoise=1.0,
    )[0]
    decoded = vae.decode(sampled["samples"])
    output = SeedVR2PostProcessing.execute(decoded, resized_images, args.color_correction)[0]
    result = to_pil_image(output)
    del output, decoded, sampled, latent, encoded, vae, model
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache(force=True)
    return result


def run_onnx_pipeline(args: argparse.Namespace) -> Image.Image:
    import comfy.model_management
    import comfy.sd
    import nodes
    from comfy_extras.nodes_seedvr import SeedVR2Conditioning, SeedVR2PostProcessing, SeedVR2Preprocess

    model_path, _ = _checkpoint_paths(args)
    onnx_dir = args.onnx_dir.expanduser().absolute()
    onnx_path = onnx_dir / "seedvr2_3b_fp16_dit.onnx"
    vae_encoder_path = onnx_dir / "seedvr2_ema_vae_encoder.onnx"
    vae_decoder_path = onnx_dir / "seedvr2_ema_vae_decoder.onnx"
    input_path = args.input.expanduser().absolute()
    for path, label in (
            (model_path, "diffusion model"),
            (onnx_path, "DiT ONNX model"),
            (vae_encoder_path, "VAE encoder ONNX model"),
            (vae_decoder_path, "VAE decoder ONNX model"),
            (input_path, "input image"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"SeedVR2 {label} not found: {path}")

    print(f"Loading ComfyUI diffusion weights for conditioning metadata: {model_path}")
    model = comfy.sd.load_diffusion_model(str(model_path))
    comfy.model_management.load_models_gpu([model], force_full_load=True)
    diffusion_model = model.model.diffusion_model

    resized_images = resize_image(load_image(input_path), args.scale)
    prepared_images = SeedVR2Preprocess.execute(resized_images)[0]
    prepared_image = prepared_images.reshape(-1, *prepared_images.shape[-3:])
    image_nchw_shape = tuple(prepared_image.movedim(-1, 1).shape)
    encoder_session = create_trt_rtx_session(
        vae_encoder_path,
        args.runtime_cache_dir.expanduser().absolute() / "vae_encoder",
        {"image": image_nchw_shape},
    )
    encoded = run_vae_encoder(encoder_session, prepared_images)
    latent = {"samples": encoded}
    positive, negative = SeedVR2Conditioning.execute(model, latent)
    height, width = encoded.shape[3:5]
    metadata_device = next(diffusion_model.parameters()).device
    metadata_dtype = next(diffusion_model.parameters()).dtype
    normal, shifted = get_seedvr_metadata(
        diffusion_model, height, width, CONTEXT_TOKENS, metadata_device, metadata_dtype
    )
    example_shapes = {
        name: tuple(tensor.shape)
        for name, tensor in zip(
            DIT_INPUT_NAMES,
            (
                encoded,
                torch.ones(1),
                torch.empty(1, CONTEXT_TOKENS, CONTEXT_WIDTH),
                torch.empty(1, 17, 1, height, width),
                *normal.tensors(),
                *shifted.tensors(),
            ),
            strict=True,
        )
    }
    session = create_trt_rtx_session(
        onnx_path,
        args.runtime_cache_dir.expanduser().absolute() / "dit",
        example_shapes,
    )
    trt_denoiser = SeedVRTrtRtxDenoiser(session, diffusion_model)

    model.model.diffusion_model = trt_denoiser
    try:
        sampled = nodes.common_ksampler(
            model,
            seed=args.seed,
            steps=1,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent=latent,
            denoise=1.0,
        )[0]
    finally:
        model.model.diffusion_model = diffusion_model
    if trt_denoiser.calls == 0:
        raise RuntimeError("The sampler completed without invoking the TensorRT RTX DiT session.")

    decoder_shape = tuple(sampled["samples"].squeeze(2).shape)
    decoder_session = create_trt_rtx_session(
        vae_decoder_path,
        args.runtime_cache_dir.expanduser().absolute() / "vae_decoder",
        {"latent": decoder_shape},
    )
    decoded = run_vae_decoder(decoder_session, sampled["samples"])
    output_images = SeedVR2PostProcessing.execute(decoded, resized_images, args.color_correction)[0]
    return to_pil_image(output_images)


def compare_images(reference: Image.Image, ort_image: Image.Image, output_dir: Path) -> None:
    reference_array = np.asarray(reference.convert("RGB"), dtype=np.float32)
    ort_array = np.asarray(ort_image.convert("RGB"), dtype=np.float32)
    if reference_array.shape != ort_array.shape:
        raise RuntimeError(f"PyTorch and ONNX image shapes differ: {reference_array.shape} versus {ort_array.shape}.")

    error = reference_array - ort_array
    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    maximum = float(np.max(np.abs(error)))
    psnr = 10.0 * np.log10(255.0 ** 2 / mse) if mse > 1.0e-9 else float("inf")
    print(f"[Comparison] MSE={mse:.4f}, MAE={mae:.4f}, max={maximum:.1f}, PSNR={psnr:.2f} dB")

    comparison = Image.new("RGB", (reference.width + ort_image.width, reference.height))
    comparison.paste(reference, (0, 0))
    comparison.paste(ort_image, (reference.width, 0))
    comparison.save(output_dir / "comparison.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify SeedVR2 ONNX against the ComfyUI PyTorch reference.")
    parser.add_argument("--comfy-root", default="D:/repos/ComfyUI")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--vae", type=Path, default=None)
    parser.add_argument("--onnx-dir", type=Path, default=Path("out/seedvr/onnx"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/seedvr/verify"))
    parser.add_argument("--runtime-cache-dir", type=Path, default=Path("out/seedvr/trt_rtx_cache"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--color-correction", choices=("none", "lab", "wavelet", "adain"), default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_stdio()
    add_comfyui_to_path(args.comfy_root)
    output_dir = args.output_dir.expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = run_pytorch_pipeline(args)
    reference_path = output_dir / "pytorch_output.png"
    reference.save(reference_path)
    print(f"[PyTorch] Saved {reference_path}")

    ort_image = run_onnx_pipeline(args)
    ort_path = output_dir / "ort_output.png"
    ort_image.save(ort_path)
    print(f"[ONNX Runtime] Saved {ort_path}")
    compare_images(reference, ort_image, output_dir)


if __name__ == "__main__":
    main()
