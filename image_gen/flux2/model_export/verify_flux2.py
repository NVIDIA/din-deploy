# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Run the FLUX.2-klein pipeline once with PyTorch and once with ONNX Runtime,
then compare the generated images pixel-by-pixel.

The ONNX path swaps the exported ONNX sessions into the diffusers pipeline in
place of the corresponding PyTorch modules. The sessions can run on CPU or the
TensorRT-RTX ONNX Runtime EP.

Outputs (written inside --output_dir if given, otherwise --onnx_dir):
    pytorch_output.png  - image from the pure-PyTorch pipeline
    ort_output.png      - image from the ONNX Runtime-backed pipeline
    comparison.png      - PyTorch (left) | ONNX Runtime (right) side-by-side
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from diffusers import Flux2KleinPipeline as FluxPipeline
from huggingface_hub import snapshot_download
from PIL import Image as PILImage

EP_NAME = "nv_tensorrt_rtx"
_REGISTERED_EP_NAME: str | None = None
DEFAULT_MODEL_NAME = "black-forest-labs/FLUX.2-klein-4b"

IO_PRECISION_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

DEFAULT_PROMPT = (
    "a photo of a forest with mist swirling around the tree trunks. "
    "The word 'FLUX.2' is painted over it in big, red brush strokes with visible texture"
)


def configure_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def resolve_model_snapshot(model_name: str, local_files_only: bool) -> Path:
    local_path = Path(model_name).expanduser()
    if local_path.exists():
        return local_path.resolve()
    return Path(snapshot_download(repo_id=model_name, local_files_only=local_files_only)).resolve()


def _np_dtype(type_name: str) -> np.dtype:
    return {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(bool)": np.bool_,
    }[type_name]


def _register_trt_rtx(args) -> str:
    global _REGISTERED_EP_NAME
    if _REGISTERED_EP_NAME is not None:
        return _REGISTERED_EP_NAME

    try:
        import onnxruntime_ep_nv_tensorrt_rtx as ep  # noqa: F401

        name, lib = ep.get_ep_name(), ep.get_library_path()
        print(f"  [ORT] Using onnxruntime-ep-nv-tensorrt-rtx ({name})")
    except ImportError:
        if not args.ep_lib:
            raise SystemExit("trt-rtx needs onnxruntime-ep-nv-tensorrt-rtx or --ep_lib")
        name, lib = EP_NAME, str(args.ep_lib)
        for d in (args.ep_dll_dir, args.trt_bin):
            if d and os.path.isdir(d):
                os.add_dll_directory(os.path.abspath(d))
    import onnxruntime as ort

    ort.register_execution_provider_library(name, lib)
    _REGISTERED_EP_NAME = name
    return name


def _make_session_options(provider: str, args):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if provider != "trt-rtx":
        return so, ["CPUExecutionProvider"]
    name = _register_trt_rtx(args)
    devices = [d for d in ort.get_ep_devices() if d.ep_name == name]
    if not devices:
        raise SystemExit(f"EP '{name}' registered but no devices were discovered")
    so.add_provider_for_devices(devices, {})
    return so, None


def _make_session(onnx_path: Path, provider: str, args):
    import onnxruntime as ort

    so, providers = _make_session_options(provider, args)
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)


class OrtSessionRunner:
    def __init__(self, session):
        self.session = session
        self.input_names = [i.name for i in session.get_inputs()]
        self.output_names = [o.name for o in session.get_outputs()]
        self.input_dtypes = {i.name: _np_dtype(i.type) for i in session.get_inputs()}
        self.output_dtypes = {o.name: _np_dtype(o.type) for o in session.get_outputs()}

    @classmethod
    def from_path(cls, onnx_path: Path, provider: str, args):
        return cls(_make_session(onnx_path, provider, args))

    def run(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self.session.run(None, feed)


def _assert_fp32_float_io(runner: OrtSessionRunner, onnx_path: Path) -> None:
    float_dtypes = {np.float16, np.float32}
    bad = [
        f"input {name}: {dtype}"
        for name, dtype in runner.input_dtypes.items()
        if dtype in float_dtypes and dtype != np.float32
    ]
    bad.extend(
        f"output {name}: {dtype}"
        for name, dtype in runner.output_dtypes.items()
        if dtype in float_dtypes and dtype != np.float32
    )
    if bad:
        raise RuntimeError(f"{onnx_path} has non-fp32 floating graph IO: {', '.join(bad)}")


class _DeviceAnchor(nn.Module):
    def __init__(self, device: str | torch.device, dtype: torch.dtype):
        super().__init__()
        self.register_buffer("_anchor", torch.empty(0, device=device, dtype=dtype), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._anchor.device

    @property
    def dtype(self) -> torch.dtype:
        return self._anchor.dtype


def _to_numpy(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    if not isinstance(tensor, torch.Tensor):
        return np.asarray(tensor, dtype=dtype)
    tensor = tensor.detach()
    if dtype == np.float32:
        tensor = tensor.to(torch.float32)
    elif dtype == np.float16:
        tensor = tensor.to(torch.float16)
    return np.ascontiguousarray(tensor.cpu().numpy().astype(dtype, copy=False))


class _OrtTransformerModule(nn.Module):
    def __init__(self, runner: OrtSessionRunner, original_config, original_dtype: torch.dtype):
        super().__init__()
        self._runner = runner
        self.config = original_config
        self.dtype = original_dtype
        self.register_buffer("_anchor", torch.empty(0, device="cuda", dtype=original_dtype), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._anchor.device

    def _input(self, name: str, tensor: torch.Tensor) -> np.ndarray:
        return _to_numpy(tensor, self._runner.input_dtypes[name])

    @contextlib.contextmanager
    def cache_context(self, _key: str):
        yield

    def __call__(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        guidance=None,
        joint_attention_kwargs=None,
        return_dict: bool = False,
        **_,
    ):
        feed = {
            "hidden_states": self._input("hidden_states", hidden_states),
            "encoder_hidden_states": self._input("encoder_hidden_states", encoder_hidden_states),
            "timestep": self._input("timestep", timestep),
            "img_ids": self._input("img_ids", img_ids),
            "txt_ids": self._input("txt_ids", txt_ids),
        }
        out = self._runner.run(feed)
        return (torch.from_numpy(out[0]).to(device=hidden_states.device, dtype=hidden_states.dtype),)


class _OrtVaeModule(nn.Module):
    def __init__(self, decoder_runner: OrtSessionRunner, original_vae):
        super().__init__()
        self._runner = decoder_runner
        self.config = original_vae.config
        self.bn = original_vae.bn
        self.register_buffer(
            "_anchor",
            torch.empty(0, device="cuda", dtype=getattr(original_vae, "dtype", torch.bfloat16)),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        return self._anchor.device

    @property
    def dtype(self) -> torch.dtype:
        return self._anchor.dtype

    def decode(self, latent_sample, return_dict=False, **_kw):
        feed = {"latent_sample": _to_numpy(latent_sample, self._runner.input_dtypes["latent_sample"])}
        out = self._runner.run(feed)
        return (torch.from_numpy(out[0]).to(device=latent_sample.device),)


def _onnx_input_shapes(onnx_path: Path) -> dict[str, tuple[int, ...]]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    fallback = {
        "input_ids": (1, 512),
        "attention_mask": (1, 512),
        "hidden_states": (1, 4096, 128),
        "encoder_hidden_states": (1, 512, 7680),
        "timestep": (1,),
        "img_ids": (1, 4096, 4),
        "txt_ids": (1, 512, 4),
        "pixel_values": (1, 3, 1024, 1024),
        "latent_sample": (1, 32, 64, 64),
    }
    shapes: dict[str, tuple[int, ...]] = {}
    for graph_input in model.graph.input:
        if graph_input.name in initializer_names:
            continue
        dims = [int(dim.dim_value) if dim.dim_value > 0 else 0 for dim in graph_input.type.tensor_type.shape.dim]
        shape = tuple(dims)
        if not shape or any(value <= 0 for value in shape):
            shape = fallback.get(graph_input.name, tuple(1 if value <= 0 else value for value in shape))
        shapes[graph_input.name] = shape
    return shapes


def _dummy_numpy(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if dtype in (np.int32, np.int64):
        return np.zeros(shape, dtype=dtype)
    if dtype == np.bool_:
        return np.zeros(shape, dtype=dtype)
    return np.random.default_rng(0).standard_normal(shape).astype(dtype)


def validate_onnx_dir(onnx_dir: Path, provider: str, args) -> None:
    print(f"\n[Validate] {onnx_dir}")
    model_paths = [
        onnx_dir / "text_encoder" / "model.onnx",
        onnx_dir / "transformer" / "model.onnx",
        onnx_dir / "vae_encoder" / "model.onnx",
        onnx_dir / "vae_decoder" / "model.onnx",
    ]
    found = False
    for onnx_path in model_paths:
        if not onnx_path.exists():
            continue
        found = True
        print(f"  [Validate] {onnx_path.parent.name}")
        runner = OrtSessionRunner.from_path(onnx_path, provider, args)
        _assert_fp32_float_io(runner, onnx_path)
        shapes = _onnx_input_shapes(onnx_path)
        feed = {}
        for inp in runner.input_names:
            shape = shapes.get(inp)
            if shape is None:
                raise RuntimeError(f"No ONNX shape found for input {inp} in {onnx_path}")
            dtype = runner.input_dtypes[inp]
            feed[inp] = _dummy_numpy(shape, dtype)
            print(f"    input {inp}: shape={shape}, dtype={dtype}")
        outputs = runner.run(feed)
        for out_name, output in zip(runner.output_names, outputs, strict=True):
            print(f"    output {out_name}: shape={tuple(output.shape)}, dtype={output.dtype}")
    if not found:
        raise RuntimeError(f"No model.onnx files found under {onnx_dir}")


def run_pytorch_pipeline(pipe, prompt: str, seed: int, image_size: int, num_steps: int) -> PILImage.Image:
    print("\n[PyTorch] Running pipeline...")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        height=image_size,
        width=image_size,
        num_inference_steps=num_steps,
        generator=generator,
    )
    return result[0][0]


def run_ort_pipeline(
    pipe,
    onnx_dir: Path,
    prompt: str,
    seed: int,
    image_size: int,
    num_steps: int,
    provider: str,
    args,
    io_dtype: torch.dtype,
    seq_len: int = 512,
) -> PILImage.Image:
    print("\n[ONNX Runtime] Building sessions and running pipeline...")

    prompt_embeds = None
    original_text_encoder = None
    te_path = onnx_dir / "text_encoder" / "model.onnx"
    if te_path.exists():
        original_text_encoder = pipe.text_encoder
        original_text_dtype = original_text_encoder.dtype
        original_text_encoder.to("cpu")
        torch.cuda.empty_cache()
        te_runner = OrtSessionRunner.from_path(te_path, provider, args)
        messages = [{"role": "user", "content": prompt}]
        chat_text = pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tok = pipe.tokenizer(
            chat_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        te_feed = {
            "input_ids": _to_numpy(tok["input_ids"], te_runner.input_dtypes["input_ids"]),
            "attention_mask": _to_numpy(tok["attention_mask"], te_runner.input_dtypes["attention_mask"]),
        }
        te_out = te_runner.run(te_feed)
        prompt_embeds = torch.from_numpy(te_out[0]).to(device="cuda", dtype=original_text_dtype)
        pipe.text_encoder = _DeviceAnchor("cuda", prompt_embeds.dtype)
        print(f"  [ORT] prompt_embeds: {tuple(prompt_embeds.shape)}, dtype={prompt_embeds.dtype}")
    else:
        print(f"  [ORT] {te_path} not found - using PyTorch text encoder.")

    original_transformer = None
    tr_path = onnx_dir / "transformer" / "model.onnx"
    if tr_path.exists():
        original_transformer = pipe.transformer
        original_config = original_transformer.config
        original_dtype = original_transformer.dtype
        original_transformer.to("cpu")
        pipe.transformer = None
        torch.cuda.empty_cache()
        tr_runner = OrtSessionRunner.from_path(tr_path, provider, args)
        pipe.transformer = _OrtTransformerModule(tr_runner, original_config, original_dtype)
    else:
        print(f"  [ORT] {tr_path} not found - using PyTorch transformer.")

    original_vae = None
    vd_path = onnx_dir / "vae_decoder" / "model.onnx"
    if vd_path.exists():
        original_vae = pipe.vae
        original_vae.to("cpu")
        torch.cuda.empty_cache()
        vd_runner = OrtSessionRunner.from_path(vd_path, provider, args)
        pipe.vae = _OrtVaeModule(vd_runner, original_vae)
    else:
        print(f"  [ORT] {vd_path} not found - using PyTorch VAE decoder.")

    try:
        execution_device = getattr(pipe, "_execution_device", torch.device("cuda"))
        if isinstance(execution_device, str):
            execution_device = torch.device(execution_device)
        generator = torch.Generator(device=execution_device.type).manual_seed(seed)
        result = pipe(
            prompt=None if prompt_embeds is not None else prompt,
            prompt_embeds=prompt_embeds,
            height=image_size,
            width=image_size,
            num_inference_steps=num_steps,
            generator=generator,
        )
        return result[0][0]
    finally:
        if original_transformer is not None:
            pipe.transformer = original_transformer
        if original_text_encoder is not None:
            pipe.text_encoder = original_text_encoder
        if original_vae is not None:
            pipe.vae = original_vae


def compare_images(ref: PILImage.Image, ort: PILImage.Image, output_dir: Path) -> None:
    ref_arr = np.array(ref.convert("RGB"), dtype=np.float32)
    ort_arr = np.array(ort.convert("RGB"), dtype=np.float32)

    if ref_arr.shape == ort_arr.shape:
        mse = float(np.mean((ref_arr - ort_arr) ** 2))
        psnr = 10.0 * np.log10(255.0**2 / mse) if mse > 1e-9 else float("inf")
        mae = float(np.mean(np.abs(ref_arr - ort_arr)))
        print(f"\n[Comparison] PyTorch vs ONNX Runtime - MSE: {mse:.2f}  PSNR: {psnr:.2f} dB  MAE: {mae:.2f}")
    else:
        print(f"\n[Comparison] Image shape mismatch - ref {ref_arr.shape} vs ORT {ort_arr.shape}")

    canvas = PILImage.new("RGB", (ref.width + ort.width, max(ref.height, ort.height)), (255, 255, 255))
    canvas.paste(ref, (0, 0))
    canvas.paste(ort, (ref.width, 0))
    cmp_path = output_dir / "comparison.png"
    canvas.save(str(cmp_path))
    print(f"[Comparison] Side-by-side saved -> {cmp_path}")


def main():
    p = argparse.ArgumentParser(description="Verify FLUX.2-klein ONNX graphs with PyTorch and ONNX Runtime.")
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model repo ID, or a local snapshot directory")
    p.add_argument("--local_files_only", action="store_true", help="Resolve --model_name from the local Hugging Face cache only")
    p.add_argument("--onnx_dir", type=str, default="./flux2_klein_onnx")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_steps", type=int, default=4)
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--io_precision", type=str, choices=list(IO_PRECISION_MAP.keys()), default="fp32")
    p.add_argument("--provider", choices=["cpu", "trt-rtx"], default="trt-rtx")
    p.add_argument("--ep_lib", default=None, help="Path to onnxruntime_providers_nv_tensorrt_rtx.dll/.so")
    p.add_argument("--ep_dll_dir", default=None, help="Dir with the plugin EP's bundled runtime DLLs")
    p.add_argument("--trt_bin", default=None, help="TensorRT-RTX bin dir, used as a DLL search path")
    p.add_argument("--validate_only", action="store_true", help="Only run ONNX Runtime validation; do not generate images")
    args = p.parse_args()
    configure_stdio()

    onnx_dir = Path(args.onnx_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else onnx_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        validate_onnx_dir(onnx_dir, args.provider, args)
        return

    io_dtype = IO_PRECISION_MAP[args.io_precision]
    dtype = torch.bfloat16
    device = "cuda"

    model_snapshot = resolve_model_snapshot(args.model_name, args.local_files_only)
    print(f"Loading pipeline from {args.model_name} ({model_snapshot})...")
    pipe = FluxPipeline.from_pretrained(str(model_snapshot), torch_dtype=dtype).to(device)

    pytorch_image = run_pytorch_pipeline(
        pipe,
        prompt=args.prompt,
        seed=args.seed,
        image_size=args.image_size,
        num_steps=args.num_steps,
    )
    pt_path = output_dir / "pytorch_output.png"
    pytorch_image.save(str(pt_path))
    print(f"[PyTorch] Image saved -> {pt_path}")

    ort_image = run_ort_pipeline(
        pipe,
        onnx_dir=onnx_dir,
        prompt=args.prompt,
        seed=args.seed,
        image_size=args.image_size,
        num_steps=args.num_steps,
        provider=args.provider,
        args=args,
        io_dtype=io_dtype,
    )
    ort_path = output_dir / "ort_output.png"
    ort_image.save(str(ort_path))
    print(f"[ONNX Runtime] Image saved -> {ort_path}")

    compare_images(pytorch_image, ort_image, output_dir)


if __name__ == "__main__":
    main()
