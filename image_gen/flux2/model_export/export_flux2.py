# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Export FLUX.2-klein-4B (or other Flux2Klein models) from a Hugging Face model
repo to ONNX for use with ONNX Runtime / TensorRT / C++ inference.

Usage:
    python export_flux.py --model_name black-forest-labs/FLUX.2-klein-4b --output ./flux2_klein_onnx
    python export_flux.py --model_name black-forest-labs/FLUX.2-klein-4b --output ./out --model transformer
    python export_flux.py --model_name black-forest-labs/FLUX.2-klein-4b --output ./out --model text_encoder --model vae_decoder

Output layout:
    <output>/
        text_encoder/model.onnx + model.onnx_data
        transformer/model.onnx + model.onnx_data
        vae_encoder/model.onnx + model.onnx_data
        vae_decoder/model.onnx + model.onnx_data
        scheduler/ (config only)
        tokenizer/ (copied as-is)
        model_index.json

All input shapes are static for a single image size (default 1024x1024, set via --image_size).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline as FluxPipeline
from huggingface_hub import snapshot_download

EXTERNAL_DATA_NAME = "model.onnx_data"
DEFAULT_MODEL_NAME = "black-forest-labs/FLUX.2-klein-4b"
DEFAULT_MODEL_OPSETS = {
    "text_encoder": 25,
    "transformer": 25,
    "vae_encoder": 22,
    "vae_decoder": 22,
}

# ONNX model IO is fp32 because the C++ pipeline owns fp32 buffers at graph boundaries.
IO_PRECISION_MAP = {
    "fp32": torch.float32,
}


def _configure_stdio():
    """Keep PyTorch ONNX status output from failing on Windows cp1252 consoles."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    input_names: list,
    output_names: list,
    opset: int,
    **kwargs
):
    """Export model to ONNX and always save weights as external data in model.onnx_data."""
    import onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        model_args,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        dynamo=True,
        opset_version=opset,
        **kwargs
    )
    # Reload and save with external data so we always use model.onnx_data
    onnx_model = onnx.load(str(output_path))
    out_dir = output_path.parent
    for f in out_dir.iterdir():
        if f.name != output_path.name:
            f.unlink()
    onnx.save_model(
        onnx_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=EXTERNAL_DATA_NAME,
        convert_attribute=False,
    )
    print(f"  ONNX saved: {output_path} (opset={opset})")


class TextEncoderPromptEmbedsWrapper(torch.nn.Module):
    """Wraps Qwen3 to output prompt_embeds (stacked hidden states from layers 9, 18, 27) for ONNX.
    Output is cast to io_dtype (inputs are int, unchanged).
    """

    def __init__(self, text_encoder, layers=(9, 18, 27), io_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.text_encoder = text_encoder
        self.layers = layers
        self.io_dtype = io_dtype

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        stacked = torch.stack([out.hidden_states[i] for i in self.layers], dim=1)
        b, nc, seq_len, hidden = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(b, seq_len, nc * hidden)
        return prompt_embeds.to(self.io_dtype)


@torch.no_grad()
def export_text_encoder(
    pipe,
    output_path: Path,
    device: str,
    opset: int,
    seq_len: int = 512,
    io_dtype: torch.dtype = torch.float32,
):
    """Export with static shape (1, seq_len) for 1024x1024 pipeline."""
    wrapper = TextEncoderPromptEmbedsWrapper(pipe.text_encoder, io_dtype=io_dtype).eval()
    dummy_input_ids = torch.randint(0, 1000, (1, seq_len), device=device, dtype=torch.long)
    dummy_attention = (dummy_input_ids != pipe.tokenizer.pad_token_id).to(torch.long)
    _onnx_export(
        wrapper,
        (dummy_input_ids, dummy_attention),
        output_path / "text_encoder" / "model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["prompt_embeds"],
        opset=opset,
    )
    print("  text_encoder exported.")


def _latent_dims_for_image_size(image_size: int, vae_scale_factor: int = 8, patch_size: int = 2):
    """Latent h,w after VAE encode; then patchified spatial size for transformer."""
    latent = image_size // vae_scale_factor  # 1024 -> 128
    patch_h = latent // patch_size           # 128  -> 64
    num_patches = patch_h * patch_h          # 4096
    return latent, patch_h, num_patches


@torch.no_grad()
def export_transformer(
    pipe,
    output_path: Path,
    device: str,
    opset: int,
    image_size: int = 1024,
    seq_len: int = 512,
    io_dtype: torch.dtype = torch.float32,
):
    """Export with static shapes for image_size x image_size (default 1024x1024)."""
    trans = pipe.transformer
    trans_cfg = trans.config
    in_channels = trans_cfg.in_channels
    joint_attention_dim = trans_cfg.joint_attention_dim
    _, patch_h, num_patches = _latent_dims_for_image_size(image_size)

    dummy_timestep = torch.randn(1, device=device, dtype=torch.float)
    t  = torch.arange(1,        device=device)
    h  = torch.arange(patch_h,  device=device)
    w  = torch.arange(patch_h,  device=device)
    ll = torch.arange(1,        device=device)
    coords = torch.cartesian_prod(t, h, w, ll)
    dummy_img_ids = coords.unsqueeze(0).expand(1, -1, -1)
    dummy_txt_ids = torch.zeros(1, seq_len, 4, device=device, dtype=torch.long)

    class TransformerExportWrapper(torch.nn.Module):
        """Inputs cast to bfloat16, output cast to io_dtype."""

        def __init__(self, transformer, io_dtype_: torch.dtype):
            super().__init__()
            self.transformer = transformer
            self.io_dtype = io_dtype_

        def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
            hidden_states = hidden_states.to(torch.bfloat16)
            encoder_hidden_states = encoder_hidden_states.to(torch.bfloat16)
            out = self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )
            return out[0].to(self.io_dtype)

    wrapper = TransformerExportWrapper(trans, io_dtype).eval()
    dummy_hidden  = torch.randn(1, num_patches, in_channels,        device=device, dtype=io_dtype)
    dummy_encoder = torch.randn(1, seq_len,     joint_attention_dim, device=device, dtype=io_dtype)
    _onnx_export(
        wrapper,
        (dummy_hidden, dummy_encoder, dummy_timestep, dummy_img_ids, dummy_txt_ids),
        output_path / "transformer" / "model.onnx",
        input_names=["hidden_states", "encoder_hidden_states", "timestep", "img_ids", "txt_ids"],
        output_names=["sample"],
        opset=opset,
    )
    print("  transformer exported.")


@torch.no_grad()
def export_vae_encoder(
    pipe,
    output_path: Path,
    device: str,
    opset: int,
    image_size: int = 1024,
    io_dtype: torch.dtype = torch.float32,
):
    """Export with static input shape (1, 3, image_size, image_size) for 1024x1024."""
    vae = pipe.vae
    vae_in_channels = vae.config.in_channels

    class VaeEncoderExportWrapper(torch.nn.Module):
        """Input cast to bfloat16, output cast to io_dtype."""

        def __init__(self, vae_module, io_dtype_: torch.dtype):
            super().__init__()
            self.vae = vae_module
            self.io_dtype = io_dtype_

        def forward(self, sample, return_dict=False):
            sample = sample.to(torch.bfloat16)
            out = self.vae.encode(sample, return_dict=return_dict)[0].mode()
            return out.to(self.io_dtype)

    wrapper = VaeEncoderExportWrapper(vae, io_dtype).eval()
    dummy_image = torch.randn(1, vae_in_channels, image_size, image_size, device=device, dtype=io_dtype)
    _onnx_export(
        wrapper,
        (dummy_image, False),
        output_path / "vae_encoder" / "model.onnx",
        input_names=["sample", "return_dict"],
        output_names=["latent_sample"],
        opset=opset,
    )
    print("  vae_encoder exported.")


@torch.no_grad()
def export_vae_decoder(
    pipe,
    output_path: Path,
    device: str,
    opset: int,
    image_size: int = 1024,
    io_dtype: torch.dtype = torch.float32,
):
    """Export with static latent input (1, 32, latent_h, latent_w) for image_size x image_size output."""
    vae = pipe.vae
    vae_latent_channels = vae.config.latent_channels
    patch_size = vae.config.patch_size
    latent_h = image_size // 16 * patch_size[0]
    latent_w = image_size // 16 * patch_size[1]

    class VaeDecoderExportWrapper(torch.nn.Module):
        """Input cast to bfloat16, output cast to io_dtype."""

        def __init__(self, vae_module, io_dtype_: torch.dtype):
            super().__init__()
            self.vae = vae_module
            self.io_dtype = io_dtype_

        def forward(self, latent_sample, return_dict=False):
            latent_sample = latent_sample.to(torch.bfloat16)
            out = self.vae.decode(latent_sample, return_dict=return_dict)[0]
            return out.to(self.io_dtype)

    wrapper = VaeDecoderExportWrapper(vae, io_dtype).eval()
    dummy_latent = torch.randn(1, vae_latent_channels, latent_h, latent_w, device=device, dtype=io_dtype)
    _onnx_export(
        wrapper,
        (dummy_latent, False),
        output_path / "vae_decoder" / "model.onnx",
        input_names=["latent_sample", "return_dict"],
        output_names=["sample"],
        opset=opset,
    )
    print("  vae_decoder exported.")


def resolve_model_snapshot(model_name: str, local_files_only: bool) -> Path:
    local_path = Path(model_name).expanduser()
    if local_path.exists():
        return local_path.resolve()
    return Path(snapshot_download(repo_id=model_name, local_files_only=local_files_only)).resolve()


def copy_scheduler_and_tokenizer(model_snapshot: Path, output_path: Path):
    for name in ("scheduler", "tokenizer"):
        src = model_snapshot / name
        dst = output_path / name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied {name}/")


def write_model_index(output_path: Path):
    model_index = {
        "_class_name": "Flux2KleinPipeline",
        "_diffusers_version": "0.38.0",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["optimum", "ORTModelForCausalLM"],
        "tokenizer": ["transformers", "Qwen2TokenizerFast"],
        "transformer": ["optimum", "ORTModel"],
        "vae": ["diffusers", "AutoencoderKLFlux2"],
    }
    with open(output_path / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2)
    print("  model_index.json written.")


def _find_trt_binary(trt_root: str | None, trt_bin: str | None) -> Path:
    if trt_bin:
        candidate = Path(trt_bin).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"TensorRT RTX binary not found: {candidate}")
        return candidate

    exe_name = "tensorrt_rtx.exe" if os.name == "nt" else "tensorrt_rtx"
    if trt_root:
        candidate = Path(trt_root).expanduser().resolve() / "bin" / exe_name
        if not candidate.is_file():
            raise FileNotFoundError(f"TensorRT RTX binary not found: {candidate}")
        return candidate

    found = shutil.which(exe_name) or shutil.which("tensorrt_rtx")
    if found:
        return Path(found).resolve()

    raise FileNotFoundError("TensorRT RTX binary not found; pass --trt_root or --trt_bin")


def _trt_subprocess_env(trt_root: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if not trt_root:
        return env

    root = Path(trt_root).expanduser().resolve()
    if os.name == "nt":
        env["PATH"] = str(root / "bin") + os.pathsep + env.get("PATH", "")
    else:
        env["LD_LIBRARY_PATH"] = str(root / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def compile_with_trt_rtx(
    onnx_path: Path,
    trt_exe: Path,
    trt_root: str | None,
    save_engine: bool = False,
    extra_args: list[str] | None = None,
):
    cmd = [str(trt_exe), f"--onnx={onnx_path}", "--skipInference"]
    if save_engine:
        cmd.append(f"--saveEngine={onnx_path.with_suffix('.engine')}")
    if extra_args:
        cmd.extend(extra_args)

    print(f"  TRT RTX compile: {onnx_path}")
    subprocess.run(cmd, check=True, env=_trt_subprocess_env(trt_root))
    print("  TRT RTX compile passed.")


EXPORTERS = {
    "text_encoder": export_text_encoder,
    "transformer":  export_transformer,
    "vae_encoder":  export_vae_encoder,
    "vae_decoder":  export_vae_decoder,
}
MODELS = list(EXPORTERS.keys())


def main():
    _configure_stdio()
    os.environ.setdefault("PYTHONUTF8", "1")

    ap = argparse.ArgumentParser(
        description="Export a FLUX.2-klein Hugging Face model to ONNX (each model in model.onnx + model.onnx_data)",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model repo ID, or a local snapshot directory",
    )
    ap.add_argument(
        "--local_files_only",
        action="store_true",
        help="Resolve --model_name from the local Hugging Face cache only.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="./flux2_klein_onnx",
        help="Output directory for ONNX models and configs",
    )
    ap.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        metavar="NAME",
        help="Model(s) to export: one or more of all, text_encoder, transformer, vae_encoder, vae_decoder (default: all)",
    )
    ap.add_argument(
        "--opset",
        type=int,
        default=None,
        help=(
            "Override ONNX opset for every selected model. By default the script "
            f"uses per-model opsets: {DEFAULT_MODEL_OPSETS}."
        ),
    )
    ap.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Image height/width for static input shapes (default 1024)",
    )
    ap.add_argument(
        "--io_precision",
        type=str,
        choices=list(IO_PRECISION_MAP.keys()),
        default="fp32",
        help="IO precision of ONNX models; only fp32 is supported by the C++ pipeline",
    )
    ap.add_argument(
        "--compile_trt",
        action="store_true",
        help="After export, build each selected ONNX model with the TensorRT RTX CLI and --skipInference.",
    )
    ap.add_argument(
        "--trt_root",
        type=str,
        default=None,
        help="TensorRT RTX install root containing bin/ and lib/ (for example TensorRT-RTX-1.5.0.114).",
    )
    ap.add_argument(
        "--trt_bin",
        type=str,
        default=None,
        help="Explicit path to tensorrt_rtx executable. Overrides --trt_root.",
    )
    ap.add_argument(
        "--trt_save_engines",
        action="store_true",
        help="Save compiled engines next to each ONNX file as model.engine.",
    )
    ap.add_argument(
        "--trt_arg",
        action="append",
        default=[],
        help="Additional argument to pass to tensorrt_rtx. May be specified more than once.",
    )
    args = ap.parse_args()

    model_snapshot = resolve_model_snapshot(args.model_name, args.local_files_only)
    output_path     = Path(args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    image_size = args.image_size
    seq_len    = 512  # static text sequence length for FLUX.2-klein

    # Parse --model: default all; otherwise one or more of all, text_encoder, transformer, …
    model_choices = ["all"] + list(MODELS)
    raw = args.model if args.model is not None else ["all"]
    to_export: list[str] = []
    for m in raw:
        if m not in model_choices:
            ap.error(f"--model must be one or more of: {', '.join(model_choices)}")
        if m == "all":
            to_export = list(MODELS)
            break
        if m not in to_export:
            to_export.append(m)
    if not to_export:
        to_export = list(MODELS)
    export_all = (args.model is None) or ("all" in raw)

    dtype    = torch.bfloat16
    device   = "cuda"
    io_dtype = IO_PRECISION_MAP[args.io_precision]

    print(
        f"Loading pipeline from {args.model_name} ({model_snapshot}) "
        f"(dtype={dtype}, image_size={image_size}, io_precision={args.io_precision})..."
    )
    pipe = FluxPipeline.from_pretrained(str(model_snapshot), torch_dtype=dtype).to(device)

    exported_model_paths: list[Path] = []
    for name in to_export:
        model_opset = args.opset if args.opset is not None else DEFAULT_MODEL_OPSETS[name]
        print(f"Exporting {name}...")
        fn = EXPORTERS[name]
        if name == "text_encoder":
            fn(pipe, output_path, device, model_opset,
               seq_len=seq_len, io_dtype=io_dtype)
        elif name == "transformer":
            fn(pipe, output_path, device, model_opset,
               image_size=image_size, seq_len=seq_len, io_dtype=io_dtype)
        elif name in ("vae_encoder", "vae_decoder"):
            fn(pipe, output_path, device, model_opset,
               image_size=image_size, io_dtype=io_dtype)
        else:
            fn(pipe, output_path, device, model_opset, io_dtype=io_dtype)
        exported_model_paths.append(output_path / name / "model.onnx")

    if export_all:
        copy_scheduler_and_tokenizer(model_snapshot, output_path)
        write_model_index(output_path)

    if args.compile_trt:
        trt_exe = _find_trt_binary(args.trt_root, args.trt_bin)
        print(f"Compiling exported ONNX models with TensorRT RTX: {trt_exe}")
        for onnx_path in exported_model_paths:
            compile_with_trt_rtx(
                onnx_path,
                trt_exe=trt_exe,
                trt_root=args.trt_root,
                save_engine=args.trt_save_engines,
                extra_args=args.trt_arg,
            )

    print(f"\nONNX export done -> {output_path}")


if __name__ == "__main__":
    main()
