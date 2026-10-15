# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Export PixelDiT text-to-image (nvidia/PixelDiT-1300M-1024px) and its Gemma-2 text encoder to ONNX
for use with ONNX Runtime / TensorRT / C++ inference.

Usage:
    python export_pixeldit.py --output ./models/PixelDiT-1300M-1024px-onnx
    python export_pixeldit.py --output ./out --model transformer --height 768 --width 1344

Output layout:
    <output>/
        text_encoder/model.onnx + model.onnx_data
        transformer/model.onnx + model.onnx_data
        tokenizer/ (Gemma tokenizer files)
        pipeline_config.json

All input shapes are static. The DiT runs negative + positive prompts as one batch of 2 (CFG);
CFG and the flow DPM-Solver++ update stay outside the graphs (see pixeldit/pipeline.py).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from pixeldit import pipeline as pixeldit_pipeline
from pixeldit.weights import DEFAULT_MODEL_NAME, DEFAULT_TEXT_ENCODER_NAME, load_gemma_text_encoder, load_pixeldit

EXTERNAL_DATA_NAME = "model.onnx_data"
DEFAULT_MODEL_OPSETS = {
    "text_encoder": 25,
    "transformer": 25,
}
DIT_BATCH = 2  # [negative, positive] for classifier-free guidance

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


class TextEncoderExportWrapper(torch.nn.Module):
    """Gemma-2 decoder final hidden state, gathered to the 300 rows the DiT consumes.

    select_index picks BOS + the last 299 positions for the positive (chi_prompt-prefixed)
    prompt, or the first 300 positions for the negative prompt. Output is cast to io_dtype.
    """

    def __init__(self, text_encoder, io_dtype: torch.dtype):
        super().__init__()
        self.text_encoder = text_encoder
        self.io_dtype = io_dtype

    def forward(self, input_ids, attention_mask, select_index):
        hidden = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)[0]
        return hidden.index_select(1, select_index).to(self.io_dtype)


class PixelDiTExportWrapper(torch.nn.Module):
    """Inputs are cast to bf16 inside PixelDiT (as upstream PixDiTTrainer does), output to io_dtype."""

    def __init__(self, model, io_dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.io_dtype = io_dtype

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        return self.model(hidden_states, timestep, encoder_hidden_states).to(self.io_dtype)


@torch.no_grad()
def export_text_encoder(text_encoder, tokenizer, layout, output_path: Path, device: str, opset: int, io_dtype: torch.dtype):
    """Export with static shape (1, text_seq_len) -> (1, txt_max_length, hidden)."""
    wrapper = TextEncoderExportWrapper(text_encoder, io_dtype).eval()
    dummy_ids, dummy_mask = pixeldit_pipeline.tokenize(tokenizer, layout.chi_prompt, layout.text_seq_len, device)
    dummy_select = torch.tensor(layout.positive_select_index(), dtype=torch.long, device=device)
    _onnx_export(
        wrapper,
        (dummy_ids, dummy_mask, dummy_select),
        output_path / "text_encoder" / "model.onnx",
        input_names=["input_ids", "attention_mask", "select_index"],
        output_names=["prompt_embeds"],
        opset=opset,
    )
    print("  text_encoder exported.")


@torch.no_grad()
def export_transformer(model, output_path: Path, device: str, opset: int, height: int, width: int, io_dtype: torch.dtype):
    """Export with static shapes (2, 3, height, width), (2,), (2, txt_max_length, txt_embed_dim)."""
    wrapper = PixelDiTExportWrapper(model, io_dtype).eval()
    txt_embed_dim = model.y_embedder.proj.in_features
    dummy_x = torch.randn(DIT_BATCH, 3, height, width, device=device, dtype=io_dtype)
    dummy_t = torch.full((DIT_BATCH,), 500.0, device=device, dtype=io_dtype)
    dummy_y = torch.randn(DIT_BATCH, model.txt_max_length, txt_embed_dim, device=device, dtype=io_dtype)
    _onnx_export(
        wrapper,
        (dummy_x, dummy_t, dummy_y),
        output_path / "transformer" / "model.onnx",
        input_names=["hidden_states", "timestep", "encoder_hidden_states"],
        output_names=["velocity"],
        opset=opset,
    )
    print("  transformer exported.")


def write_pipeline_config(output_path: Path, args, tokenizer, layout, sampler) -> None:
    config = {
        "model_name": args.model_name,
        "text_encoder_name": args.text_encoder_name,
        "height": args.height,
        "width": args.width,
        "io_precision": args.io_precision,
        "compute_dtype": "bf16",
        "dit_batch": DIT_BATCH,
        "dit_batch_order": ["negative", "positive"],
        "timestep_scale": pixeldit_pipeline.NUM_TRAIN_TIMESTEPS,
        "text": {
            "chi_prompt": layout.chi_prompt,
            "chi_prompt_tokens": layout.chi_prompt_tokens,
            "text_seq_len": layout.text_seq_len,
            "txt_max_length": layout.txt_max_length,
            "padding_side": "right",
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "positive_select_index": layout.positive_select_index(),
            "negative_select_index": layout.negative_select_index(),
        },
        "sampler": {
            "algorithm": "flow_dpm_solver++_multistep",
            "order": 2,
            "lower_order_final": True,
            "t_start": 1.0,
            "t_end": 1.0 / pixeldit_pipeline.NUM_TRAIN_TIMESTEPS,
            "steps": sampler.steps,
            "cfg_scale": sampler.cfg_scale,
            "flow_shift": sampler.flow_shift,
            "negative_prompt": sampler.negative_prompt,
        },
    }
    with open(output_path / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("  pipeline_config.json written.")


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


MODELS = ["text_encoder", "transformer"]


def main():
    _configure_stdio()

    ap = argparse.ArgumentParser(
        description="Export PixelDiT T2I and its Gemma-2 text encoder to ONNX (each model in model.onnx + model.onnx_data)",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model repo ID, a local snapshot directory, or a local pixeldit_t2i_v1.pth",
    )
    ap.add_argument(
        "--text_encoder_name",
        type=str,
        default=DEFAULT_TEXT_ENCODER_NAME,
        help="Hugging Face repo ID or local directory of the Gemma-2 text encoder",
    )
    ap.add_argument(
        "--local_files_only",
        action="store_true",
        help="Resolve models from the local Hugging Face cache only.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="./pixeldit_onnx",
        help="Output directory for ONNX models and configs",
    )
    ap.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        metavar="NAME",
        help="Model(s) to export: one or more of all, text_encoder, transformer (default: all)",
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
    ap.add_argument("--height", type=int, default=1024, help="Image height for static shapes (multiple of 16, default 1024)")
    ap.add_argument("--width", type=int, default=1024, help="Image width for static shapes (multiple of 16, default 1024)")
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

    if args.height % 16 or args.width % 16:
        ap.error("--height and --width must be multiples of 16 (the PixelDiT patch size)")

    output_path = Path(args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    model_choices = ["all"] + MODELS
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

    device = "cuda"
    io_dtype = IO_PRECISION_MAP[args.io_precision]

    print(f"Loading tokenizer from {args.text_encoder_name}...")
    tokenizer, text_encoder = None, None
    if "text_encoder" in to_export:
        tokenizer, text_encoder = load_gemma_text_encoder(args.text_encoder_name, device, args.local_files_only)
        print(f"  Gemma attention implementation: {text_encoder.config._attn_implementation}")
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, local_files_only=args.local_files_only)
    layout = pixeldit_pipeline.TextLayout.from_tokenizer(tokenizer)
    print(f"  chi_prompt tokens={layout.chi_prompt_tokens}, text_seq_len={layout.text_seq_len}")

    exported_model_paths: list[Path] = []
    for name in to_export:
        model_opset = args.opset if args.opset is not None else DEFAULT_MODEL_OPSETS[name]
        print(f"Exporting {name}...")
        if name == "text_encoder":
            export_text_encoder(text_encoder, tokenizer, layout, output_path, device, model_opset, io_dtype)
            del text_encoder
            torch.cuda.empty_cache()
        elif name == "transformer":
            print(f"Loading PixelDiT from {args.model_name} (dtype=bf16, {args.height}x{args.width})...")
            model = load_pixeldit(args.model_name, device, torch.bfloat16, args.local_files_only)
            model.set_image_size(args.height, args.width)
            export_transformer(model, output_path, device, model_opset, args.height, args.width, io_dtype)
            del model
            torch.cuda.empty_cache()
        exported_model_paths.append(output_path / name / "model.onnx")

    if "transformer" in to_export:
        tokenizer.save_pretrained(str(output_path / "tokenizer"))
        print("  Saved tokenizer/")
        write_pipeline_config(output_path, args, tokenizer, layout, pixeldit_pipeline.SamplerConfig())

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
