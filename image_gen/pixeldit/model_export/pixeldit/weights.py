# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the PixelDiT T2I checkpoint and the Gemma-2 text encoder."""

import inspect
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from pixdit_core.pixeldit_t2i import PixDiT_T2I

DEFAULT_MODEL_NAME = "nvidia/PixelDiT-1300M-1024px"
CHECKPOINT_FILE = "pixeldit_t2i_v1.pth"
DEFAULT_TEXT_ENCODER_NAME = "Efficient-Large-Model/gemma-2-2b-it"

# Upstream PixDiTTrainer wraps PixDiT_T2I as `core`; _repa_projector is a training-only head.
CORE_PREFIX = "core."
TRAINING_ONLY_PREFIXES = ("_repa_projector.",)


def resolve_checkpoint(model_name: str, local_files_only: bool = False) -> tuple[Path, dict]:
    """Return (checkpoint .pth path, PixDiT_T2I constructor arguments) from a HF repo ID, local dir, or local .pth."""
    local = Path(model_name).expanduser()
    if local.is_file():
        config_path = local.parent / "config.json"
        checkpoint = local
    elif local.is_dir():
        config_path = local / "config.json"
        checkpoint = local / CHECKPOINT_FILE
    else:
        checkpoint = Path(hf_hub_download(model_name, CHECKPOINT_FILE, local_files_only=local_files_only))
        config_path = Path(hf_hub_download(model_name, "config.json", local_files_only=local_files_only))

    if not config_path.is_file():
        raise FileNotFoundError(f"PixelDiT config.json not found next to the checkpoint: {config_path}")
    hf_config = json.loads(config_path.read_text(encoding="utf-8"))
    parameters = inspect.signature(PixDiT_T2I.__init__).parameters
    config = {k: v for k, v in hf_config.items() if k in parameters}
    return checkpoint.resolve(), config


def core_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    state = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=True)
    state = state.get("state_dict", state)
    core = {}
    for key, value in state.items():
        if key.startswith(CORE_PREFIX):
            core[key[len(CORE_PREFIX):]] = value
        elif not key.startswith(TRAINING_ONLY_PREFIXES):
            raise KeyError(f"Unexpected checkpoint key: {key}")
    return core


def load_pixeldit(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
) -> PixDiT_T2I:
    """Build PixDiT_T2I, load weights strictly, cast to dtype."""
    checkpoint, config = resolve_checkpoint(model_name, local_files_only)
    model = PixDiT_T2I(**config)
    model.load_state_dict(core_state_dict(checkpoint), strict=True)
    return model.eval().requires_grad_(False).to(device=device, dtype=dtype)


def dit_callable(model: PixDiT_T2I):
    """dit(x, timestep, y) for pipeline.py: casts the inputs to the model dtype first, like upstream PixDiTTrainer.

    The bf16 cast of the timestep is part of the reference behaviour (999.75 rounds to 1000)."""
    dtype = next(model.parameters()).dtype

    def dit(x, timestep, y):
        return model(x.to(dtype), timestep.to(dtype), y.to(dtype))

    return dit


def load_gemma_text_encoder(
    model_name: str = DEFAULT_TEXT_ENCODER_NAME,
    device: str = "cuda",
    local_files_only: bool = False,
    attn_implementation: str | None = None,
):
    """(tokenizer, Gemma-2 decoder stack) exactly as upstream builder.get_tokenizer_and_text_encoder."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    tokenizer.padding_side = "right"
    kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}
    text_encoder = (
        AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, local_files_only=local_files_only, **kwargs
        )
        .get_decoder()
        .eval()
        .requires_grad_(False)
        .to(device)
    )
    return tokenizer, text_encoder
