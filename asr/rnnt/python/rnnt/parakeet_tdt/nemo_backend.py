# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from torch import nn

from rnnt.audio import Audio

artifacts_base = os.environ.get("DIN_DEPLOY_ARTIFACTS_PATH")
ARTIFACTS_DIR = Path(artifacts_base) / "parakeet" if artifacts_base else Path("artifacts") / "parakeet"
MODEL_DIR = ARTIFACTS_DIR / "model"
ONNX_DIR = ARTIFACTS_DIR / "onnx"
DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_MODEL_DIR = MODEL_DIR


def load_nemo_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_nemo_asr_model(model_id, target_device)
    model = model.to(device=target_device, dtype=dtype)
    model.eval()
    return model


def transcribe_paths(model: nn.Module, paths: list[str | Path], batch_size: int = 1) -> list[str]:
    results = model.transcribe([str(path) for path in paths], batch_size=batch_size)
    return [_result_text(result) for result in results]


def transcribe_audio(model: nn.Module, audio: Audio) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        sf.write(handle.name, audio.samples, audio.sampling_rate)
        return transcribe_paths(model, [handle.name])[0]


def apply_attention(model: nn.Module, attention: str, att_context_size: list[int] | None) -> None:
    if not hasattr(model, "change_attention_model"):
        return
    if attention == "rel_pos":
        model.change_attention_model("rel_pos", None)
    else:
        model.change_attention_model("rel_pos_local_attn", att_context_size or [256, 256])


def configure_encoder_sdpa(encoder: nn.Module) -> None:
    for module in [encoder, *getattr(encoder, "layers", [])]:
        if hasattr(module, "use_pytorch_sdpa"):
            module.use_pytorch_sdpa = True
        self_attn = getattr(module, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "use_pytorch_sdpa"):
            self_attn.use_pytorch_sdpa = True


def write_tokenizer_artifacts(model: nn.Module, output: Path) -> None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("NeMo Parakeet model does not expose a tokenizer.")

    sentencepiece = getattr(tokenizer, "tokenizer", tokenizer)
    if hasattr(sentencepiece, "get_piece_size") and hasattr(sentencepiece, "id_to_piece"):
        vocab = {sentencepiece.id_to_piece(index): index for index in range(sentencepiece.get_piece_size())}
        (output / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}}, indent=2), encoding="utf-8")
        return
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "save"):
        tokenizer.tokenizer.save(str(output / "tokenizer.json"))
        return
    if hasattr(tokenizer, "save_to"):
        tokenizer.save_to(str(output / "tokenizer.json"))
        return
    if hasattr(tokenizer, "vocab_path"):
        vocab_path = Path(tokenizer.vocab_path)
        if vocab_path.is_file():
            (output / "tokenizer.json").write_bytes(vocab_path.read_bytes())
            return
    raise RuntimeError("Unable to write tokenizer.json from NeMo Parakeet tokenizer.")


def _load_nemo_asr_model(model_id: str, device: torch.device) -> nn.Module:
    try:
        from nemo.collections import asr as nemo_asr
    except ImportError as exc:
        raise RuntimeError("Parakeet requires the optional `nemo_toolkit[asr]` dependency.") from exc

    model_path = Path(model_id)
    if model_path.is_file() and model_path.suffix == ".nemo":
        return nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=device)
    return nemo_asr.models.ASRModel.from_pretrained(model_name=model_id, map_location=device)


def _result_text(result: Any) -> str:
    if hasattr(result, "text"):
        return str(result.text).strip()
    return str(result).strip()
