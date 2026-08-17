# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from torch import nn

from rnnt.audio import Audio

DEFAULT_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
DEFAULT_ATT_CONTEXT_SIZE = (56, 3)
LANG_TAG_RE = re.compile(r"<[^>\s]+>")


def strip_lang_tags(text: str) -> str:
    return LANG_TAG_RE.sub("", text).strip()


def load_nemo_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    device: str | torch.device | None = None,
    target_lang: str = "auto",
    att_context_size: tuple[int, int] = DEFAULT_ATT_CONTEXT_SIZE,
    dtype: torch.dtype = torch.float32,
    configure_for_export: bool = False,
) -> nn.Module:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    nemo_model = _load_nemo_asr_model(model_id, target_device)
    nemo_model = nemo_model.to(device=target_device, dtype=dtype)
    nemo_model.eval()
    _configure_nemo_model(
        nemo_model,
        target_lang=target_lang,
        att_context_size=att_context_size,
        configure_for_export=configure_for_export,
    )
    return nemo_model


def prompt_dictionary(nemo_model: nn.Module) -> dict[str, int]:
    for key in ("train_ds", "validation_ds", "test_ds"):
        data = nemo_model.cfg.get(key)
        if data is not None and "prompt_dictionary" in data:
            return {str(lang): int(prompt_id) for lang, prompt_id in data.prompt_dictionary.items()}
    raise KeyError("prompt_dictionary")


def prompt_id(nemo_model: nn.Module, target_lang: str) -> int:
    prompts = prompt_dictionary(nemo_model)
    if target_lang not in prompts:
        raise ValueError(f"Unsupported target language: {target_lang}")
    return prompts[target_lang]


def transcribe_paths(nemo_model: nn.Module, paths: list[str | Path], target_lang: str) -> list[str]:
    override_config = _prompt_transcribe_config(target_lang)
    results = nemo_model.transcribe([str(path) for path in paths], override_config=override_config)
    return [getattr(result, "text", str(result)).strip() for result in results]


def transcribe_audio(nemo_model: nn.Module, audio: Audio, target_lang: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        sf.write(handle.name, audio.samples, audio.sampling_rate)
        return transcribe_paths(nemo_model, [handle.name], target_lang)[0]


def write_tokenizer_artifacts(nemo_model: nn.Module, output: Path) -> None:
    tokenizer = nemo_model.tokenizer.tokenizer
    (output / "tokenizer.model").write_bytes(tokenizer.serialized_model_proto())
    pieces = [tokenizer.id_to_piece(index) for index in range(tokenizer.get_piece_size())]
    (output / "vocab.txt").write_text("\n".join(pieces) + "\n", encoding="utf-8")


def configure_nemo_encoder_sdpa(encoder: nn.Module) -> None:
    encoder_flag_updated = False
    if hasattr(encoder, "use_pytorch_sdpa"):
        encoder.use_pytorch_sdpa = True
        encoder_flag_updated = True

    layer_flags_updated = 0
    attention_flags_updated = 0
    attention_classes: dict[str, int] = {}
    for layer in getattr(encoder, "layers", []):
        if hasattr(layer, "use_pytorch_sdpa"):
            layer.use_pytorch_sdpa = True
            layer_flags_updated += 1
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            continue
        attention_classes[type(self_attn).__name__] = attention_classes.get(type(self_attn).__name__, 0) + 1
        if hasattr(self_attn, "use_pytorch_sdpa"):
            self_attn.use_pytorch_sdpa = True
            attention_flags_updated += 1

    print(
        "NeMo encoder SDPA export config: "
        f"encoder_flag={encoder_flag_updated}, "
        f"layer_flags={layer_flags_updated}, "
        f"attention_flags={attention_flags_updated}, "
        f"attention_classes={attention_classes}",
        file=sys.stderr,
    )


def _load_nemo_asr_model(model_id: str, device: torch.device) -> nn.Module:
    try:
        from nemo.collections import asr as nemo_asr
    except ImportError as exc:
        raise RuntimeError("Nemotron requires the optional `nemo_toolkit[asr]` dependency.") from exc

    model_path = Path(model_id)
    if model_path.is_file() and model_path.suffix == ".nemo":
        return nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=device)
    return nemo_asr.models.ASRModel.from_pretrained(model_name=model_id, map_location=device)


def _configure_nemo_model(
    nemo_model: nn.Module,
    *,
    target_lang: str,
    att_context_size: tuple[int, int],
    configure_for_export: bool,
) -> None:
    if hasattr(nemo_model, "set_inference_prompt"):
        nemo_model.set_inference_prompt(target_lang)
    if configure_for_export and hasattr(nemo_model, "set_export_config"):
        nemo_model.set_export_config({"cache_support": True})
    encoder = nemo_model.encoder
    if hasattr(encoder, "set_default_att_context_size"):
        encoder.set_default_att_context_size(list(att_context_size))
    if configure_for_export:
        configure_nemo_encoder_sdpa(encoder)
        if hasattr(encoder, "export_cache_support"):
            encoder.export_cache_support = True
    encoder.eval()


def _prompt_transcribe_config(target_lang: str) -> Any | None:
    try:
        from nemo.collections.asr.models.rnnt_bpe_models_prompt import RNNTPromptTranscribeConfig
    except Exception:
        return None
    override_config = RNNTPromptTranscribeConfig(batch_size=1, verbose=False, target_lang=target_lang)
    override_config.use_lhotse = False
    return override_config
