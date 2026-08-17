# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import numpy as np

from .data import (
    DEFAULT_MAX_SAMPLES_PER_LANGUAGE,
    DEFAULT_SPLIT,
    ValidationItem,
    iter_audio_dir,
    iter_fleurs,
    iter_librispeech_long,
    iter_manifest,
    maybe_group_long_form,
    parse_languages,
    parse_max_samples,
)
from .metrics import ValidationSummary, normalize_text, student_losses

T = TypeVar("T")
FAST_SAMPLES_PER_LANGUAGE = 20


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    validate_args(args)

    cpp_backend = None
    model = None
    tokenizer = None
    if args.backend == "cpp":
        if args.quantized_modelopt_path is not None:
            raise ValueError("--quantized-modelopt-path is only supported with --backend python.")
        if args.student_loss:
            raise ValueError("--student-loss is only supported with --backend python.")
        if args.max_feature_frames is not None:
            raise ValueError("--max-feature-frames is only supported with --backend python.")
        cpp_backend = load_cpp_backend(args)
    else:
        model, tokenizer = load_python_model(args)

    teacher = None
    if args.student_loss:
        if args.asr_model != "parakeet":
            raise ValueError("--student-loss is only supported with --asr-model parakeet.")
        import torch

        from rnnt.parakeet_tdt.nemo_backend import DEFAULT_MODEL_ID, load_nemo_model

        teacher = load_nemo_model(args.model or DEFAULT_MODEL_ID, device=args.device, dtype=torch.float32)
        apply_attention(teacher, args.attention, args.att_context_size)

    summary = ValidationSummary()
    prediction_path = output / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as predictions:
        items = progress(limited_items(load_items(args), args.limit), total=progress_total(args), desc="validating")
        for item in items:
            started = time.perf_counter()
            result = transcribe_item(model, tokenizer, cpp_backend, item, args)
            elapsed = time.perf_counter() - started

            has_reference = bool(normalize_text(item.text))
            if has_reference:
                wer = summary.add(
                    language=item.language,
                    duration=item.duration,
                    transcribe_seconds=elapsed,
                    reference=item.text,
                    hypothesis=result.text,
                )
            else:
                summary.add_unscored(duration=item.duration, transcribe_seconds=elapsed)
                wer = None
            record: dict[str, Any] = {
                "id": item.id,
                "language": item.language,
                "duration": item.duration,
                "reference_available": has_reference,
                "reference": item.text,
                "hypothesis": result.text,
                "reference_normalized": normalize_text(item.text),
                "hypothesis_normalized": normalize_text(result.text),
                "wer": wer.wer if wer else None,
                "errors": wer.errors if wer else None,
                "reference_words": wer.reference_words if wer else 0,
                "transcribe_seconds": elapsed,
                "rtf": elapsed / item.duration if item.duration else 0.0,
            }
            if teacher is not None:
                record["student_loss"] = student_losses(teacher, model, item.audio)
            predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
            predictions.flush()

    write_summary(output / "summary.json", args, summary)
    print(json.dumps(summary.as_dict(), indent=2))


def load_items(args: argparse.Namespace) -> Iterator[ValidationItem]:
    if args.manifest is not None:
        items: Iterable[ValidationItem] = iter_manifest(args.manifest)
    elif args.audio_dir is not None:
        items = iter_audio_dir(args.audio_dir)
    elif args.dataset == "fleurs":
        items = iter_fleurs(
            parse_languages(args.languages),
            split=args.split,
            max_samples_per_language=effective_max_samples_per_language(args),
        )
    else:
        items = iter_librispeech_long(
            split=args.split,
            max_samples=args.limit,
        )
    yield from maybe_group_long_form(items, args.long_form_target_seconds)


def limited_items(items: Iterable[ValidationItem], limit: int | None) -> Iterator[ValidationItem]:
    for index, item in enumerate(items):
        if limit is not None and index >= limit:
            break
        yield item


def progress[T](items: Iterable[T], *, total: int | None, desc: str) -> Iterable[T]:
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, total=total, desc=desc, unit="utt")


def progress_total(args: argparse.Namespace) -> int | None:
    if args.limit is not None:
        return args.limit
    if args.manifest is not None or args.audio_dir is not None or args.long_form_target_seconds:
        return None
    if args.dataset == "librispeech-long":
        return args.limit
    max_samples = effective_max_samples_per_language(args)
    if max_samples is None:
        return None
    return len(parse_languages(args.languages)) * max_samples


def load_python_model(args: argparse.Namespace):
    dtype = parse_dtype(args.dtype)
    if args.asr_model == "parakeet":
        from rnnt.parakeet_tdt.nemo_backend import DEFAULT_MODEL_ID, load_nemo_model

        model = load_nemo_model(
            args.model or DEFAULT_MODEL_ID,
            device=args.device,
            dtype=dtype,
        )
        tokenizer = None
        apply_attention(model, args.attention, args.att_context_size)
    else:
        from rnnt.nemotron_asr.nemo_backend import DEFAULT_ATT_CONTEXT_SIZE, DEFAULT_MODEL_ID, load_nemo_model

        model = load_nemo_model(
            args.model or DEFAULT_MODEL_ID,
            device=args.device,
            target_lang=args.target_lang,
            att_context_size=DEFAULT_ATT_CONTEXT_SIZE,
            dtype=dtype,
        )
        tokenizer = None

    if args.quantized_modelopt_path is not None:
        from .quantization import restore_modelopt_quantized_model

        model, _ = restore_modelopt_quantized_model(
            model,
            args.quantized_modelopt_path,
            summary_path=args.output / "quant_summary.txt",
        )
    return model, tokenizer


def load_cpp_backend(args: argparse.Namespace):
    if args.asr_model == "parakeet":
        from rnnt.parakeet_tdt.cpp_backend import CppBackend

        return CppBackend(
            model_dir=args.cpp_model_dir,
            provider=args.cpp_provider,
            ep_cache=args.cpp_ep_cache,
            ep_context_dir=args.cpp_ep_context_dir,
        )

    from rnnt.nemotron_asr.cpp_backend import CppBackend

    return CppBackend(
        model_dir=args.cpp_model_dir,
        provider=args.cpp_provider,
        ep_cache=args.cpp_ep_cache,
        ep_context_dir=args.cpp_ep_context_dir,
        lang_id=args.target_lang,
    )


def transcribe_item(model, tokenizer, cpp_backend, item: ValidationItem, args: argparse.Namespace):
    if cpp_backend is not None:
        raw = cpp_backend.transcribe_audio(item.audio)
        return SimpleNamespace(text=str(raw["text"]), raw=raw, timestamp={})
    assert model is not None
    if args.asr_model == "nemotron":
        return transcribe_nemotron_item(model, tokenizer, item, args)

    from rnnt.parakeet_tdt.nemo_backend import transcribe_audio

    del tokenizer
    return SimpleNamespace(text=transcribe_audio(model, item.audio), timestamp={})


def transcribe_nemotron_item(model, tokenizer, item: ValidationItem, args: argparse.Namespace):
    from rnnt.nemotron_asr.nemo_backend import strip_lang_tags, transcribe_audio

    del tokenizer
    text = strip_lang_tags(transcribe_audio(model, item.audio, args.target_lang))
    return SimpleNamespace(text=text.strip(), timestamp={})


def apply_attention(model, attention: str, att_context_size: str | None) -> None:
    from rnnt.parakeet_tdt.nemo_backend import apply_attention as apply_parakeet_attention

    context = None if attention == "rel_pos" else parse_att_context_size(att_context_size)
    apply_parakeet_attention(model, attention, context)


def parse_att_context_size(value: str | None) -> list[int]:
    if not value:
        return [256, 256]
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--att-context-size must have two comma-separated integers.")
    return parts


def parse_dtype(value: str):
    import torch

    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {value}")


def validate_args(args: argparse.Namespace) -> None:
    if args.asr_model == "nemotron":
        if args.student_loss:
            raise ValueError("--student-loss is only supported with --asr-model parakeet.")
        if args.max_feature_frames is not None:
            raise ValueError("--max-feature-frames is only supported with --asr-model parakeet.")
        if args.window_seconds is not None:
            raise ValueError("--window-seconds is only supported with --asr-model parakeet.")
    elif args.backend == "python":
        if args.max_feature_frames is not None:
            raise ValueError("--max-feature-frames belonged to the removed custom Parakeet sampler.")
        if args.window_seconds is not None:
            raise ValueError("--window-seconds belonged to the removed custom Parakeet sampler.")


def effective_max_samples_per_language(args: argparse.Namespace) -> int | None:
    if args.fast:
        return args.fast_samples_per_language
    return parse_max_samples(args.max_samples_per_language)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    summary: ValidationSummary,
) -> None:
    config = {
        "dataset": args.dataset,
        "backend": args.backend,
        "manifest": str(args.manifest) if args.manifest else None,
        "audio_dir": str(args.audio_dir) if args.audio_dir else None,
        "languages": (
            parse_languages(args.languages)
            if args.manifest is None and args.audio_dir is None and args.dataset == "fleurs"
            else None
        ),
        "split": args.split if args.manifest is None and args.audio_dir is None else None,
        "max_samples_per_language": (
            args.max_samples_per_language
            if args.manifest is None and args.audio_dir is None and args.dataset == "fleurs"
            else None
        ),
        "fast": args.fast,
        "fast_samples_per_language": args.fast_samples_per_language if args.fast else None,
        "effective_max_samples_per_language": (
            effective_max_samples_per_language(args)
            if args.manifest is None and args.audio_dir is None and args.dataset == "fleurs"
            else None
        ),
        "attention": args.attention,
        "att_context_size": args.att_context_size,
        "window_seconds": args.window_seconds,
        "max_feature_frames": args.max_feature_frames,
        "budget_left_context_seconds": args.budget_left_context_seconds,
        "budget_right_context_seconds": args.budget_right_context_seconds,
        "budget_chunk_seconds": args.budget_chunk_seconds,
        "long_form_target_seconds": args.long_form_target_seconds,
        "student_loss": args.student_loss,
        "asr_model": args.asr_model,
        "model": args.model,
        "model_dir": str(args.model_dir) if args.model_dir else None,
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "quantized_modelopt_path": str(args.quantized_modelopt_path) if args.quantized_modelopt_path else None,
        "target_lang": args.target_lang if args.asr_model == "nemotron" else None,
        "cpp_model_dir": str(args.cpp_model_dir) if args.backend == "cpp" else None,
        "cpp_provider": args.cpp_provider if args.backend == "cpp" else None,
        "cpp_ep_cache": str(args.cpp_ep_cache) if args.backend == "cpp" else None,
        "cpp_ep_context_dir": str(args.cpp_ep_context_dir) if args.backend == "cpp" else None,
    }
    payload = {"config": config, "summary": summary.as_dict()}
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small RNNT validation pass.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", choices=("fleurs", "librispeech-long"), default="fleurs")
    source.add_argument("--manifest", type=Path)
    source.add_argument("--audio-dir", type=Path)
    parser.add_argument("--languages", default=None, help="Comma-separated FLEURS configs.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--max-samples-per-language",
        default=str(DEFAULT_MAX_SAMPLES_PER_LANGUAGE),
        help="FLEURS cap per language; use 0 or none for the full split.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional global item limit.")
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shortcut for a small validation run over each configured language.",
    )
    parser.add_argument(
        "--fast-samples-per-language",
        type=int,
        default=FAST_SAMPLES_PER_LANGUAGE,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument("--asr-model", choices=("parakeet", "nemotron"), default="parakeet")
    parser.add_argument("--model", default=None, help="Optional Hugging Face model id or local model path.")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--quantized-modelopt-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--attention", choices=("rel_pos", "rel_pos_local_attn"), default="rel_pos")
    parser.add_argument("--att-context-size", default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--max-feature-frames", type=int, default=None)
    parser.add_argument("--budget-left-context-seconds", type=float, default=10.0)
    parser.add_argument("--budget-right-context-seconds", type=float, default=5.0)
    parser.add_argument(
        "--budget-chunk-seconds",
        type=float,
        default=None,
        help="Optional core chunk override. By default, use the largest chunk that fits the frame budget.",
    )
    parser.add_argument("--long-form-target-seconds", type=float, default=None)
    parser.add_argument("--student-loss", action="store_true")
    parser.add_argument("--target-lang", default="auto")
    parser.add_argument("--cpp-model-dir", type=Path, default=None)
    parser.add_argument("--cpp-provider", choices=("trt-rtx", "cpu"), default="trt-rtx")
    parser.add_argument("--cpp-ep-cache", type=Path, default=None)
    parser.add_argument("--cpp-ep-context-dir", type=Path, default=None)

    args = parser.parse_args()
    resolve_defaults(args)
    return args


def resolve_defaults(args: argparse.Namespace) -> None:
    if args.cpp_model_dir is None:
        args.cpp_model_dir = Path("artifacts") / args.asr_model / "onnx"
    if args.cpp_ep_cache is None:
        args.cpp_ep_cache = Path("artifacts") / args.asr_model / "trt_rtx_cache"
    if args.cpp_ep_context_dir is None:
        args.cpp_ep_context_dir = Path("artifacts") / args.asr_model / "ep_context"


if __name__ == "__main__":
    main()
