# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    from rnnt.audio import audio_duration

    from .nemo_backend import DEFAULT_ATT_CONTEXT_SIZE, load_nemo_model, transcribe_paths

    args = parse_args()
    started = time.perf_counter()
    duration_seconds = sum(audio_duration(path) for path in args.audiofile)

    load_started = time.perf_counter()
    model = load_nemo_model(
        args.model,
        device=args.device,
        target_lang=args.target_lang,
        att_context_size=DEFAULT_ATT_CONTEXT_SIZE,
    )
    load_seconds = time.perf_counter() - load_started

    transcribe_started = time.perf_counter()
    texts = transcribe_paths(model, args.audiofile, args.target_lang)
    transcribe_seconds = time.perf_counter() - transcribe_started

    for path, text in zip(args.audiofile, texts, strict=True):
        if args.strip_lang_tags:
            from .nemo_backend import strip_lang_tags

            text = strip_lang_tags(text)
        if len(args.audiofile) > 1:
            print(f"{path}: {text}")
        else:
            print(text)

    total_seconds = time.perf_counter() - started
    print()
    print(f"device: {model.device}")
    print(f"audio: {duration_seconds:.2f}s")
    print(f"load: {load_seconds:.2f}s")
    print(f"transcribe: {transcribe_seconds:.2f}s")
    print(f"rtf: {transcribe_seconds / duration_seconds:.3f}")
    print(f"total: {total_seconds:.2f}s")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with Nemotron 3.5 ASR.")
    parser.add_argument("audiofile", nargs="+", type=Path)
    parser.add_argument("--model", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-lang", "--target_lang", default="auto")
    parser.add_argument("--strip-lang-tags", "--strip_lang_tags", type=parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-strip-lang-tags", "--no_strip_lang_tags", dest="strip_lang_tags", action="store_false")
    parser.add_argument("--keep-lang-tags", dest="strip_lang_tags", action="store_false")
    return parser.parse_args(argv)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("boolean value must be true or false.")


if __name__ == "__main__":
    main()
