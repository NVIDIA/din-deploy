# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf
import torch

from .nemo_backend import apply_attention, load_nemo_model, transcribe_paths


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    duration_seconds = audio_duration(args.audiofile)

    load_started = time.perf_counter()
    model = load_nemo_model(args.model, device=args.device)
    load_seconds = time.perf_counter() - load_started

    if args.local_window:
        apply_attention(model, "rel_pos_local_attn", [256, 256])

    transcribe_started = time.perf_counter()
    if args.timestamps != "none":
        raise ValueError("Parakeet NeMo CLI currently uses NeMo text transcription; token timestamps are not exposed.")
    text = transcribe_paths(model, [args.audiofile])[0]
    transcribe_seconds = time.perf_counter() - transcribe_started

    print(text)

    total_seconds = time.perf_counter() - started
    print()
    print(f"device: {args.device}")
    print(f"audio: {duration_seconds:.2f}s")
    print(f"load: {load_seconds:.2f}s")
    print(f"transcribe: {transcribe_seconds:.2f}s")
    print(f"rtf: {transcribe_seconds / duration_seconds:.3f}")
    print(f"total: {total_seconds:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with NeMo Parakeet TDT.")
    parser.add_argument("audiofile", type=Path)
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timestamps", "--timesteps", choices=("none", "segment", "token"), default="none")
    parser.add_argument("--local-window", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames / info.samplerate)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
