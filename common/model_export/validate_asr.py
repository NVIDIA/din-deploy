# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-file ASR validation through the native CLI; no Python runtime bindings."""

import argparse
import contextlib
import json
import math
import subprocess
import sys
from pathlib import Path


def validation_parser(model_id: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare the C++ ASR CLI with a NeMo reference on real audio.")
    parser.add_argument("--audio", type=Path, required=True, help="Real audio file (16 kHz mono WAV recommended).")
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model", default=model_id, help="NeMo model ID or local .nemo checkpoint used for export.")
    parser.add_argument("--device", default="cpu", help="Device for the NeMo reference.")
    parser.add_argument("--provider", choices=("cpu", "trt-rtx"), default="cpu")
    parser.add_argument("--truth", type=Path, help="Optional human reference transcript.")
    parser.add_argument(
        "--max-wer", type=float, help="Fail above this WER; uses --truth when supplied, otherwise NeMo."
    )
    return parser


def validate(args: argparse.Namespace, reference, cli_args=()) -> None:
    # Reject missing inputs before loading a large reference model.
    for path in (args.audio, args.cli):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_wer is not None and (not math.isfinite(args.max_wer) or args.max_wer < 0):
        raise ValueError("--max-wer must be finite and nonnegative.")
    truth = args.truth.read_text(encoding="utf-8").strip() if args.truth else None
    command = [
        str(args.cli.resolve()),
        str(args.audio.resolve()),
        "--model-dir",
        str(args.model_dir.resolve()),
        "--provider",
        args.provider,
        "--json",
        *cli_args,
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    hypothesis = json.loads(result.stdout)["transcription"]
    with contextlib.redirect_stdout(sys.stderr):
        expected = reference(args)
    from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase, wer

    normalize = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
    score = wer(normalize(expected), normalize(hypothesis))
    output = {"reference": expected, "transcription": hypothesis, "wer_against_nemo": score}
    if truth is not None:
        output["truth"] = truth
        score = output["wer_against_truth"] = wer(normalize(truth), normalize(hypothesis))
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.max_wer is not None and score > args.max_wer:
        raise SystemExit(f"WER {score:.4f} exceeds {args.max_wer:.4f}")
