# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


def test_validation_cli_accepts_nemotron_quantized_checkpoint(monkeypatch, tmp_path):
    from rnnt.validation import __main__ as validation

    checkpoint = tmp_path / "auto_quantize_model.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "validation",
            "--output",
            str(tmp_path / "out"),
            "--asr-model",
            "nemotron",
            "--quantized-modelopt-path",
            str(checkpoint),
        ],
    )

    args = validation.parse_args()

    assert args.asr_model == "nemotron"
    assert args.quantized_modelopt_path == checkpoint
    assert args.cpp_model_dir == Path("artifacts/nemotron/onnx")


def test_validation_cli_defaults_to_parakeet_artifacts(monkeypatch, tmp_path):
    from rnnt.validation import __main__ as validation

    monkeypatch.setattr("sys.argv", ["validation", "--output", str(tmp_path / "out")])

    args = validation.parse_args()

    assert args.asr_model == "parakeet"
    assert args.cpp_model_dir == Path("artifacts/parakeet/onnx")
