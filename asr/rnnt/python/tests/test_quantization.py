# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def test_quantization_cli_sets_model_defaults(monkeypatch, tmp_path):
    from rnnt.validation import quantization

    monkeypatch.setattr(
        "sys.argv",
        ["quantization", "--model", "nemotron", "--precision", "fp8", "--output", str(tmp_path)],
    )

    args = quantization.parse_args()

    assert args.effective_bits == 8.0


def test_quantization_cli_defaults_int4_effective_bits(monkeypatch, tmp_path):
    from rnnt.validation import quantization

    monkeypatch.setattr(
        "sys.argv",
        ["quantization", "--model", "parakeet", "--precision", "int4", "--output", str(tmp_path)],
    )

    args = quantization.parse_args()

    assert args.effective_bits == 4.8
