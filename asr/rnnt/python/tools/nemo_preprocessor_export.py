# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Literal

import torch
from torch import nn

CONSTANT = 1e-5


class NeMoPreprocessorExportWrapper(nn.Module):
    def __init__(self, preprocessor: nn.Module, output_kind: Literal["lengths", "mask"]):
        super().__init__()
        self.preprocessor = preprocessor
        self.output_kind = output_kind

    def forward(self, input_signal, lengths):
        # The exported C++ pipeline feeds one unpadded waveform per invocation.
        # Deriving length from the symbolic sample dimension avoids a
        # data-dependent guard inside NeMo's normalize_batch(seq_len == 1).
        export_lengths = torch.ones_like(lengths.to(torch.long)) * input_signal.shape[1]
        features, feature_lengths = self.preprocessor(input_signal=input_signal, length=export_lengths)
        features = features.transpose(1, 2).float()
        if self.output_kind == "mask":
            frame_idx = torch.arange(features.shape[1], device=features.device)
            return features, frame_idx[None] < feature_lengths[:, None]
        return features, feature_lengths.to(torch.int32)


def export_nemo_preprocessor(
    preprocessor: nn.Module,
    output_path: Path,
    *,
    output_kind: Literal["lengths", "mask"],
    output_names: list[str],
    opset: int,
    external_data: bool,
    dynamo: bool,
    optimize: bool,
    length_dtype: torch.dtype,
) -> None:
    samples = torch.zeros(1, 16000, device="cpu")
    lengths = torch.tensor([16000], device="cpu", dtype=length_dtype)
    samples_dim = torch.export.Dim("samples", min=320)
    dynamic_axes = {
        "input_signal": {1: "samples"},
        output_names[0]: {1: "frames"},
    }
    if output_kind == "mask":
        dynamic_axes[output_names[1]] = {1: "frames"}
    options = {
        "opset_version": opset,
        "dynamo": dynamo,
        "optimize": optimize,
        "external_data": external_data,
    }
    if dynamo:
        options["dynamic_shapes"] = ({1: samples_dim}, {})
    else:
        options["dynamic_axes"] = dynamic_axes
    with exportable_normalize_batch():
        torch.onnx.export(
            NeMoPreprocessorExportWrapper(preprocessor.cpu().float().eval(), output_kind).eval(),
            (samples, lengths),
            str(output_path),
            input_names=["input_signal", "lengths"],
            output_names=output_names,
            **options,
        )


@contextlib.contextmanager
def exportable_normalize_batch():
    from nemo.collections.asr.parts.preprocessing import features

    original = features.normalize_batch
    features.normalize_batch = normalize_batch_without_item_guard
    try:
        yield
    finally:
        features.normalize_batch = original


def normalize_batch_without_item_guard(x, seq_len, normalize_type):
    if normalize_type != "per_feature":
        return x, None, None

    batch_size = x.shape[0]
    max_time = x.shape[2]
    time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
    valid_mask = time_steps < seq_len.unsqueeze(1)
    x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(axis=2)
    x_mean_denominator = valid_mask.sum(axis=1)
    x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)
    x_std = torch.sqrt(
        torch.sum(torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0) ** 2, axis=2)
        / (x_mean_denominator.unsqueeze(1) - 1.0)
    )
    x_std = x_std.masked_fill(x_std.isnan(), 0.0)
    x_std = x_std + CONSTANT
    normalized = (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    normalized = normalized.masked_fill(~valid_mask.unsqueeze(1), 0.0)
    return normalized, x_mean, x_std
