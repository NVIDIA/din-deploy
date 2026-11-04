# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-only separable antialiased resize for fixed tracking graph dimensions.

TensorRT RTX cannot parse ONNX Resize with antialias=1. Gather and weighted
reduction express the same triangle filter without an ONNX Resize extension.
The dynamic image preprocessing graph retains the standard ONNX operator.
"""

import math
from contextlib import contextmanager


def axis_table(source, target):
    scale = source / target
    support = max(scale, 1.0)
    taps = 2 * math.ceil(support) + 1
    indices, weights = [], []
    for position in range(target):
        center = (position + 0.5) * scale
        first = max(0, math.floor(center - support + 0.5))
        last = min(source, math.floor(center + support + 0.5))
        row = [max(0.0, 1 - abs((i + 0.5 - center) / support)) for i in range(first, last)]
        total = sum(row)
        indices.extend(list(range(first, last)) + [0] * (taps - len(row)))
        weights.append([weight / total for weight in row] + [0.0] * (taps - len(row)))
    return indices, weights


def resize(input, size):
    import torch

    value = input
    for axis, target in ((-1, size[1]), (-2, size[0])):
        source = input.shape[axis]
        if source == target:
            continue
        indices, weights = axis_table(source, target)
        idx = torch.tensor(indices, dtype=torch.int64, device=input.device)
        weight = torch.tensor(weights, dtype=input.dtype, device=input.device)
        if axis == -2:
            value = value.transpose(-2, -1)
        shape = value.shape[:-1]
        value = (value.index_select(-1, idx).reshape(*shape, target, weight.shape[-1]) * weight).sum(-1)
        if axis == -2:
            value = value.transpose(-2, -1)
    return value


@contextmanager
def tensor_antialias():
    from torch.nn import functional as F

    original = F.interpolate

    def interpolate(
        input,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
        if antialias and mode == "bilinear" and size is not None and align_corners is False:
            return resize(input, size)
        return original(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)

    F.interpolate = interpolate
    try:
        yield
    finally:
        F.interpolate = original
