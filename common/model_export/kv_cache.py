# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity KV updates for autoregressive ONNX exports."""

import torch


class FixedKVCache:
    """Per-forward adapter; K/V are each layer's first two tensors, with sequence on axis 2.

    Call update in layer order; present contains the updated K/V pairs.
    Positions contain one index per new token. TensorScatter requires contiguous
    positions, opset 24, and matching past/present buffer bindings in TensorRT RTX.
    """

    def __init__(self, past, positions, *, layer_stride=2, tensor_scatter=False):
        self.past, self.positions, self.present = past, positions, []
        if positions.ndim != 1:
            self.positions = positions.reshape(-1)
        self.layer_stride = layer_stride
        self.tensor_scatter = tensor_scatter

    def update(self, key, value, layer_idx):
        start = self.layer_stride * layer_idx
        updated = []
        for cache, data in zip(self.past[start : start + 2], (key, value), strict=True):
            if self.tensor_scatter and torch.onnx.is_in_onnx_export():
                data = torch.onnx.ops.symbolic(
                    "TensorScatter",
                    (cache, data, self.positions[:1]),
                    {"axis": 2, "mode": "linear"},
                    dtype=data.dtype,
                    shape=cache.shape,
                    version=24,
                )
            else:
                data = cache.index_copy(2, self.positions, data)
            updated.append(data)
        self.present.extend(updated)
        return tuple(updated)
