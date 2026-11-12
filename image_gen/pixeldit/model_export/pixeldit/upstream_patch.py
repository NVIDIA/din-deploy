# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export-time patches for upstream PixelDiT T2I (pip package `pixeldit`, imported as `pixdit_core`).

The DiT is upstream's `pixdit_core.pixeldit_t2i.PixDiT_T2I`, installed from NVlabs/PixelDiT at UPSTREAM_COMMIT
(the `pixeldit` extra in the repository's root pyproject.toml). Upstream runs as-is everywhere except inside
`export_patches(...)`, which makes three changes for the duration of the ONNX export and restores everything on exit:

  * RoPE uses real (cos, sin) pairs instead of complex tensors. Required: upstream unsqueezes complex tensors
    (precompute_freqs_cis_2d, and freqs_cis[None, :, None, :] in apply_rotary_emb), and the ONNX exporter has no
    lowering for unsqueeze on complex inputs. Other complex ops (polar, view_as_complex/real, multiply) would
    export. Matches upstream up to float rounding (last-bit differences in the rotation tables and in the
    rotation itself).
  * F.unfold / F.fold (patchify / unpatchify, kernel == stride) become reshape + permute, which is bit-identical.
    Required for TensorRT RTX: its ONNX parser rejects Col2Im (what F.fold exports to) in every dtype, and in bf16
    ONNX Runtime's CPU provider has no Col2Im to fall back to either.
  * Position tables are computed eagerly for the export size and cached before tracing, so the exporter sees
    constants and never writes traced tensors into upstream's lazily filled dict caches.

The patches replace module-level functions and torch.nn.functional.fold/unfold process-wide while active, so the
export must not run concurrently with other code using them.
"""

import contextlib
import importlib.metadata
import json
import warnings
from typing import Tuple

import torch
import torch.nn.functional as F

import pixdit_core.modules as upstream_modules
import pixdit_core.pixeldit_c2i as upstream_c2i
import pixdit_core.pixeldit_t2i as upstream_t2i

# Commit of NVlabs/PixelDiT these patches were written and validated against (keep in sync with the `pixeldit` extra
# in the root pyproject.toml).
UPSTREAM_COMMIT = "f7d7c35f71893d81c6e71d2576ea80601a1e8529"


# ---------------------------------------------------------------------------
# Real-valued RoPE: rotations as [N, head_dim // 2, 2] (cos, sin), the layout of torch.view_as_real
# ---------------------------------------------------------------------------


def _rotations(angles: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, scale=16.0):
    """Upstream modules.precompute_freqs_cis_2d, returning (cos, sin) pairs instead of complex numbers."""
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos.reshape(-1), freqs).float()
    y_freqs = torch.outer(y_pos.reshape(-1), freqs).float()
    # [N, dim / 4, 2 (x, y), 2 (cos, sin)] -> [N, dim / 2, 2], alternating x and y like upstream's complex table.
    rotations = torch.stack([_rotations(x_freqs), _rotations(y_freqs)], dim=-2)
    return rotations.reshape(height * width, -1, 2)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Upstream modules.apply_rotary_emb with real arithmetic: (re + i im) * (cos + i sin)."""
    freqs_cis = freqs_cis[None, :, None, :, :]
    cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        real, imag = x_[..., 0], x_[..., 1]
        return torch.stack([real * cos - imag * sin, real * sin + imag * cos], dim=-1).flatten(3).type_as(x)

    return rotate(xq), rotate(xk)


def fetch_pos_text(self, length, device):
    """Upstream PixDiT_T2I.fetch_pos_text, building (cos, sin) pairs instead of torch.polar."""
    if length in self.precompute_pos_txt:
        return self.precompute_pos_txt[length].to(device)
    head_dim = self.hidden_size // self.num_groups
    freqs = 1.0 / (self.text_rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.arange(0, length, device=device).float().unsqueeze(1) * freqs.unsqueeze(0)
    self.precompute_pos_txt[length] = _rotations(angles)
    return self.precompute_pos_txt[length]


# ---------------------------------------------------------------------------
# F.unfold / F.fold for non-overlapping patches (kernel == stride) as reshape + permute
# ---------------------------------------------------------------------------


def _patch_size(kernel_size, stride, kwargs) -> int:
    kernel = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    step = stride if isinstance(stride, int) else stride[0]
    square = isinstance(kernel_size, int) or kernel_size[0] == kernel_size[1]
    if not square or kernel != step or kwargs.get("padding", 0) not in (0, (0, 0)) or kwargs.get(
        "dilation", 1
    ) not in (1, (1, 1)):
        raise NotImplementedError("export patch only supports unfold/fold with square kernel == stride, no padding")
    return kernel


def unfold(x, kernel_size, stride=1, **kwargs):
    """F.unfold for kernel == stride: [B, C, H, W] -> [B, C * p * p, L], same element order."""
    p = _patch_size(kernel_size, stride, kwargs)
    B, C, H, W = x.shape
    return x.reshape(B, C, H // p, p, W // p, p).permute(0, 1, 3, 5, 2, 4).reshape(B, C * p * p, (H // p) * (W // p))


def fold(cols, output_size, kernel_size, stride=1, **kwargs):
    """F.fold for kernel == stride: [B, C * p * p, L] -> [B, C, H, W], same result."""
    p = _patch_size(kernel_size, stride, kwargs)
    H, W = output_size
    B = cols.shape[0]
    C = cols.shape[1] // (p * p)
    return cols.reshape(B, C, p, p, H // p, W // p).permute(0, 1, 4, 2, 5, 3).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Patch context
# ---------------------------------------------------------------------------


def installed_upstream_commit() -> str | None:
    """Commit of the installed `pixeldit` package, if it was installed from git; None otherwise."""
    try:
        direct_url = importlib.metadata.distribution("pixeldit").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not direct_url:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def _require(obj, name: str) -> None:
    if not hasattr(obj, name):
        owner = getattr(obj, "__name__", type(obj).__name__)
        raise RuntimeError(
            f"upstream PixelDiT changed: {owner}.{name} not found; update pixeldit/upstream_patch.py "
            f"(written against NVlabs/PixelDiT@{UPSTREAM_COMMIT[:7]})"
        )


def _clear_caches(model) -> None:
    model.precompute_pos.clear()
    model.precompute_pos_txt.clear()
    for block in model.pixel_blocks:
        block._pos_cache.clear()
    model.last_repa_tokens = None


@contextlib.contextmanager
def export_patches(model: upstream_t2i.PixDiT_T2I, height: int, width: int, txt_len: int):
    """Patch upstream PixDiT_T2I for export at a fixed image size and text length; restores everything on exit."""
    commit = installed_upstream_commit()
    if commit is not None and commit != UPSTREAM_COMMIT:
        warnings.warn(
            f"pixeldit is installed from NVlabs/PixelDiT@{commit[:7]}, but these export patches were validated "
            f"against @{UPSTREAM_COMMIT[:7]}; re-run the PixelDiT export validation after changing the version"
        )

    changes = [
        (upstream_modules, "apply_rotary_emb", apply_rotary_emb),  # RotaryAttention (PiT blocks)
        (upstream_t2i, "apply_rotary_emb", apply_rotary_emb),  # MMDiTJointAttention, imported by name
        (upstream_modules, "precompute_freqs_cis_2d", precompute_freqs_cis_2d),
        (upstream_t2i, "precompute_freqs_cis_2d", precompute_freqs_cis_2d),  # PixDiT_T2I.fetch_pos
        (upstream_c2i, "precompute_freqs_cis_2d", precompute_freqs_cis_2d),
        (model, "fetch_pos_text", fetch_pos_text.__get__(model)),
        (F, "unfold", unfold),  # PixDiT_T2I.forward calls torch.nn.functional.unfold / fold inline
        (F, "fold", fold),
    ]
    changes += [(block, "_rope_fn", precompute_freqs_cis_2d) for block in model.pixel_blocks]  # bound at init
    for obj, name, _ in changes:
        _require(obj, name)
    for name in ("precompute_pos", "precompute_pos_txt", "fetch_pos", "pixel_embedder", "last_repa_tokens"):
        _require(model, name)
    for block in model.pixel_blocks:
        _require(block, "_pos_cache")
        _require(block, "_fetch_pos")
    _require(model.pixel_embedder, "_fetch_pixel_pos_image")

    # (obj, name, previous value, whether obj itself held the attribute, e.g. not a class method)
    saved = [(obj, name, getattr(obj, name), name in vars(obj)) for obj, name, _ in changes]
    try:
        for obj, name, value in changes:
            setattr(obj, name, value)
        _clear_caches(model)  # drop complex tables from earlier eager runs
        # Fill the caches with real tensors so tracing reads constants instead of writing traced tensors.
        parameter = next(model.parameters())
        p = model.patch_size
        model.fetch_pos(height // p, width // p, parameter.device)
        model.fetch_pos_text(min(txt_len, model.txt_max_length), parameter.device)
        for block in model.pixel_blocks:
            block._fetch_pos(height // p, width // p, parameter.device)
        if model.use_pixel_abs_pos:
            model.pixel_embedder._fetch_pixel_pos_image(height, width, parameter.device, parameter.dtype)
        yield model
    finally:
        for obj, name, value, own in reversed(saved):
            if own:
                setattr(obj, name, value)
            else:
                delattr(obj, name)  # e.g. the instance override of the PixDiT_T2I.fetch_pos_text method
        _clear_caches(model)  # upstream rebuilds its own complex tables on the next eager call
