# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PixelDiT text-to-image DiT, vendored from NVlabs/PixelDiT (pixdit_core/) for ONNX export.

Numerically equivalent to upstream PixDiTTrainer + PixDiT_T2I at inference, with three
export-friendly changes:
  * RoPE uses real cos/sin tables instead of complex tensors (torch.polar / view_as_complex).
  * Patchify / unpatchify use reshape + permute instead of F.unfold / F.fold.
  * Position tables are built once by set_image_size() instead of lazily filled Python dict
    caches, so tracing never writes fake tensors into a cache and they export as graph
    constants. They are plain attributes (not buffers) so Module.to(dtype) leaves the fp32
    RoPE tables alone, as upstream's complex64 tables are.
Training-only pieces (REPA projector, attention masks, class embedder) are dropped.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Matches nvidia/PixelDiT-1300M-1024px config.json.
PIXELDIT_1300M_CONFIG = {
    "in_channels": 3,
    "patch_size": 16,
    "num_groups": 24,
    "hidden_size": 1536,
    "pixel_hidden_size": 16,
    "pixel_attn_hidden_size": 1152,
    "pixel_num_groups": 16,
    "patch_depth": 14,
    "pixel_depth": 2,
    "txt_embed_dim": 2304,
    "txt_max_length": 300,
    "text_rope_theta": 10000.0,
}


# ---------------------------------------------------------------------------
# Position tables (match pixdit_core/modules.py)
# ---------------------------------------------------------------------------


def _sincos_1d(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def pixel_sincos_pos_embed(embed_dim: int, height: int, width: int) -> np.ndarray:
    """[H*W, D] absolute pixel embedding (get_2d_sincos_pos_embed[_from_grid] upstream)."""
    grid_h = np.arange(height, dtype=np.float32)
    grid_w = np.arange(width, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, height, width)
    emb_h = _sincos_1d(embed_dim // 2, grid[0])
    emb_w = _sincos_1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def rope_2d_angles(dim: int, height: int, width: int, theta: float = 10000.0, scale: float = 16.0) -> torch.Tensor:
    """[H*W, dim/2] angles; upstream precompute_freqs_cis_2d is torch.polar(1, angles)."""
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos.reshape(-1), freqs).float()
    y_freqs = torch.outer(y_pos.reshape(-1), freqs).float()
    return torch.stack([x_freqs, y_freqs], dim=-1).reshape(height * width, -1)


def rope_1d_angles(head_dim: int, length: int, theta: float, device=None) -> torch.Tensor:
    """[L, head_dim/2] angles; upstream PixDiT_T2I.fetch_pos_text."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(0, length, device=device).float().unsqueeze(1)
    return positions * freqs.unsqueeze(0)


def apply_rotary_emb(xq, xk, cos, sin):
    """Real-valued equivalent of upstream complex apply_rotary_emb.

    xq/xk: [B, N, H, D] with (even, odd) pairs along D; cos/sin: [N, D/2].
    """
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    def rotate(x):
        x = x.float().reshape(*x.shape[:-1], -1, 2)
        x_even, x_odd = x[..., 0], x[..., 1]
        out = torch.stack([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)
        return out.flatten(3)

    return rotate(xq).type_as(xq), rotate(xk).type_as(xk)


# ---------------------------------------------------------------------------
# Building blocks (match pixdit_core/modules.py)
# ---------------------------------------------------------------------------


def apply_adaln(x, shift, scale):
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        return self.linear(self.norm(x))


class PatchTokenEmbedder(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, norm_layer=None, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class PixelTokenEmbedder(nn.Module):
    """Per-pixel linear embedding + absolute sin/cos position, grouped into [B*L, P*P, D]."""

    def __init__(self, in_channels: int, hidden_size_output: int):
        super().__init__()
        self.hidden_size_output = int(hidden_size_output)
        self.proj = nn.Linear(in_channels, self.hidden_size_output, bias=True)

    def forward(self, inputs, pos_full, patch_size: int):
        B, _, H, W = inputs.shape
        Hs, Ws = H // patch_size, W // patch_size
        x = self.proj(inputs.permute(0, 2, 3, 1))
        x = x + pos_full.to(x.dtype).view(H, W, self.hidden_size_output).unsqueeze(0)
        x = x.view(B, Hs, patch_size, Ws, patch_size, self.hidden_size_output)
        x = x.permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B * Hs * Ws, patch_size * patch_size, self.hidden_size_output)


class RotaryAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(dim // num_heads)
        self.k_norm = RMSNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rotary_emb(self.q_norm(q), self.k_norm(k), cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class PiTBlock(nn.Module):
    """Pixel-level block: per-pixel adaLN from patch tokens + attention over compressed patches."""

    def __init__(self, pixel_hidden_size, patch_hidden_size, patch_size, attn_hidden_size, attn_num_heads, mlp_ratio=4.0):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)
        self.attn_dim = int(attn_hidden_size)
        p2 = patch_size * patch_size
        self.compress_to_attn = nn.Linear(p2 * self.pixel_dim, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, p2 * self.pixel_dim, bias=True)
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(self.attn_dim, num_heads=attn_num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(nn.Linear(patch_hidden_size, 6 * self.pixel_dim * p2, bias=True))

    def forward(self, x, s_cond, batch, num_patches, cos, sin):
        BL, P2, _ = x.shape
        cond = self.adaLN_modulation(s_cond).view(BL, P2, 6 * self.pixel_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond, 6, dim=-1)
        x_norm = apply_adaln(self.norm1(x), shift_msa, scale_msa)
        x_comp = self.compress_to_attn(x_norm.view(BL, P2 * self.pixel_dim)).view(batch, num_patches, self.attn_dim)
        attn_out = self.attn(x_comp, cos, sin)
        attn_exp = self.expand_from_attn(attn_out.view(BL, self.attn_dim)).view(BL, P2, self.pixel_dim)
        x = x + gate_msa * attn_exp
        return x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))


class MMDiTJointAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm_x = RMSNorm(head_dim)
        self.k_norm_x = RMSNorm(head_dim)
        self.q_norm_y = RMSNorm(head_dim)
        self.k_norm_y = RMSNorm(head_dim)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)

    def forward(self, x, y, img_cos, img_sin, txt_cos, txt_sin):
        B, Nx, C = x.shape
        Ny = y.shape[1]
        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qx, kx, vx = self.q_norm_x(qkv_x[0]), self.k_norm_x(qkv_x[1]), qkv_x[2]
        qkv_y = self.qkv_y(y).reshape(B, Ny, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qy, ky, vy = self.q_norm_y(qkv_y[0]), self.k_norm_y(qkv_y[1]), qkv_y[2]

        qx, kx = apply_rotary_emb(qx, kx, img_cos, img_sin)
        qy, ky = apply_rotary_emb(qy, ky, txt_cos, txt_sin)

        q = torch.cat([qy.transpose(1, 2), qx.transpose(1, 2)], dim=2)
        k = torch.cat([ky.transpose(1, 2), kx.transpose(1, 2)], dim=2)
        v = torch.cat([vy.transpose(1, 2), vx.transpose(1, 2)], dim=2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        out_y = out[:, :, :Ny, :].transpose(1, 2).reshape(B, Ny, C)
        out_x = out[:, :, Ny:, :].transpose(1, 2).reshape(B, Nx, C)
        return self.proj_x(out_x), self.proj_y(out_y)


class MMDiTBlockT2I(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0):
        super().__init__()
        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation_img = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_txt = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, y, c, img_cos, img_sin, txt_cos, txt_sin):
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(6, dim=-1)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c).chunk(6, dim=-1)

        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_x, attn_y = self.attn(x_norm, y_norm, img_cos, img_sin, txt_cos, txt_sin)
        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y

        x = x + gate_mlp_x * self.mlp_x(apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y


# ---------------------------------------------------------------------------
# PixelDiT
# ---------------------------------------------------------------------------


class PixelDiT(nn.Module):
    """PixDiT_T2I with upstream PixDiTTrainer input handling (inputs cast to the weight dtype).

    forward(x [B,3,H,W], timestep [B] (already scaled by 1000), y [B,L,txt_embed_dim]) -> [B,3,H,W]
    velocity in the weight dtype. Call set_image_size(H, W) after moving/casting the model and
    before forward/export.
    """

    def __init__(
        self,
        in_channels=3,
        patch_size=16,
        num_groups=24,
        hidden_size=1536,
        pixel_hidden_size=16,
        pixel_attn_hidden_size=1152,
        pixel_num_groups=16,
        patch_depth=14,
        pixel_depth=2,
        txt_embed_dim=2304,
        txt_max_length=300,
        text_rope_theta=10000.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.num_groups = int(num_groups)
        self.hidden_size = int(hidden_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.pixel_attn_hidden_size = int(pixel_attn_hidden_size)
        self.pixel_num_groups = int(pixel_num_groups)
        self.txt_max_length = int(txt_max_length)
        self.text_rope_theta = float(text_rope_theta)

        self.pixel_embedder = PixelTokenEmbedder(in_channels, self.pixel_hidden_size)
        self.s_embedder = PatchTokenEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(hidden_size)
        self.y_embedder = PatchTokenEmbedder(txt_embed_dim, hidden_size, bias=True, norm_layer=RMSNorm)
        self.y_pos_embedding = nn.Parameter(torch.zeros(1, self.txt_max_length, hidden_size))
        self.patch_blocks = nn.ModuleList([MMDiTBlockT2I(hidden_size, num_groups) for _ in range(patch_depth)])
        self.pixel_blocks = nn.ModuleList(
            [
                PiTBlock(
                    self.pixel_hidden_size,
                    hidden_size,
                    patch_size=self.patch_size,
                    attn_hidden_size=self.pixel_attn_hidden_size,
                    attn_num_heads=self.pixel_num_groups,
                )
                for _ in range(pixel_depth)
            ]
        )
        self.final_layer = FinalLayer(self.pixel_hidden_size, self.in_channels)
        self.image_size = None

    @property
    def dtype(self) -> torch.dtype:
        return self.final_layer.linear.weight.dtype

    def set_image_size(self, height: int, width: int) -> None:
        """Build position tables for a fixed (height, width) on the model's device and dtype.

        Matches upstream: 2D RoPE angles on CPU, text RoPE on device, cos/sin in fp32, and the
        float64 pixel sin/cos table cast straight to the weight dtype.
        """
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"Image size {height}x{width} must be a multiple of patch_size={self.patch_size}")
        device = self.final_layer.linear.weight.device
        hs, ws = height // self.patch_size, width // self.patch_size
        img = rope_2d_angles(self.hidden_size // self.num_groups, hs, ws)
        pix = rope_2d_angles(self.pixel_attn_hidden_size // self.pixel_num_groups, hs, ws)
        txt = rope_1d_angles(self.hidden_size // self.num_groups, self.txt_max_length, self.text_rope_theta, device)
        pixel_pos = torch.from_numpy(pixel_sincos_pos_embed(self.pixel_hidden_size, height, width))
        self.img_cos, self.img_sin = img.cos().to(device), img.sin().to(device)
        self.pix_cos, self.pix_sin = pix.cos().to(device), pix.sin().to(device)
        self.txt_cos, self.txt_sin = txt.cos(), txt.sin()
        self.pixel_pos = pixel_pos.to(device=device, dtype=self.dtype)
        self.image_size = (height, width)

    def forward(self, x, timestep, y):
        dtype = self.dtype
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        B, _, H, W = x.shape
        if self.image_size != (H, W):
            raise RuntimeError(f"Input is {H}x{W} but set_image_size() was {self.image_size}")
        if self.pixel_pos.dtype != dtype or self.pixel_pos.device != x.device:
            raise RuntimeError("Model was moved or cast after set_image_size(); call it again")
        p = self.patch_size
        Hs, Ws = H // p, W // p
        L = Hs * Ws

        x_patches = x.view(B, self.in_channels, Hs, p, Ws, p).permute(0, 2, 4, 1, 3, 5).reshape(B, L, -1)
        t_emb = self.t_embedder(timestep.view(-1)).view(B, -1, self.hidden_size)

        Ltxt = min(y.shape[1], self.txt_max_length)
        y = y[:, :Ltxt, :]
        y_emb = self.y_embedder(y).view(B, Ltxt, self.hidden_size)
        y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb.dtype)

        condition = F.silu(t_emb)
        s = self.s_embedder(x_patches)
        txt_cos, txt_sin = self.txt_cos[:Ltxt], self.txt_sin[:Ltxt]
        for block in self.patch_blocks:
            s, y_emb = block(s, y_emb, condition, self.img_cos, self.img_sin, txt_cos, txt_sin)
        s = F.silu(t_emb + s)

        s_cond = s.view(B * L, self.hidden_size)
        x_pixels = self.pixel_embedder(x, self.pixel_pos, p)
        for block in self.pixel_blocks:
            x_pixels = block(x_pixels, s_cond, B, L, self.pix_cos, self.pix_sin)

        x_pixels = self.final_layer(x_pixels)
        x_pixels = x_pixels.view(B, Hs, Ws, p, p, self.in_channels).permute(0, 5, 1, 3, 2, 4)
        return x_pixels.reshape(B, self.in_channels, H, W)
