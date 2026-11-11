# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# !/usr/bin/env python3
"""Export ComfyUI's checkpoint-backed SeedVR2 VAE and DiT to ONNX.

The VAE encoder/decoder are separate ONNX components with dynamic image
height/width. The DiT is exported as a third component with matching dynamic
latent height/width. Batch and the 58x5120 conditioning tensor are fixed.
Images have one frame; video exports always have a bounded dynamic temporal
axis (T>=5). --video-frames chooses the capture example, not a fixed clip length.
DiT computation defaults to BF16 with FP16 graph I/O; the VAE remains FP16.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

DEFAULT_REPO_ID = "Comfy-Org/SeedVR2"
CONTEXT_TOKENS = 58
CONTEXT_WIDTH = 5120
LATENT_CHANNELS = 16
# Shared with the C++ kVideoFrameRange contract; video exports need no sidecar.
VIDEO_FRAME_RANGE = (5, 33)
DIT_INPUT_NAMES = (
    "latent",
    "timestep",
    "context",
    "condition",
    "normal_window_tgt_idx",
    "normal_window_src_idx",
    "normal_concat_tgt_idx",
    "normal_concat_src_idx",
    "normal_padded_flat_idx",
    "normal_padded_gather_idx",
    "normal_valid_keys",
    "normal_freqs",
    "shifted_window_tgt_idx",
    "shifted_window_src_idx",
    "shifted_concat_tgt_idx",
    "shifted_concat_src_idx",
    "shifted_padded_flat_idx",
    "shifted_padded_gather_idx",
    "shifted_valid_keys",
    "shifted_freqs",
)


@dataclass(frozen=True)
class WindowMetadata:
    """Resolution-dependent tensors consumed by one DiT window configuration."""

    window_tgt_idx: torch.Tensor
    window_src_idx: torch.Tensor
    window_shape: torch.Tensor
    concat_tgt_idx: torch.Tensor
    concat_src_idx: torch.Tensor
    padded_flat_idx: torch.Tensor
    padded_gather_idx: torch.Tensor
    valid_keys: torch.Tensor
    freqs: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.window_tgt_idx,
            self.window_src_idx,
            self.concat_tgt_idx,
            self.concat_src_idx,
            self.padded_flat_idx,
            self.padded_gather_idx,
            self.valid_keys,
            self.freqs,
        )


_METADATA_CACHE: dict[tuple[object, ...], tuple[WindowMetadata, WindowMetadata]] = {}


def _attention_signature(block: torch.nn.Module) -> tuple[object, ...]:
    attention = block.attn
    rope = attention.rope
    return (
        attention.window_method,
        tuple(attention.window),
        attention.heads,
        attention.head_dim,
        type(rope),
        bool(getattr(rope, "mm", False)),
        tuple(rope.rope.freqs.shape) if rope is not None else None,
    )


def validate_shared_metadata_config(
        diffusion_model: torch.nn.Module,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Assert that all blocks can share one normal and one shifted pack."""
    if len(diffusion_model.blocks) < 2:
        raise ValueError("SeedVR2 must contain at least a normal and shifted attention block.")

    normal = diffusion_model.blocks[0]
    shifted = diffusion_model.blocks[1]
    expected_by_method = {
        normal.attn.window_method: _attention_signature(normal),
        shifted.attn.window_method: _attention_signature(shifted),
    }
    if len(expected_by_method) != 2:
        raise ValueError("The first two SeedVR2 blocks do not use distinct normal and shifted windows.")

    for index, block in enumerate(diffusion_model.blocks):
        method = block.attn.window_method
        if method not in expected_by_method:
            raise ValueError(f"Block {index} has unsupported window method {method!r}.")
        if _attention_signature(block) != expected_by_method[method]:
            raise ValueError(f"Block {index} is incompatible with the shared {method!r} metadata pack.")
    return normal.attn, shifted.attn


@torch.no_grad()
def _generate_metadata_pack(
        attention: torch.nn.Module,
        latent_height: int,
        latent_width: int,
        text_length: int,
        device: torch.device,
        dtype: torch.dtype,
    temporal: int = 1,
) -> WindowMetadata:
    import comfy.ldm.seedvr.model as seedvr_model

    patch_height = latent_height // 2
    patch_width = latent_width // 2
    vid_shape = torch.tensor([[temporal, patch_height, patch_width]], device=device, dtype=torch.long)

    def make_window(x: torch.Tensor) -> list[torch.Tensor]:
        temporal, height, width, _ = x.shape
        slices = attention.window_op((temporal, height, width), attention.window)
        return [x[st, sh, sw] for st, sh, sw in slices]

    partition, _, window_shape, window_count, _, _ = seedvr_model.window_idx(vid_shape, make_window)
    patched_tokens = temporal * patch_height * patch_width
    window_tgt_idx = partition(torch.arange(patched_tokens, device=device)).to(torch.long)
    window_src_idx = torch.argsort(window_tgt_idx)

    vid_len = window_shape.prod(-1)
    txt_len = torch.tensor([text_length], device=device, dtype=torch.long)
    concat, _ = seedvr_model.repeat_concat_idx(vid_len, txt_len, window_count)
    video_ids = torch.arange(patched_tokens, device=device)
    text_ids = torch.arange(patched_tokens, patched_tokens + text_length, device=device)
    concat_tgt_idx = concat(video_ids, text_ids).to(torch.long)
    concat_src_idx = torch.argsort(concat_tgt_idx)

    lengths = (window_shape.prod(-1) + text_length).tolist()
    max_length = max(lengths)
    padded_flat_idx = []
    padded_gather_idx = []
    valid_keys = []
    offset = 0
    for window_index, length in enumerate(lengths):
        padded_flat_idx.extend(window_index * max_length + position for position in range(length))
        padded_gather_idx.extend(offset + position for position in range(length))
        padded_gather_idx.extend(0 for _ in range(max_length - length))
        valid_keys.append([True] * length + [False] * (max_length - length))
        offset += length
    if offset != concat_tgt_idx.numel():
        raise RuntimeError("Padded attention metadata does not match packed attention length.")

    txt_shape = torch.full((window_shape.shape[0], 1), text_length, device=device, dtype=torch.long)
    if attention.rope.mm:
        vid_freqs, txt_freqs = attention.rope.get_freqs(window_shape, txt_shape)
    else:
        # 7B uses pixel RoPE for video only. Keep the same matrix input ABI;
        # identity matrices leave the repeated text queries/keys unrotated.
        angles = attention.rope.get_freqs(window_shape)[..., ::2]
        cosine, sine = angles.cos(), angles.sin()
        vid_freqs = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
            angles.shape[0], angles.shape[1], 2, 2
        )
        txt_freqs = torch.eye(2, device=device).expand(
            window_shape.shape[0] * text_length, angles.shape[1], 2, 2
        )
    freqs = torch.cat([vid_freqs, txt_freqs], dim=0)
    freqs = torch.index_select(freqs, 0, concat_tgt_idx).to(
        device=device, dtype=dtype if attention.rope.mm and dtype != torch.bfloat16 else torch.float32
    )
    return WindowMetadata(
        window_tgt_idx=window_tgt_idx,
        window_src_idx=window_src_idx,
        window_shape=window_shape.to(device=device, dtype=torch.long),
        concat_tgt_idx=concat_tgt_idx,
        concat_src_idx=concat_src_idx,
        padded_flat_idx=torch.tensor(padded_flat_idx, device=device, dtype=torch.long),
        padded_gather_idx=torch.tensor(padded_gather_idx, device=device, dtype=torch.long),
        valid_keys=torch.tensor(valid_keys, device=device, dtype=torch.bool),
        freqs=freqs,
    )


def get_seedvr_metadata(
        diffusion_model: torch.nn.Module,
        latent_height: int,
        latent_width: int,
        text_length: int,
        device: torch.device,
        dtype: torch.dtype,
        temporal: int = 1,
) -> tuple[WindowMetadata, WindowMetadata]:
    """Return cached normal and shifted metadata for one latent resolution."""
    if latent_height < 2 or latent_width < 2 or latent_height % 2 or latent_width % 2:
        raise ValueError("SeedVR2 latent height and width must be even values of at least 2.")
    normal_attention, shifted_attention = validate_shared_metadata_config(diffusion_model)
    key = (temporal, latent_height, latent_width, text_length, str(device), dtype,
           _attention_signature(diffusion_model.blocks[0]))
    if key not in _METADATA_CACHE:
        _METADATA_CACHE[key] = (
            _generate_metadata_pack(normal_attention, latent_height, latent_width, text_length, device, dtype, temporal),
            _generate_metadata_pack(shifted_attention, latent_height, latent_width, text_length, device, dtype, temporal),
        )
    return _METADATA_CACHE[key]


def add_comfyui_to_path(comfy_root: str | None) -> Path:
    candidate = comfy_root or os.environ.get("COMFYUI_ROOT")
    if not candidate:
        raise RuntimeError("Pass --comfy-root or set COMFYUI_ROOT; this script does not install ComfyUI.")
    root = Path(candidate).expanduser().resolve()
    if not (root / "comfy" / "ldm" / "seedvr" / "model.py").is_file():
        raise FileNotFoundError(f"{root} does not contain ComfyUI's SeedVR2 implementation.")
    sys.path.insert(0, str(root))
    return root


class SeedVRVaeEncoderExportWrapper(torch.nn.Module):
    """Encode a normalized NCHW image to SeedVR's collapsed 16-channel latent."""

    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # The native VAE consumes raw [-1, 1] video in (B, C, T, H, W).
        # ComfyUI's VAE.encode then applies comfy_format_encoded.  Match that
        # latent scaling before passing the result to the diffusion model.
        from comfy.ldm.seedvr.constants import BYTEDANCE_VAE_SCALING_FACTOR
        return self.vae.encode(image.mul(2.0).sub(1.0).unsqueeze(2)).mul(BYTEDANCE_VAE_SCALING_FACTOR)


class SeedVRVaeDecoderExportWrapper(torch.nn.Module):
    """Decode a collapsed SeedVR latent to a normalized NCHW image."""

    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # The VAE returns raw (B, C, T, H, W).  This matches ComfyUI's VAE
        # output normalization, while leaving crop/postprocessing outside ONNX.
        image = self.vae.decode(latent).squeeze(2).add(1.0).div(2.0)
        # Clamp only after spatial tile blending in the CUDA pipeline.
        return image


class SeedVRVideoEncoderExportWrapper(torch.nn.Module):
    """BCTHW video with native causal temporal compression (4n+1 -> n+1)."""
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        from comfy.ldm.seedvr.constants import BYTEDANCE_VAE_SCALING_FACTOR
        # _encode bypasses the Python temporal slicing/cache loop; the full clip
        # follows the same causal convolution path with MemoryState.DISABLED.
        moments = self.vae._encode(image.mul(2.0).sub(1.0))
        return moments.chunk(2, dim=1)[0].mul(BYTEDANCE_VAE_SCALING_FACTOR)


class SeedVRVideoDecoderExportWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        from comfy.ldm.seedvr.constants import BYTEDANCE_VAE_SCALING_FACTOR
        return self.vae._decode(latent / BYTEDANCE_VAE_SCALING_FACTOR).add(1.0).div(2.0)


def _export_causal_conv_forward(self, input: torch.Tensor, memory_state=None, memory_cache=None) -> torch.Tensor:
    """Fold repeated temporal slices into a spatial kernel for one-frame inputs."""
    del memory_state, memory_cache
    padded_frames = 1 + self.temporal_padding * 2
    temporal_kernel_extent = 1 + (self.kernel_size[0] - 1) * self.dilation[0]
    temporal_outputs = 1 + (padded_frames - temporal_kernel_extent) // self.stride[0]
    if input.shape[2] == 1 and temporal_outputs == 1:
        # The causal extension is [frame, ..., frame], so every temporal
        # kernel slice sees the same frame. Precompute the spatial weight
        # before export to avoid materializing the repeated-frame activation.
        return torch.nn.functional.conv2d(
            input[:, :, 0].float(),
            self._folded_spatial_weight,
            self.bias.float() if self.bias is not None else None,
            self.stride[1:],
            self.padding[1:],
            self.dilation[1:],
            self.groups,
        ).to(input.dtype).unsqueeze(2)
    from comfy.ldm.seedvr.vae import extend_head

    input = extend_head(input, times=self.temporal_padding * 2)
    return torch.nn.functional.conv3d(
        input,
        self.weight,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


def _export_causal_norm_wrapper(norm_layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Use the normal VAE norm path without a symbolic memory-size branch."""
    input_dtype = x.dtype
    if isinstance(norm_layer, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
        if x.ndim == 4:
            return norm_layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).to(input_dtype)
        if x.ndim == 5:
            return norm_layer(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).to(input_dtype)
    if isinstance(norm_layer, (torch.nn.GroupNorm, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
        if x.ndim <= 4:
            return norm_layer(x).to(input_dtype)
        if x.ndim == 5:
            batch, channels, temporal, height, width = x.shape
            x = x.transpose(1, 2).reshape(batch * temporal, channels, height, width)
            x = norm_layer(x)
            return x.reshape(batch, temporal, x.size(1), x.size(2), x.size(3)).transpose(1, 2).to(input_dtype)
    raise TypeError(f"SeedVR2 VAE unsupported norm layer type: {type(norm_layer).__name__}")


def _export_scaled_dot_product_attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """Opt-in workaround for older ORT TensorRT RTX providers rejecting Attention."""
    if args or kwargs.get("attn_mask") is not None or kwargs.get("dropout_p", 0.0) != 0.0 or kwargs.get("is_causal", False):
        raise ValueError("The SeedVR2 VAE export expects unmasked, noncausal attention without dropout.")
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


@contextlib.contextmanager
def trt_rtx_vae_export_patches(vae: torch.nn.Module, *, decompose_attention: bool = False):
    """Keep ComfyUI unmodified while exporting the folded VAE graph."""
    import comfy.ldm.seedvr.vae as comfy_seedvr_vae
    import comfy.ops as comfy_ops

    original_causal_conv_forward = comfy_seedvr_vae.InflatedCausalConv3d.forward
    original_causal_norm_wrapper = comfy_seedvr_vae.causal_norm_wrapper
    original_scaled_dot_product_attention = comfy_ops.scaled_dot_product_attention
    folded_modules = []
    for module in vae.modules():
        if isinstance(module, comfy_seedvr_vae.InflatedCausalConv3d):
            if hasattr(module, "_folded_spatial_weight"):
                raise RuntimeError("Temporary folded convolution weight already exists.")
            module.register_buffer(
                "_folded_spatial_weight",
                module.weight.detach().float().sum(dim=2),
                persistent=False,
            )
            folded_modules.append(module)
    comfy_seedvr_vae.InflatedCausalConv3d.forward = _export_causal_conv_forward
    comfy_seedvr_vae.causal_norm_wrapper = _export_causal_norm_wrapper
    if decompose_attention:
        comfy_ops.scaled_dot_product_attention = _export_scaled_dot_product_attention
    try:
        yield
    finally:
        for module in folded_modules:
            delattr(module, "_folded_spatial_weight")
        if decompose_attention:
            comfy_ops.scaled_dot_product_attention = original_scaled_dot_product_attention
        comfy_seedvr_vae.causal_norm_wrapper = original_causal_norm_wrapper
        comfy_seedvr_vae.InflatedCausalConv3d.forward = original_causal_conv_forward


class SeedVRDiTExportAdapter(torch.nn.Module):
    """Tensor-only NaDiT execution path with external window metadata."""

    def __init__(self, diffusion_model: torch.nn.Module):
        super().__init__()
        self.diffusion_model = diffusion_model

    @staticmethod
    def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Apply ComfyUI's Flux-format partial RoPE using standard tensor ops."""
        rotary_dims = freqs.shape[-3] * 2
        rotary = x[..., :rotary_dims].transpose(0, 1)
        remainder = x[..., rotary_dims:]
        pairs = rotary.to(freqs.dtype).reshape(rotary.shape[0], rotary.shape[1], freqs.shape[-3], 1, 2)
        rotated = freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]
        rotated = rotated.reshape(rotary.shape).to(x.dtype).transpose(0, 1)
        return torch.cat((rotated, remainder), dim=-1)

    @staticmethod
    def _rms_norm(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
        if value.dtype == torch.bfloat16:
            return torch.nn.functional.rms_norm(value, module.normalized_shape, module.weight, module.eps)
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + module.eps)
        normalized = normalized.to(value.dtype)
        if module.weight is not None:
            normalized = normalized * module.weight.to(device=value.device, dtype=value.dtype)
        return normalized

    @classmethod
    def _mm_rms_norm(
            cls, module: torch.nn.Module, vid: torch.Tensor, txt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vid_module = module.all if module.shared_weights else module.vid
        vid = cls._rms_norm(vid_module, vid)
        if module.vid_only:
            return vid, txt
        txt_module = module.all if module.shared_weights else module.txt
        return vid, cls._rms_norm(txt_module, txt.to(device=vid.device, dtype=vid.dtype))

    @staticmethod
    def _padded_attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            padded_flat_idx: torch.Tensor,
            padded_gather_idx: torch.Tensor,
            valid_keys: torch.Tensor,
    ) -> torch.Tensor:
        """Exportable variable-length attention with CPU-generated padding metadata."""
        window_count, max_length = valid_keys.shape
        padded_shape = (window_count * max_length, q.shape[1], q.shape[2])
        q_padded = q.index_select(0, padded_gather_idx)
        k_padded = k.index_select(0, padded_gather_idx)
        v_padded = v.index_select(0, padded_gather_idx)
        batch_shape = (window_count, max_length, q.shape[1], q.shape[2])
        q_padded = q_padded.reshape(batch_shape).transpose(1, 2)
        k_padded = k_padded.reshape(batch_shape).transpose(1, 2)
        v_padded = v_padded.reshape(batch_shape).transpose(1, 2)

        # Preserve both the previous FP32 attention math and Dynamo's native
        # ONNX Attention translation; RTX can fuse the resulting operator.
        # Native PyTorch uses BF16 SDPA for the 7B BF16 path. Retain the
        # established FP32 attention computation for FP16 exports.
        attention_dtype = torch.bfloat16 if q.dtype == torch.bfloat16 else torch.float32
        output = torch.nn.functional.scaled_dot_product_attention(
            q_padded.to(attention_dtype), k_padded.to(attention_dtype), v_padded.to(attention_dtype),
            attn_mask=valid_keys[:, None, None, :], dropout_p=0.0,
        ).to(q.dtype)
        output = output.transpose(1, 2).reshape(padded_shape)
        return output.index_select(0, padded_flat_idx)

    def _attention(
            self,
            attention: torch.nn.Module,
            vid: torch.Tensor,
            txt: torch.Tensor,
            window_tgt_idx: torch.Tensor,
            window_src_idx: torch.Tensor,
            concat_tgt_idx: torch.Tensor,
            concat_src_idx: torch.Tensor,
            padded_flat_idx: torch.Tensor,
            padded_gather_idx: torch.Tensor,
            valid_keys: torch.Tensor,
            freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vid_qkv, txt_qkv = attention.proj_qkv(vid, txt)
        vid_qkv = torch.index_select(vid_qkv, 0, window_tgt_idx)
        vid_qkv = vid_qkv.reshape(vid_qkv.shape[0], 3, attention.heads, attention.head_dim)
        txt_qkv = txt_qkv.reshape(txt_qkv.shape[0], 3, attention.heads, attention.head_dim)
        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)
        vid_q, txt_q = self._mm_rms_norm(attention.norm_q, vid_q, txt_q)
        vid_k, txt_k = self._mm_rms_norm(attention.norm_k, vid_k, txt_k)

        window_count = valid_keys.shape[0]
        txt_q = txt_q.repeat(window_count, 1, 1)
        txt_k = txt_k.repeat(window_count, 1, 1)
        txt_v = txt_v.repeat(window_count, 1, 1)
        q = torch.index_select(torch.cat((vid_q, txt_q), dim=0), 0, concat_tgt_idx)
        k = torch.index_select(torch.cat((vid_k, txt_k), dim=0), 0, concat_tgt_idx)
        v = torch.index_select(torch.cat((vid_v, txt_v), dim=0), 0, concat_tgt_idx)
        q = self._apply_rope(q, freqs)
        k = self._apply_rope(k, freqs)

        output = self._padded_attention(q, k, v, padded_flat_idx, padded_gather_idx, valid_keys)
        output = torch.index_select(output, 0, concat_src_idx)
        video_tokens = window_tgt_idx.shape[0]
        vid_out = output[:video_tokens]
        txt_out = (
            output[video_tokens:].reshape(CONTEXT_TOKENS, window_count, attention.heads, attention.head_dim).mean(1)
        )
        vid_out = vid_out.flatten(1, 2)
        txt_out = txt_out.flatten(1, 2)
        vid_out = torch.index_select(vid_out, 0, window_src_idx)
        return attention.proj_out(vid_out, txt_out)

    def _block(
            self,
            block: torch.nn.Module,
            vid: torch.Tensor,
            txt: torch.Tensor,
            emb: torch.Tensor,
            metadata: tuple[torch.Tensor, ...],
            cache: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from comfy.ldm.seedvr.model import MMArg

        # Batch is fixed to one: AdaSingle's [1, channels] timestep values
        # broadcast over both token streams. Passing hid_len would instead
        # export repeat_interleave as CumSum and data-dependent Range ops.
        ada_kwargs = {
            "emb": emb,
            "cache": cache,
            "branch_tag": MMArg("vid", "txt"),
        }
        vid_attn, txt_attn = self._mm_rms_norm(block.attn_norm, vid, txt)
        vid_attn, txt_attn = block.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        vid_attn, txt_attn = self._attention(block.attn, vid_attn, txt_attn, *metadata)
        vid_attn, txt_attn = block.ada(vid_attn, txt_attn, layer="attn", mode="out", **ada_kwargs)
        vid_attn, txt_attn = vid_attn + vid, txt_attn + txt

        vid_mlp, txt_mlp = self._mm_rms_norm(block.mlp_norm, vid_attn, txt_attn)
        vid_mlp, txt_mlp = block.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        vid_mlp, txt_mlp = block.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = block.ada(vid_mlp, txt_mlp, layer="mlp", mode="out", **ada_kwargs)
        return vid_mlp + vid_attn, txt_mlp + txt_attn

    def forward(
            self,
            latent: torch.Tensor,
            timestep: torch.Tensor,
            context: torch.Tensor,
            condition: torch.Tensor,
            normal_window_tgt_idx: torch.Tensor,
            normal_window_src_idx: torch.Tensor,
            normal_concat_tgt_idx: torch.Tensor,
            normal_concat_src_idx: torch.Tensor,
            normal_padded_flat_idx: torch.Tensor,
            normal_padded_gather_idx: torch.Tensor,
            normal_valid_keys: torch.Tensor,
            normal_freqs: torch.Tensor,
            shifted_window_tgt_idx: torch.Tensor,
            shifted_window_src_idx: torch.Tensor,
            shifted_concat_tgt_idx: torch.Tensor,
            shifted_concat_src_idx: torch.Tensor,
            shifted_padded_flat_idx: torch.Tensor,
            shifted_padded_gather_idx: torch.Tensor,
            shifted_valid_keys: torch.Tensor,
            shifted_freqs: torch.Tensor,
    ) -> torch.Tensor:
        model = self.diffusion_model
        patch_t, patch_h, patch_w = model.vid_in.patch_size
        if patch_t != 1:
            raise RuntimeError("The SeedVR2 export expects temporal patch size one.")
        combined = torch.cat((latent, condition), dim=1).permute(0, 2, 3, 4, 1)
        combined = combined.reshape(
            1,
            latent.shape[2],
            latent.shape[3] // patch_h,
            patch_h,
            latent.shape[4] // patch_w,
            patch_w,
            combined.shape[-1],
        )
        combined = combined.permute(0, 1, 2, 4, 3, 5, 6)
        vid = combined.reshape(-1, patch_t * patch_h * patch_w * combined.shape[-1])
        vid = model.vid_in.proj(vid)
        txt = model.txt_in(context.reshape(CONTEXT_TOKENS, CONTEXT_WIDTH))
        emb = model.emb_in(timestep, device=vid.device, dtype=vid.dtype)

        normal = (
            normal_window_tgt_idx,
            normal_window_src_idx,
            normal_concat_tgt_idx,
            normal_concat_src_idx,
            normal_padded_flat_idx,
            normal_padded_gather_idx,
            normal_valid_keys,
            normal_freqs,
        )
        shifted = (
            shifted_window_tgt_idx,
            shifted_window_src_idx,
            shifted_concat_tgt_idx,
            shifted_concat_src_idx,
            shifted_padded_flat_idx,
            shifted_padded_gather_idx,
            shifted_valid_keys,
            shifted_freqs,
        )
        from comfy.ldm.seedvr.model import Cache

        cache = Cache(disable=False)
        normal_method = model.blocks[0].attn.window_method
        for block in model.blocks:
            metadata = normal if block.attn.window_method == normal_method else shifted
            vid, txt = self._block(block, vid, txt, emb, metadata, cache)

        if model.vid_out_norm:
            vid = self._rms_norm(model.vid_out_norm, vid)
            # Upstream's output Ada reuses emb_repeat_0_vid from the first
            # block, whose embedding is split across the attn/mlp layers.
            # Preserve that slice explicitly now that broadcasting skips
            # AdaSingle's repeat cache (the output Ada has only one layer).
            first_ada = model.blocks[0].ada
            first_ada = first_ada.all if first_ada.shared_weights else first_ada.vid
            output_emb = emb.reshape(emb.shape[0], -1, len(first_ada.layers), 3)[:, :, 0, :].flatten(1)
            vid = model.vid_out_ada(
                vid,
                emb=output_emb,
                layer="out",
                mode="in",
                cache=cache,
                branch_tag="vid",
            )
        vid = model.vid_out.proj(vid)
        output_channels = vid.shape[-1] // (patch_t * patch_h * patch_w)
        vid = vid.reshape(
            1,
            latent.shape[2],
            latent.shape[3] // patch_h,
            latent.shape[4] // patch_w,
            patch_t,
            patch_h,
            patch_w,
            output_channels,
        )
        return vid.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            1, output_channels, latent.shape[2], latent.shape[3], latent.shape[4]
        )


def load_diffusion_model(model_path: Path, dtype: torch.dtype | None = None) -> tuple[torch.nn.Module, torch.device, torch.dtype]:
    import comfy.model_management
    import comfy.sd

    if not model_path.is_file():
        raise FileNotFoundError(f"SeedVR diffusion model not found: {model_path}")
    patcher = comfy.sd.load_diffusion_model(str(model_path))
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    diffusion_model = patcher.model.diffusion_model.eval()
    if dtype is not None:
        # Native inference casts weights, but keeps FP32 frequency tables.
        fp32_buffers = {name: value for name, value in diffusion_model.named_buffers()
                        if value.dtype == torch.float32}
        diffusion_model.to(dtype=dtype)
        for name, value in fp32_buffers.items():
            parent, _, attribute = name.rpartition(".")
            setattr(diffusion_model.get_submodule(parent), attribute, value)
        diffusion_model.dtype = dtype
    parameter = next(diffusion_model.parameters())
    return diffusion_model, parameter.device, parameter.dtype


def load_vae_model(vae_path: Path) -> tuple[torch.nn.Module, torch.device, torch.dtype]:
    import comfy.model_management
    import comfy.sd
    import comfy.utils

    if not vae_path.is_file():
        raise FileNotFoundError(f"SeedVR VAE not found: {vae_path}")
    vae_state_dict, vae_metadata = comfy.utils.load_torch_file(str(vae_path), return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_state_dict, metadata=vae_metadata, dtype=torch.float16)
    vae.throw_exception_if_invalid()
    comfy.model_management.load_models_gpu([vae.patcher], force_full_load=True)
    vae_model = vae.first_stage_model.eval()
    parameter = next(vae_model.parameters())
    return vae_model, parameter.device, parameter.dtype


def dummy_inputs(
        device: torch.device, dtype: torch.dtype, temporal: int, height: int, width: int
) -> tuple[torch.Tensor, ...]:
    if temporal < 1:
        raise ValueError(f"Latent temporal length must be at least 1, got {temporal}.")
    if height < 2 or width < 2 or height % 2 or width % 2:
        raise ValueError("--latent-height and --latent-width must be even values of at least 2.")
    latent = torch.randn(1, LATENT_CHANNELS, temporal, height, width, device=device, dtype=dtype)
    timestep = torch.ones(1, device=device, dtype=dtype)
    context = torch.randn(1, CONTEXT_TOKENS, CONTEXT_WIDTH, device=device, dtype=dtype)
    condition = torch.randn(1, LATENT_CHANNELS + 1, temporal, height, width, device=device, dtype=dtype)
    return latent, timestep, context, condition


def dynamic_dit_shapes(max_video_frames: int | None = None) -> tuple[object, ...]:
    """Tie H/W and mark all resolution-dependent metadata lengths dynamic."""
    latent_height = 2 * torch.export.Dim("latent_height", min=1, max=2048)
    latent_width = 2 * torch.export.Dim("latent_width", min=1, max=2048)
    spatial = {3: latent_height, 4: latent_width}
    if max_video_frames is not None:
        spatial[2] = torch.export.Dim("latent_frames", min=2, max=(max_video_frames - 1) // 4 + 1)
    auto = torch.export.Dim.AUTO
    metadata_shapes = tuple(
        {0: auto, 1: auto} if name.endswith("_valid_keys") else {0: auto}
        for name in DIT_INPUT_NAMES[4:]
    )
    return (spatial, None, None, spatial, *metadata_shapes)


def dit_inputs_with_metadata(
        diffusion_model: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        temporal: int,
        height: int,
        width: int,
) -> tuple[torch.Tensor, ...]:
    inputs = dummy_inputs(device, dtype, temporal, height, width)
    normal, shifted = get_seedvr_metadata(diffusion_model, height, width, CONTEXT_TOKENS, device, dtype, temporal)
    return (*inputs, *normal.tensors(), *shifted.tensors())


@torch.no_grad()
def verify_torch_parity(
        diffusion_model: torch.nn.Module,
        adapter: SeedVRDiTExportAdapter,
        inputs: tuple[torch.Tensor, ...],
) -> None:
    latent, timestep, context, condition = inputs[:4]
    expected = diffusion_model(
        latent.clone(),
        timestep.clone(),
        context.clone(),
        disable_cache=False,
        condition=condition.clone(),
        transformer_options={"cond_or_uncond": [0]},
    )
    actual = adapter(*inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    maximum_error = (actual.float() - expected.float()).abs().max().item()
    print(f"PyTorch parity passed (maximum absolute error {maximum_error:.6g}).")


def validate_metadata_shapes(diffusion_model: torch.nn.Module, device: torch.device, dtype: torch.dtype,
                             temporal: int = 1) -> None:
    observations: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
    for height, width in ((64, 96), (112, 112), (112, 160)):
        normal, shifted = get_seedvr_metadata(diffusion_model, height, width, CONTEXT_TOKENS, device, dtype, temporal)
        expected_tokens = temporal * (height // 2) * (width // 2)
        for pack in (normal, shifted):
            if pack.window_tgt_idx.numel() != expected_tokens:
                raise RuntimeError(f"Metadata for {height}x{width} has an invalid video token count.")
            if pack.window_src_idx.numel() != expected_tokens:
                raise RuntimeError(f"Metadata for {height}x{width} has an invalid inverse token count.")
            if pack.concat_tgt_idx.numel() != pack.freqs.shape[0]:
                raise RuntimeError(f"Metadata for {height}x{width} has mismatched attention/frequency lengths.")
            if pack.padded_flat_idx.numel() != pack.concat_tgt_idx.numel():
                raise RuntimeError(f"Metadata for {height}x{width} has mismatched padded attention length.")
        observations.append(
            (
                height,
                width,
                (normal.window_shape.shape[0], normal.concat_tgt_idx.shape[0]),
                (shifted.window_shape.shape[0], shifted.concat_tgt_idx.shape[0]),
            )
        )
    if len({item[2:] for item in observations}) == 1:
        raise RuntimeError("Metadata window and attention lengths did not vary across test shapes.")
    for height, width, normal_counts, shifted_counts in observations:
        print(
            f"Metadata {height}x{width}: normal windows/tokens={normal_counts}, shifted windows/tokens={shifted_counts}"
        )


def export_onnx(
        model: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output_path: Path,
        artifacts_dir: Path,
        input_names: tuple[str, ...],
        output_names: tuple[str, ...],
        dynamic_shapes: object | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        dynamo: bool = True,
        opset_version: int = 24,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamo=dynamo,
        external_data=True,
        dynamic_shapes=dynamic_shapes,
        dynamic_axes=dynamic_axes,
        # The export errors are already sufficiently descriptive.  Suppressing
        # the Markdown report avoids a secondary Windows console encoding error
        # when torch emits symbols outside the active code page.
        report=False,
        artifacts_dir=str(artifacts_dir),
    )


def add_fp16_dit_boundaries(output_path: Path) -> None:
    """Keep native DiT computation, with FP16 I/O for the C++ CUDA pipeline."""
    import onnx

    model = onnx.load(str(output_path), load_external_data=False)
    renamed = {}
    before, after = [], []
    for value in model.graph.input:
        if value.name not in DIT_INPUT_NAMES[:4] or value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            continue
        native_name = "__seedvr_native_" + value.name
        renamed[value.name] = native_name
        before.append(onnx.helper.make_node("Cast", [value.name], [native_name],
                                           name=native_name, to=value.type.tensor_type.elem_type))
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
    for value in model.graph.output:
        if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            continue
        native_name = "__seedvr_native_" + value.name
        renamed[value.name] = native_name
        after.append(onnx.helper.make_node("Cast", [native_name], [value.name],
                                          name=value.name + "_fp16", to=onnx.TensorProto.FLOAT16))
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
    if not renamed:
        return
    for node in model.graph.node:
        for names in (node.input, node.output):
            for index, name in enumerate(names):
                names[index] = renamed.get(name, name)
    for value in model.graph.value_info:
        value.name = renamed.get(value.name, value.name)
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend((*before, *nodes, *after))
    onnx.save_model(model, str(output_path))


def name_dynamic_dit_dimensions(output_path: Path) -> None:
    """Give ORT's free-dimension overrides names stable across PyTorch exports."""
    import onnx

    model = onnx.load(str(output_path), load_external_data=False)
    names = {}
    for value in model.graph.input:
        for axis, dimension in enumerate(value.type.tensor_type.shape.dim):
            if dimension.dim_param:
                names.setdefault(dimension.dim_param, f"{value.name}_dim{axis}")
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        for dimension in value.type.tensor_type.shape.dim:
            if dimension.dim_param in names:
                dimension.dim_param = names[dimension.dim_param]
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    onnx.save_model(model, str(temporary))
    temporary.replace(output_path)


def capture_and_validate_exported_program(
        adapter: SeedVRDiTExportAdapter,
        diffusion_model: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        device: torch.device,
        dtype: torch.dtype,
        max_video_frames: int | None = None,
) -> torch.export.ExportedProgram:
    if (inputs[DIT_INPUT_NAMES.index("normal_valid_keys")].shape[0] < 2 or
            inputs[DIT_INPUT_NAMES.index("shifted_valid_keys")].shape[0] < 2):
        raise ValueError(
            "The DiT capture example must produce multiple normal and shifted windows; "
            "use the default 64x96 latent or a larger shape."
        )
    exported = torch.export.export(adapter, inputs, dynamic_shapes=dynamic_dit_shapes(max_video_frames), strict=False)
    exported_module = exported.module()
    example_height, example_width = inputs[0].shape[3:5]
    cases = [(inputs[0].shape[2], example_height, example_width),
             (inputs[0].shape[2], example_height, example_width + 16)]
    if max_video_frames is not None:
        cases.extend((t, example_height, example_width) for t in (2, 3))
    for temporal, height, width in cases:
        shape_inputs = dit_inputs_with_metadata(diffusion_model, device, dtype, temporal, height, width)
        with torch.no_grad():
            expected = adapter(*shape_inputs)
            actual = exported_module(*shape_inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print(f"Exported PyTorch program passed at latent T={temporal}, {height}x{width}.")
    return exported


def export_vae_components(
        vae: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        image_height: int,
        image_width: int,
        encoder_output: Path,
        decoder_output: Path,
        artifacts_dir: Path,
        export_encoder: bool,
        export_decoder: bool,
        decompose_attention: bool = False,
        video_frames: int = 1,
        opset_version: int = 24,
        max_video_frames: int = VIDEO_FRAME_RANGE[1],
) -> None:
    if image_height < 16 or image_width < 16 or image_height % 16 or image_width % 16:
        raise ValueError("--image-height and --image-width must be multiples of 16 and at least 16.")
    generator = torch.Generator(device=device).manual_seed(0)
    image = torch.randn(1, 3, image_height, image_width, device=device, dtype=dtype, generator=generator)
    latent = torch.randn(
        1, LATENT_CHANNELS, image_height // 8, image_width // 8,
        device=device, dtype=dtype, generator=generator,
    )
    encoder = SeedVRVaeEncoderExportWrapper(vae).eval()
    decoder = SeedVRVaeDecoderExportWrapper(vae).eval()
    spatial_axis = 2
    if video_frames > 1:
        spatial_axis = 3
        image = torch.randn(1, 3, video_frames, image_height, image_width, device=device, dtype=dtype, generator=generator)
        latent = torch.randn(1, LATENT_CHANNELS, (video_frames - 1) // 4 + 1,
                             image_height // 8, image_width // 8, device=device, dtype=dtype, generator=generator)
        encoder = SeedVRVideoEncoderExportWrapper(vae).eval()
        decoder = SeedVRVideoDecoderExportWrapper(vae).eval()
    def export_component(wrapper, example, path, input_name, output_name, spatial_multiple):
        shapes = {spatial_axis: spatial_multiple * torch.export.Dim(input_name + "_height_units", min=1),
                  spatial_axis + 1: spatial_multiple * torch.export.Dim(input_name + "_width_units", min=1)}
        if video_frames > 1:
            latent_frames = torch.export.Dim("latent_frames", min=2, max=(max_video_frames - 1) // 4 + 1)
            shapes[2] = 4 * latent_frames - 3 if input_name == "image" else latent_frames
            captured = torch.export.export(wrapper, (example,), dynamic_shapes=(shapes,), strict=False)
            for frames in (5, 9):
                size = list(example.shape)
                size[2] = frames if input_name == "image" else (frames - 1) // 4 + 1
                value = torch.randn(*size, device=device, dtype=dtype)
                with torch.no_grad():
                    torch.testing.assert_close(captured.module()(value), wrapper(value), rtol=2e-3, atol=4e-3)
            export_onnx(captured, (), path, artifacts_dir / ("vae_encoder" if input_name == "image" else "vae_decoder"),
                        (input_name,), (output_name,), opset_version=opset_version)
            name_dynamic_dit_dimensions(path)
        else:
            export_onnx(wrapper, (example,), path, artifacts_dir / ("vae_encoder" if input_name == "image" else "vae_decoder"),
                        (input_name,), (output_name,), dynamic_shapes=(shapes,), opset_version=opset_version)
    with torch.no_grad():
        expected_encoded = encoder(image)
        expected_decoded = decoder(latent)
        if video_frames > 1:
            from comfy.ldm.seedvr.constants import BYTEDANCE_VAE_SCALING_FACTOR
            torch.testing.assert_close(expected_encoded, vae.encode(image * 2 - 1) * BYTEDANCE_VAE_SCALING_FACTOR,
                                       rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(expected_decoded, vae.decode(latent).add(1).div(2), rtol=2e-2, atol=2e-2)
        with trt_rtx_vae_export_patches(vae, decompose_attention=decompose_attention):
            actual_encoded = encoder(image)
            actual_decoded = decoder(latent)
    # Folding sums temporal weights before convolution and changes FP16 rounding.
    torch.testing.assert_close(actual_encoded, expected_encoded, rtol=2e-3, atol=4e-3)
    torch.testing.assert_close(actual_decoded, expected_decoded, rtol=2e-3, atol=4e-3)
    print("TensorRT RTX VAE export patch passed eager parity.")
    if export_encoder:
        with trt_rtx_vae_export_patches(vae, decompose_attention=decompose_attention):
            export_component(encoder, image, encoder_output, "image", "latent", 16)
        print(f"Exported dynamic SeedVR VAE encoder: {encoder_output}")
        validate_vae_onnx(encoder_output, "image", "latent", opset_version, video=video_frames > 1)
    if export_decoder:
        with trt_rtx_vae_export_patches(vae, decompose_attention=decompose_attention):
            export_component(decoder, latent, decoder_output, "latent", "image", 2)
        print(f"Exported dynamic SeedVR VAE decoder: {decoder_output}")
        validate_vae_onnx(decoder_output, "latent", "image", opset_version, video=video_frames > 1)


def validate_onnx(output_path: Path, expected_attention_count: int = 32, *, video: bool = False) -> None:
    import onnx

    # The 3B external-data model exceeds protobuf's 2 GiB in-memory checker
    # limit.  Passing the path lets ONNX validate the adjacent .data file.
    onnx.checker.check_model(str(output_path))
    model = onnx.load(str(output_path), load_external_data=False)
    if not any(item.domain == "" and item.version >= 23 for item in model.opset_import):
        raise RuntimeError(f"{output_path} requires ONNX opset 23 or newer for Attention.")
    if any(node.op_type == "CumSum" for node in model.graph.node):
        raise RuntimeError("The batch-one DiT export must broadcast Ada embeddings without CumSum/repeat_interleave.")
    attention_count = sum(node.domain in ("", "ai.onnx") and node.op_type == "Attention"
                          for node in model.graph.node)
    if attention_count != expected_attention_count:
        raise RuntimeError(f"Expected {expected_attention_count} native DiT Attention nodes, found {attention_count}.")
    tensor_inputs = {value.name: value for value in model.graph.input}
    if set(tensor_inputs) != set(DIT_INPUT_NAMES):
        raise RuntimeError(f"Expected inputs {DIT_INPUT_NAMES}, found {tuple(tensor_inputs)}.")
    for value in (tensor_inputs["latent"], tensor_inputs["condition"], *model.graph.output):
        dims = value.type.tensor_type.shape.dim
        if not all(dims[index].dim_param for index in (3, 4)):
            raise RuntimeError(f"{value.name} does not retain dynamic spatial dimensions.")
        if video and not dims[2].dim_param:
            raise RuntimeError(f"{value.name} does not retain a dynamic temporal dimension.")
        if not video and (dims[2].dim_param or dims[2].dim_value != 1):
            raise RuntimeError(f"{value.name} image temporal length must remain fixed at one.")
    for name in DIT_INPUT_NAMES[4:]:
        if not tensor_inputs[name].type.tensor_type.shape.dim[0].dim_param:
            raise RuntimeError(f"{name} does not retain its dynamic leading dimension.")
    output_dims = model.graph.output[0].type.tensor_type.shape.dim
    if not output_dims[3].dim_param or not output_dims[4].dim_param:
        raise RuntimeError("The denoised latent does not retain dynamic spatial dimensions.")
    nonstandard = sorted(
        {(node.domain, node.op_type) for node in model.graph.node if node.domain not in ("", "ai.onnx")}
    )
    aten = sorted({node.op_type for node in model.graph.node if "ATen" in node.op_type})
    sequences = sorted({node.op_type for node in model.graph.node if node.op_type.startswith("Sequence")})
    if nonstandard or aten or sequences:
        raise RuntimeError(
            f"ONNX graph contains unsupported nodes: domains={nonstandard}, ATen={aten}, sequences={sequences}"
        )
    print(f"ONNX structural validation passed: {output_path}")


def validate_vae_onnx(output_path: Path, input_name: str, output_name: str, opset_version: int = 24,
                      *, video: bool = False) -> None:
    import onnx

    model = onnx.load(str(output_path), load_external_data=False)
    if not any(item.domain == "" and item.version == opset_version for item in model.opset_import):
        raise RuntimeError(f"{output_path} is not an ONNX opset {opset_version} model.")
    onnx.checker.check_model(str(output_path))
    values = {value.name: value for value in (*model.graph.input, *model.graph.output)}
    for name in (input_name, output_name):
        if name not in values:
            raise RuntimeError(f"Expected {name!r} in {output_path}, found {list(values)}.")
        dims = values[name].type.tensor_type.shape.dim
        if video and (len(dims) != 5 or not dims[2].dim_param):
            raise RuntimeError(f"{name} must retain a dynamic video temporal dimension in {output_path}.")
        if not dims[-2].dim_param or not dims[-1].dim_param:
            raise RuntimeError(f"{name} does not retain dynamic spatial dimensions in {output_path}.")
    print(f"VAE ONNX structural validation passed: {output_path}")


def export_positive_context(diffusion_model: torch.nn.Module, output_path: Path) -> None:
    context = diffusion_model.positive_conditioning.detach().to(device="cpu", dtype=torch.float16).contiguous()
    if tuple(context.shape) != (CONTEXT_TOKENS, CONTEXT_WIDTH):
        raise RuntimeError(f"Expected positive conditioning [58, 5120], got {tuple(context.shape)}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(context.view(torch.uint16).numpy().tobytes())
    print(f"Exported SeedVR positive conditioning: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the ComfyUI SeedVR2 VAE and DiT to dynamic-H/W ONNX.")
    parser.add_argument("--comfy-root", default="out/seedvr/ComfyUI")
    parser.add_argument("--snapshot", type=Path, default=None,
                        help="Optional local checkpoint snapshot; otherwise download from Hugging Face's global cache.")
    parser.add_argument("--dit-dtype", choices=("bf16", "fp16"), default="bf16",
                        help="DiT weight/compute dtype (default: bf16). VAE remains fp16.")
    parser.add_argument("--native-dit-io", action="store_true",
                        help="Keep DiT I/O at its compute dtype; default FP16 boundaries support the C++ pipeline.")
    parser.add_argument(
        "--model", default=None,
        help="Hugging Face repository slug or local DiT checkpoint (default: Comfy-Org/SeedVR2)."
    )
    parser.add_argument("--variant", choices=("3b", "7b"), default=None,
                        help="Checkpoint variant (default: 3b; inferred from local checkpoint filenames).")
    parser.add_argument(
        "--vae", type=Path, default=None, help="Defaults to seedvr2_ema_vae_fp16.safetensors in --snapshot."
    )
    parser.add_argument(
        "--components", "--component",
        nargs="+",
        choices=("vae_encoder", "vae_decoder", "dit", "context"),
        default=("vae_encoder", "vae_decoder", "dit"),
        help="Components to export; VAE components can be exported independently of the DiT.",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="DiT output path; default directory: models/SeedVR2-<variant>[-video]-onnx.")
    parser.add_argument("--vae-encoder-output", type=Path, default=None)
    parser.add_argument("--vae-decoder-output", type=Path, default=None)
    parser.add_argument("--positive-context-output", type=Path, default=None)
    parser.add_argument("--decompose-vae-attention", action="store_true",
                        help="Export VAE attention as matmul/softmax for ONNX Runtime TensorRT RTX compatibility; default: native ONNX Attention.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("out/seedvr/export-artifacts"))
    parser.add_argument("--opset", type=int, default=24,
                        help="ONNX opset version for all models; must be at least 23 for Attention (default: 24).")
    parser.add_argument("--video-frames", type=int, default=1,
                        help="Video capture example of 4n+1 frames (>=5); all video exports have dynamic T/H/W. Default 1 exports images.")
    parser.add_argument("--latent-height", type=int, default=64, help="Even multi-window example latent height (default: 64, representing 512 pixels).")
    parser.add_argument("--latent-width", type=int, default=64, help="Even multi-window example latent width (default: 64, representing 512 pixels).")
    parser.add_argument(
        "--image-height", type=int, default=64, help="Example VAE image height; must be a multiple of 16 (default: 64)."
    )
    parser.add_argument(
        "--image-width", type=int, default=64, help="Example VAE image width; must be a multiple of 16 (default: 64)."
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--check-torch-parity", action="store_true",
                        help="Compare the export adapter against the native PyTorch model (off by default).")
    parser.add_argument("--skip-metadata-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.opset < 23:
        raise ValueError("--opset must be at least 23 for native ONNX Attention.")
    if args.video_frames < 1 or (args.video_frames - 1) % 4:
        raise ValueError("--video-frames must be 4n+1 (1, 5, 9, 13, ...).")
    args.temporal = (args.video_frames - 1) // 4 + 1
    video = args.video_frames > 1
    if video and not VIDEO_FRAME_RANGE[0] <= args.video_frames <= VIDEO_FRAME_RANGE[1]:
        raise ValueError("Video capture length must be 4n+1 in [5,33].")
    max_video_frames = VIDEO_FRAME_RANGE[1] if video else None
    add_comfyui_to_path(args.comfy_root)
    components = set(args.components)
    # A repository slug has exactly namespace/name; checkpoint files retain
    # their path semantics, including snapshot symlinks used by ComfyUI.
    model_source = args.model or DEFAULT_REPO_ID
    model_file = Path(model_source).expanduser()
    model_parts = model_source.split("/")
    from_repo = (not model_file.is_file() and model_file.suffix.lower() not in (".safetensors", ".pt", ".pth", ".ckpt", ".bin") and
                 len(model_parts) == 2 and all(model_parts) and "\\" not in model_source and ":" not in model_source)
    repo_id = model_source if from_repo else DEFAULT_REPO_ID
    explicit_model = None if from_repo else model_file
    local_variant = next((v for v in ("3b", "7b") if explicit_model and v in explicit_model.stem.lower()), None)
    if args.variant and local_variant and args.variant != local_variant:
        raise ValueError("--variant conflicts with the local checkpoint filename.")
    variant = args.variant or local_variant or "3b"
    checkpoint_name = f"seedvr2_{variant}_fp16.safetensors"
    snapshot = args.snapshot or Path()
    if args.snapshot is None and (({"dit", "context"} & components and explicit_model is None) or
                                  ({"vae_encoder", "vae_decoder"} & components and args.vae is None)):
        from huggingface_hub import snapshot_download

        patterns = []
        if {"dit", "context"} & components and explicit_model is None:
            patterns.append(f"diffusion_models/{checkpoint_name}")
        if {"vae_encoder", "vae_decoder"} & components and args.vae is None:
            patterns.append("vae/seedvr2_ema_vae_fp16.safetensors")
        snapshot = Path(snapshot_download(repo_id=repo_id, allow_patterns=patterns))
    model_path = explicit_model or snapshot / "diffusion_models" / checkpoint_name
    vae_path = args.vae or snapshot / "vae" / "seedvr2_ema_vae_fp16.safetensors"
    # Preserve the Hugging Face snapshot symlink suffix for ComfyUI's
    # safetensors loader; .resolve() would yield an extensionless blob path.
    model_path = model_path.expanduser().absolute()
    default_output = (f"models/SeedVR2-{variant.upper()}-video-onnx/seedvr2_video_dit.onnx" if video else
                      f"models/SeedVR2-{variant.upper()}-onnx/seedvr2_{variant}_{args.dit_dtype}_dit.onnx")
    output_path = (args.output or Path(default_output)).expanduser().absolute()
    artifacts_dir = args.artifacts_dir.expanduser().absolute()
    vae_encoder_output = (
        (args.vae_encoder_output or output_path.with_name("seedvr2_ema_vae_encoder.onnx")).expanduser().absolute()
    )
    vae_decoder_output = (
        (args.vae_decoder_output or output_path.with_name("seedvr2_ema_vae_decoder.onnx")).expanduser().absolute()
    )
    positive_context_output = (
        (args.positive_context_output or output_path.with_name(f"seedvr2_{variant}_fp16_positive_context.bin"))
        .expanduser()
        .absolute()
    )
    if args.video_frames > 1:
        output_path = output_path.with_name("seedvr2_video_dit.onnx")
        vae_encoder_output = vae_encoder_output.with_name("seedvr2_video_vae_encoder.onnx")
        vae_decoder_output = vae_decoder_output.with_name("seedvr2_video_vae_decoder.onnx")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if {"vae_encoder", "vae_decoder"} & components:
        print(f"Loading SeedVR VAE: {vae_path}")
        vae, device, dtype = load_vae_model(vae_path)
        print(f"VAE export device={device}, dtype={dtype}")
        export_vae_components(
            vae,
            device,
            dtype,
            args.image_height,
            args.image_width,
            vae_encoder_output,
            vae_decoder_output,
            artifacts_dir,
            export_encoder="vae_encoder" in components,
            export_decoder="vae_decoder" in components,
            decompose_attention=args.decompose_vae_attention,
            video_frames=args.video_frames,
            opset_version=args.opset,
            max_video_frames=VIDEO_FRAME_RANGE[1],
        )
    if {"dit", "context"} & components:
        print(f"Loading SeedVR diffusion model: {model_path}")
        diffusion_model, device, dtype = load_diffusion_model(
            model_path, torch.bfloat16 if args.dit_dtype == "bf16" else None
        )
        print(f"DiT export device={device}, dtype={dtype}")
        export_positive_context(diffusion_model, positive_context_output)

    if "dit" in components:
        adapter = SeedVRDiTExportAdapter(diffusion_model).eval()
        inputs = dit_inputs_with_metadata(
            diffusion_model, device, dtype, args.temporal, args.latent_height, args.latent_width
        )
        if not args.skip_metadata_validation:
            validate_metadata_shapes(diffusion_model, device, dtype, args.temporal)
        if args.check_torch_parity:
            verify_torch_parity(diffusion_model, adapter, inputs)
        captured = capture_and_validate_exported_program(adapter, diffusion_model, inputs, device, dtype, max_video_frames)
        export_onnx(
            captured if video else adapter,
            () if video else inputs,
            output_path,
            artifacts_dir / "dit",
            DIT_INPUT_NAMES,
            ("denoised_latent",),
            None if video else dynamic_dit_shapes(),
            opset_version=args.opset,
        )
        if not args.native_dit_io:
            add_fp16_dit_boundaries(output_path)
        name_dynamic_dit_dimensions(output_path)
        print(f"Exported dynamic SeedVR DiT: {output_path}")
        if not args.skip_validation:
            validate_onnx(output_path, len(diffusion_model.blocks), video=video)


if __name__ == "__main__":
    main()
