# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# !/usr/bin/env python3
"""Export ComfyUI's checkpoint-backed SeedVR2 VAE and DiT to ONNX.

The VAE encoder/decoder are separate ONNX components with dynamic image
height/width. The DiT is exported as a third component with matching dynamic
latent height/width. Batch, temporal length (one frame), and the 58x5120
conditioning tensor are fixed.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# Torch's ONNX progress messages contain Unicode status glyphs.  The native
# Windows console can still default to cp1252, which otherwise obscures the
# actual export error with a second UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_SNAPSHOT = Path("D:/hf/hub/models--Comfy-Org--SeedVR2/snapshots/673340c8a66db62b84e4099def7d01d337ae12dc")
CONTEXT_TOKENS = 58
CONTEXT_WIDTH = 5120
LATENT_CHANNELS = 16
DIT_INPUT_NAMES = (
    "latent",
    "timestep",
    "context",
    "condition",
    "normal_window_tgt_idx",
    "normal_window_shape",
    "normal_concat_tgt_idx",
    "normal_concat_src_idx",
    "normal_freqs",
    "shifted_window_tgt_idx",
    "shifted_window_shape",
    "shifted_concat_tgt_idx",
    "shifted_concat_src_idx",
    "shifted_freqs",
)


@dataclass(frozen=True)
class WindowMetadata:
    """Resolution-dependent tensors consumed by one DiT window configuration."""

    window_tgt_idx: torch.Tensor
    window_shape: torch.Tensor
    concat_tgt_idx: torch.Tensor
    concat_src_idx: torch.Tensor
    freqs: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.window_tgt_idx,
            self.window_shape,
            self.concat_tgt_idx,
            self.concat_src_idx,
            self.freqs,
        )


_METADATA_CACHE: dict[tuple[int, int, int, str, torch.dtype], tuple[WindowMetadata, WindowMetadata]] = {}


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
    if getattr(diffusion_model, "_7b_version", False):
        raise ValueError("The dynamic metadata exporter currently supports only SeedVR2 3B.")
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
) -> WindowMetadata:
    import comfy.ldm.seedvr.model as seedvr_model

    patch_height = latent_height // 2
    patch_width = latent_width // 2
    vid_shape = torch.tensor([[1, patch_height, patch_width]], device=device, dtype=torch.long)

    def make_window(x: torch.Tensor) -> list[torch.Tensor]:
        temporal, height, width, _ = x.shape
        slices = attention.window_op((temporal, height, width), attention.window)
        return [x[st, sh, sw] for st, sh, sw in slices]

    partition, _, window_shape, window_count, _, _ = seedvr_model.window_idx(vid_shape, make_window)
    patched_tokens = patch_height * patch_width
    window_tgt_idx = partition(torch.arange(patched_tokens, device=device)).to(torch.long)

    vid_len = window_shape.prod(-1)
    txt_len = torch.tensor([text_length], device=device, dtype=torch.long)
    concat, _ = seedvr_model.repeat_concat_idx(vid_len, txt_len, window_count)
    video_ids = torch.arange(patched_tokens, device=device)
    text_ids = torch.arange(patched_tokens, patched_tokens + text_length, device=device)
    concat_tgt_idx = concat(video_ids, text_ids).to(torch.long)
    concat_src_idx = torch.argsort(concat_tgt_idx)

    txt_shape = torch.full((window_shape.shape[0], 1), text_length, device=device, dtype=torch.long)
    vid_freqs, txt_freqs = attention.rope.get_freqs(window_shape, txt_shape)
    freqs = torch.cat([vid_freqs, txt_freqs], dim=0)
    freqs = torch.index_select(freqs, 0, concat_tgt_idx).to(device=device, dtype=dtype)
    return WindowMetadata(
        window_tgt_idx=window_tgt_idx,
        window_shape=window_shape.to(device=device, dtype=torch.long),
        concat_tgt_idx=concat_tgt_idx,
        concat_src_idx=concat_src_idx,
        freqs=freqs,
    )


def get_seedvr_metadata(
        diffusion_model: torch.nn.Module,
        latent_height: int,
        latent_width: int,
        text_length: int,
        device: torch.device,
        dtype: torch.dtype,
) -> tuple[WindowMetadata, WindowMetadata]:
    """Return cached normal and shifted metadata for one latent resolution."""
    if latent_height < 2 or latent_width < 2 or latent_height % 2 or latent_width % 2:
        raise ValueError("SeedVR2 latent height and width must be even values of at least 2.")
    normal_attention, shifted_attention = validate_shared_metadata_config(diffusion_model)
    key = (latent_height, latent_width, text_length, str(device), dtype)
    if key not in _METADATA_CACHE:
        _METADATA_CACHE[key] = (
            _generate_metadata_pack(normal_attention, latent_height, latent_width, text_length, device, dtype),
            _generate_metadata_pack(shifted_attention, latent_height, latent_width, text_length, device, dtype),
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
        return self.vae.encode(image.mul(2.0).sub(1.0).unsqueeze(2))


class SeedVRVaeDecoderExportWrapper(torch.nn.Module):
    """Decode a collapsed SeedVR latent to a normalized NCHW image."""

    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # The VAE returns raw (B, C, T, H, W).  This matches ComfyUI's VAE
        # output normalization, while leaving crop/postprocessing outside ONNX.
        return self.vae.decode(latent).squeeze(2).add(1.0).div(2.0).clamp(0.0, 1.0)


def _export_extend_head(tensor: torch.Tensor, times: int = 2, memory=None) -> torch.Tensor:
    """Equivalent single-call VAE padding without a rank-dependent Tile shape."""
    if memory is not None:
        return torch.cat((memory.to(tensor), tensor), dim=2)
    if times < 0:
        raise ValueError(f"SeedVR2 VAE extend_head expected times >= 0, got {times}.")
    if times == 0:
        return tensor
    return torch.cat((*([tensor[:, :, :1]] * times), tensor), dim=2)


def _export_causal_conv_forward(self, input: torch.Tensor, memory_state=None, memory_cache=None) -> torch.Tensor:
    """One-frame causal convolution expressed as a native padded Conv."""
    del memory_state, memory_cache
    input = _export_extend_head(input, times=self.temporal_padding * 2)
    return torch.nn.functional.conv3d(
        input,
        self.weight,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


@contextlib.contextmanager
def trt_rtx_vae_export_patches():
    """Keep ComfyUI unmodified while emitting a TensorRT-friendly VAE graph."""
    import comfy.ldm.seedvr.vae as comfy_seedvr_vae

    original_extend_head = comfy_seedvr_vae.extend_head
    original_causal_conv_forward = comfy_seedvr_vae.InflatedCausalConv3d.forward
    comfy_seedvr_vae.extend_head = _export_extend_head
    comfy_seedvr_vae.InflatedCausalConv3d.forward = _export_causal_conv_forward
    try:
        yield
    finally:
        comfy_seedvr_vae.InflatedCausalConv3d.forward = original_causal_conv_forward
        comfy_seedvr_vae.extend_head = original_extend_head


class SeedVRDiTExportAdapter(torch.nn.Module):
    """Tensor-only 3B NaDiT execution path with external window metadata."""

    def __init__(self, diffusion_model: torch.nn.Module):
        super().__init__()
        if getattr(diffusion_model, "_7b_version", False):
            raise ValueError("The ONNX execution adapter currently supports only SeedVR2 3B.")
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
            lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Exportable variable-length attention via padded batched matmuls."""
        cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        total = q.shape[0]
        sequence_starts = torch.zeros(total, device=q.device, dtype=torch.long)
        sequence_starts = sequence_starts.index_fill(0, cumulative[1:-1], 1)
        sequence_ids = sequence_starts.cumsum(0)
        positions = torch.arange(total, device=q.device) - cumulative.index_select(0, sequence_ids)
        max_length = lengths.max()
        flat_indices = sequence_ids * max_length + positions

        padded_shape = (lengths.shape[0] * max_length, q.shape[1], q.shape[2])
        q_padded = q.new_zeros(padded_shape).index_copy(0, flat_indices, q)
        k_padded = k.new_zeros(padded_shape).index_copy(0, flat_indices, k)
        v_padded = v.new_zeros(padded_shape).index_copy(0, flat_indices, v)
        batch_shape = (lengths.shape[0], max_length, q.shape[1], q.shape[2])
        q_padded = q_padded.reshape(batch_shape).transpose(1, 2)
        k_padded = k_padded.reshape(batch_shape).transpose(1, 2)
        v_padded = v_padded.reshape(batch_shape).transpose(1, 2)

        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q_padded.float(), k_padded.float().transpose(-2, -1)) * scale
        valid_keys = torch.arange(max_length, device=q.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~valid_keys[:, None, None, :], torch.finfo(scores.dtype).min)
        output = torch.matmul(torch.softmax(scores, dim=-1), v_padded.float()).to(q.dtype)
        output = output.transpose(1, 2).reshape(padded_shape)
        return output.index_select(0, flat_indices)

    def _attention(
            self,
            attention: torch.nn.Module,
            vid: torch.Tensor,
            txt: torch.Tensor,
            window_tgt_idx: torch.Tensor,
            window_shape: torch.Tensor,
            concat_tgt_idx: torch.Tensor,
            concat_src_idx: torch.Tensor,
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

        window_count = window_shape.shape[0]
        txt_q = txt_q.repeat(window_count, 1, 1)
        txt_k = txt_k.repeat(window_count, 1, 1)
        txt_v = txt_v.repeat(window_count, 1, 1)
        q = torch.index_select(torch.cat((vid_q, txt_q), dim=0), 0, concat_tgt_idx)
        k = torch.index_select(torch.cat((vid_k, txt_k), dim=0), 0, concat_tgt_idx)
        v = torch.index_select(torch.cat((vid_v, txt_v), dim=0), 0, concat_tgt_idx)
        q = self._apply_rope(q, freqs)
        k = self._apply_rope(k, freqs)

        lengths = window_shape.prod(-1) + CONTEXT_TOKENS
        output = self._padded_attention(q, k, v, lengths)
        output = torch.index_select(output, 0, concat_src_idx)
        video_tokens = window_tgt_idx.shape[0]
        vid_out = output[:video_tokens]
        txt_out = (
            output[video_tokens:].reshape(CONTEXT_TOKENS, window_count, attention.heads, attention.head_dim).mean(1)
        )
        vid_out = vid_out.flatten(1, 2)
        txt_out = txt_out.flatten(1, 2)
        vid_out = torch.index_select(vid_out, 0, torch.argsort(window_tgt_idx))
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

        hid_len = MMArg(
            torch.ones(1, device=vid.device, dtype=torch.long) * vid.shape[0],
            torch.full((1,), CONTEXT_TOKENS, device=txt.device, dtype=torch.long),
        )
        ada_kwargs = {
            "emb": emb,
            "hid_len": hid_len,
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
            normal_window_shape: torch.Tensor,
            normal_concat_tgt_idx: torch.Tensor,
            normal_concat_src_idx: torch.Tensor,
            normal_freqs: torch.Tensor,
            shifted_window_tgt_idx: torch.Tensor,
            shifted_window_shape: torch.Tensor,
            shifted_concat_tgt_idx: torch.Tensor,
            shifted_concat_src_idx: torch.Tensor,
            shifted_freqs: torch.Tensor,
    ) -> torch.Tensor:
        model = self.diffusion_model
        patch_t, patch_h, patch_w = model.vid_in.patch_size
        if patch_t != 1:
            raise RuntimeError("The SeedVR2 3B export expects temporal patch size one.")
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
            normal_window_shape,
            normal_concat_tgt_idx,
            normal_concat_src_idx,
            normal_freqs,
        )
        shifted = (
            shifted_window_tgt_idx,
            shifted_window_shape,
            shifted_concat_tgt_idx,
            shifted_concat_src_idx,
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
            vid = model.vid_out_ada(
                vid,
                emb=emb,
                layer="out",
                mode="in",
                hid_len=torch.ones(1, device=vid.device, dtype=torch.long) * vid.shape[0],
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


def load_diffusion_model(model_path: Path) -> tuple[torch.nn.Module, torch.device, torch.dtype]:
    import comfy.model_management
    import comfy.sd

    if not model_path.is_file():
        raise FileNotFoundError(f"SeedVR diffusion model not found: {model_path}")
    patcher = comfy.sd.load_diffusion_model(str(model_path))
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    diffusion_model = patcher.model.diffusion_model.eval()
    parameter = next(diffusion_model.parameters())
    return diffusion_model, parameter.device, parameter.dtype


def load_vae_model(vae_path: Path) -> tuple[torch.nn.Module, torch.device, torch.dtype]:
    import comfy.model_management
    import comfy.sd
    import comfy.utils

    if not vae_path.is_file():
        raise FileNotFoundError(f"SeedVR VAE not found: {vae_path}")
    vae_state_dict, vae_metadata = comfy.utils.load_torch_file(str(vae_path), return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_state_dict, metadata=vae_metadata)
    vae.throw_exception_if_invalid()
    comfy.model_management.load_models_gpu([vae.patcher], force_full_load=True)
    vae_model = vae.first_stage_model.eval()
    parameter = next(vae_model.parameters())
    return vae_model, parameter.device, parameter.dtype


def dummy_inputs(
        device: torch.device, dtype: torch.dtype, temporal: int, height: int, width: int
) -> tuple[torch.Tensor, ...]:
    if temporal < 1:
        raise ValueError(f"--temporal must be at least 1, got {temporal}.")
    if height < 2 or width < 2 or height % 2 or width % 2:
        raise ValueError("--latent-height and --latent-width must be even values of at least 2.")
    latent = torch.randn(1, LATENT_CHANNELS, temporal, height, width, device=device, dtype=dtype)
    timestep = torch.ones(1, device=device, dtype=dtype)
    context = torch.randn(1, CONTEXT_TOKENS, CONTEXT_WIDTH, device=device, dtype=dtype)
    condition = torch.randn(1, LATENT_CHANNELS + 1, temporal, height, width, device=device, dtype=dtype)
    return latent, timestep, context, condition


def dynamic_dit_shapes() -> tuple[object, ...]:
    """Tie H/W and mark all resolution-dependent metadata lengths dynamic."""
    latent_height = 2 * torch.export.Dim("latent_height", min=1, max=2048)
    latent_width = 2 * torch.export.Dim("latent_width", min=1, max=2048)
    spatial = {3: latent_height, 4: latent_width}
    auto = torch.export.Dim.AUTO
    return (
        spatial,
        None,
        None,
        spatial,
        {0: auto},
        {0: auto},
        {0: auto},
        {0: auto},
        {0: auto},
        {0: auto},
        {0: auto},
        {0: auto},
        {0: auto},
        {0: auto},
    )


def dit_inputs_with_metadata(
        diffusion_model: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        temporal: int,
        height: int,
        width: int,
) -> tuple[torch.Tensor, ...]:
    inputs = dummy_inputs(device, dtype, temporal, height, width)
    normal, shifted = get_seedvr_metadata(diffusion_model, height, width, CONTEXT_TOKENS, device, dtype)
    return (*inputs, *normal.tensors(), *shifted.tensors())


@torch.no_grad()
def verify_eager_adapter(
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
    print(f"Eager adapter parity passed (maximum absolute error {maximum_error:.6g}).")


def validate_metadata_shapes(diffusion_model: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> None:
    observations: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
    for height, width in ((64, 96), (112, 112), (112, 160)):
        normal, shifted = get_seedvr_metadata(diffusion_model, height, width, CONTEXT_TOKENS, device, dtype)
        expected_tokens = (height // 2) * (width // 2)
        for pack in (normal, shifted):
            if pack.window_tgt_idx.numel() != expected_tokens:
                raise RuntimeError(f"Metadata for {height}x{width} has an invalid video token count.")
            if pack.concat_tgt_idx.numel() != pack.freqs.shape[0]:
                raise RuntimeError(f"Metadata for {height}x{width} has mismatched attention/frequency lengths.")
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
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=22 if dynamo else 20,
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


def capture_and_validate_exported_program(
        adapter: SeedVRDiTExportAdapter,
        diffusion_model: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        device: torch.device,
        dtype: torch.dtype,
) -> torch.export.ExportedProgram:
    if inputs[5].shape[0] < 2 or inputs[10].shape[0] < 2:
        raise ValueError(
            "The DiT capture example must produce multiple normal and shifted windows; "
            "use the default 64x96 latent or a larger shape."
        )
    exported = torch.export.export(adapter, inputs, dynamic_shapes=dynamic_dit_shapes(), strict=False)
    exported_module = exported.module()
    example_height, example_width = inputs[0].shape[3:5]
    for height, width in (
            (example_height, example_width),
            (example_height, example_width + 16),
    ):
        shape_inputs = dit_inputs_with_metadata(diffusion_model, device, dtype, 1, height, width)
        with torch.no_grad():
            expected = adapter(*shape_inputs)
            actual = exported_module(*shape_inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print(f"Exported PyTorch program passed at latent {height}x{width}.")
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
) -> None:
    if image_height < 16 or image_width < 16 or image_height % 16 or image_width % 16:
        raise ValueError("--image-height and --image-width must be multiples of 16 and at least 16.")
    image = torch.randn(1, 3, image_height, image_width, device=device, dtype=dtype)
    latent = torch.randn(1, LATENT_CHANNELS, image_height // 8, image_width // 8, device=device, dtype=dtype)
    encoder = SeedVRVaeEncoderExportWrapper(vae).eval()
    decoder = SeedVRVaeDecoderExportWrapper(vae).eval()
    with torch.no_grad():
        expected_encoded = encoder(image)
        expected_decoded = decoder(latent)
        with trt_rtx_vae_export_patches():
            actual_encoded = encoder(image)
            actual_decoded = decoder(latent)
    torch.testing.assert_close(actual_encoded, expected_encoded, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual_decoded, expected_decoded, rtol=2e-3, atol=2e-3)
    print("TensorRT RTX VAE export patch passed eager parity.")
    if export_encoder:
        with trt_rtx_vae_export_patches():
            export_onnx(
                encoder,
                (image,),
                encoder_output,
                artifacts_dir / "vae_encoder",
                ("image",),
                ("latent",),
                dynamic_axes={
                    "image": {2: "image_height", 3: "image_width"},
                    "latent": {2: "latent_height", 3: "latent_width"},
                },
                dynamo=False,
            )
        print(f"Exported dynamic SeedVR VAE encoder: {encoder_output}")
        validate_vae_onnx(encoder_output, "image", "latent")
    if export_decoder:
        with trt_rtx_vae_export_patches():
            export_onnx(
                decoder,
                (latent,),
                decoder_output,
                artifacts_dir / "vae_decoder",
                ("latent",),
                ("image",),
                dynamic_axes={
                    "latent": {2: "latent_height", 3: "latent_width"},
                    "image": {2: "image_height", 3: "image_width"},
                },
                dynamo=False,
            )
        print(f"Exported dynamic SeedVR VAE decoder: {decoder_output}")
        validate_vae_onnx(decoder_output, "latent", "image")


def validate_onnx(output_path: Path) -> None:
    import onnx

    # The 3B external-data model exceeds protobuf's 2 GiB in-memory checker
    # limit.  Passing the path lets ONNX validate the adjacent .data file.
    onnx.checker.check_model(str(output_path))
    model = onnx.load(str(output_path), load_external_data=False)
    tensor_inputs = {value.name: value for value in model.graph.input}
    if set(tensor_inputs) != set(DIT_INPUT_NAMES):
        raise RuntimeError(f"Expected inputs {DIT_INPUT_NAMES}, found {tuple(tensor_inputs)}.")
    for value in (tensor_inputs["latent"], tensor_inputs["condition"]):
        dims = value.type.tensor_type.shape.dim
        if not all(dims[index].dim_param for index in (3, 4)):
            raise RuntimeError(f"{value.name} does not retain dynamic spatial dimensions.")
        if dims[2].dim_param or dims[2].dim_value != 1:
            raise RuntimeError(f"{value.name} temporal length must remain fixed at one.")
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


def validate_vae_onnx(output_path: Path, input_name: str, output_name: str) -> None:
    import onnx

    model = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(model)
    values = {value.name: value for value in (*model.graph.input, *model.graph.output)}
    for name in (input_name, output_name):
        if name not in values:
            raise RuntimeError(f"Expected {name!r} in {output_path}, found {list(values)}.")
        dims = values[name].type.tensor_type.shape.dim
        if not dims[2].dim_param or not dims[3].dim_param:
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
    parser.add_argument("--comfy-root", default="D:/repos/ComfyUI")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--model", type=Path, default=None, help="Defaults to seedvr2_3b_fp16.safetensors in --snapshot."
    )
    parser.add_argument(
        "--vae", type=Path, default=None, help="Defaults to seedvr2_ema_vae_fp16.safetensors in --snapshot."
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("vae_encoder", "vae_decoder", "dit", "context"),
        default=("vae_encoder", "vae_decoder", "dit"),
        help="Components to export; VAE components can be exported independently of the DiT.",
    )
    parser.add_argument("--output", type=Path, default=Path("out/seedvr/onnx/seedvr2_3b_fp16_dit.onnx"))
    parser.add_argument("--vae-encoder-output", type=Path, default=None)
    parser.add_argument("--vae-decoder-output", type=Path, default=None)
    parser.add_argument("--positive-context-output", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("out/seedvr/onnx/export-artifacts"))
    parser.add_argument("--temporal", type=int, default=1, help="Example latent temporal length.")
    parser.add_argument("--latent-height", type=int, default=64, help="Even multi-window example latent height.")
    parser.add_argument("--latent-width", type=int, default=96, help="Even multi-window example latent width.")
    parser.add_argument(
        "--image-height", type=int, default=32, help="Example VAE image height; must be a multiple of 16."
    )
    parser.add_argument(
        "--image-width", type=int, default=32, help="Example VAE image width; must be a multiple of 16."
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-eager-parity", action="store_true")
    parser.add_argument("--skip-metadata-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_comfyui_to_path(args.comfy_root)
    model_path = args.model or args.snapshot / "diffusion_models" / "seedvr2_3b_fp16.safetensors"
    vae_path = args.vae or args.snapshot / "vae" / "seedvr2_ema_vae_fp16.safetensors"
    # Preserve the Hugging Face snapshot symlink suffix for ComfyUI's
    # safetensors loader; .resolve() would yield an extensionless blob path.
    model_path = model_path.expanduser().absolute()
    output_path = args.output.expanduser().absolute()
    artifacts_dir = args.artifacts_dir.expanduser().absolute()
    vae_encoder_output = (
        (args.vae_encoder_output or output_path.with_name("seedvr2_ema_vae_encoder.onnx")).expanduser().absolute()
    )
    vae_decoder_output = (
        (args.vae_decoder_output or output_path.with_name("seedvr2_ema_vae_decoder.onnx")).expanduser().absolute()
    )
    positive_context_output = (
        (args.positive_context_output or output_path.with_name("seedvr2_3b_fp16_positive_context.bin"))
        .expanduser()
        .absolute()
    )

    components = set(args.components)
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
        )
    if {"dit", "context"} & components:
        print(f"Loading SeedVR diffusion model: {model_path}")
        diffusion_model, device, dtype = load_diffusion_model(model_path)
        print(f"DiT export device={device}, dtype={dtype}")
        export_positive_context(diffusion_model, positive_context_output)

    if "dit" in components:
        adapter = SeedVRDiTExportAdapter(diffusion_model).eval()
        inputs = dit_inputs_with_metadata(
            diffusion_model, device, dtype, args.temporal, args.latent_height, args.latent_width
        )
        if not args.skip_metadata_validation:
            validate_metadata_shapes(diffusion_model, device, dtype)
        if not args.skip_eager_parity:
            verify_eager_adapter(diffusion_model, adapter, inputs)
        capture_and_validate_exported_program(adapter, diffusion_model, inputs, device, dtype)
        export_onnx(
            adapter,
            inputs,
            output_path,
            artifacts_dir / "dit",
            DIT_INPUT_NAMES,
            ("denoised_latent",),
            dynamic_dit_shapes(),
        )
        print(f"Exported dynamic SeedVR DiT: {output_path}")
        if not args.skip_validation:
            validate_onnx(output_path)


if __name__ == "__main__":
    main()
