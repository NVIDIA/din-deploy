# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def patch_sam2_rope_for_onnx(model: nn.Module) -> None:
    import sam2.modeling.position_encoding as position_encoding
    import sam2.modeling.sam.transformer as transformer
    from sam2.modeling.sam.transformer import RoPEAttention

    position_encoding.compute_axial_cis = compute_axial_cis_real
    position_encoding.apply_rotary_enc = apply_rotary_enc_real
    transformer.compute_axial_cis = compute_axial_cis_real
    transformer.apply_rotary_enc = apply_rotary_enc_real
    for module in model.modules():
        if isinstance(module, RoPEAttention):
            theta = getattr(module.compute_cis, "keywords", {}).get("theta", 10000.0)
            dim = module.internal_dim // module.num_heads
            module.compute_cis = lambda end_x, end_y, dim=dim, theta=theta: compute_axial_cis_real(
                dim, end_x, end_y, theta
            )
            module.freqs_cis = module.compute_cis(64, 64)


def patch_sam2_fpn_neck_for_fp16_onnx() -> None:
    from sam2.modeling.backbones.image_encoder import FpnNeck

    FpnNeck.forward = fpn_neck_forward_fp16_onnx


def fpn_neck_forward_fp16_onnx(
    self: nn.Module, xs: list[torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    out = [None] * len(self.convs)
    pos = [None] * len(self.convs)
    prev_features = None
    n = len(self.convs) - 1
    for i in range(n, -1, -1):
        x = xs[i]
        lateral_features = self.convs[n - i](x)
        if i in self.fpn_top_down_levels and prev_features is not None:
            if self.fpn_interp_model != "nearest":
                raise RuntimeError("SAM2 FP16 ONNX export patch only supports nearest FPN interpolation")
            top_down_features = prev_features.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
            prev_features = lateral_features + top_down_features
            if self.fuse_type == "avg":
                prev_features = prev_features / 2
        else:
            prev_features = lateral_features
        out[i] = prev_features
        pos[i] = self.position_encoding(prev_features).to(prev_features.dtype)
    return out, pos


def compute_axial_cis_real(dim: int, end_x: int, end_y: int, theta: float = 10000.0) -> torch.Tensor:
    from sam2.modeling.position_encoding import init_t_xy

    end_x = int(end_x)
    end_y = int(end_y)
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t_x, t_y = init_t_xy(end_x, end_y)
    angles = torch.cat([torch.outer(t_x, freqs_x), torch.outer(t_y, freqs_y)], dim=-1)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)


def apply_rotary_enc_real(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if freqs_cis.is_complex():
        freqs_cis = torch.view_as_real(freqs_cis)
    xq_out = rotate_with_real_freqs(xq, freqs_cis)
    if xk.shape[-2] == 0:
        return xq_out.type_as(xq).to(xq.device), xk
    k_freqs = freqs_cis
    if repeat_freqs_k:
        repeat = xk.shape[-2] // xq.shape[-2]
        k_freqs = k_freqs.unsqueeze(1).expand(-1, repeat, -1, -1).flatten(0, 1)
    xk_out = rotate_with_real_freqs(xk, k_freqs)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def rotate_with_real_freqs(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    cos = freqs[:, :, 0].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    sin = freqs[:, :, 1].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


class SAM2ImageEncoder(nn.Module):
    def __init__(self, sam_model: nn.Module) -> None:
        super().__init__()
        self.model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.no_mem_embed = sam_model.no_mem_embed

    @torch.no_grad()
    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_out = self.image_encoder(image)
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])

        output_dtype = image.dtype
        feature_maps = [
            x.to(dtype=output_dtype) for x in backbone_out["backbone_fpn"][-self.model.num_feature_levels :]
        ]
        vision_pos_embeds = [
            x.to(dtype=output_dtype) for x in backbone_out["vision_pos_enc"][-self.model.num_feature_levels :]
        ]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]

        raw_vision_feat = vision_feats[-1]
        raw_vision_pos = vision_pos_embeds[-1].flatten(2).permute(2, 0, 1)
        image_vision_feats = list(vision_feats)
        image_vision_feats[-1] = image_vision_feats[-1] + self.no_mem_embed.to(dtype=output_dtype)
        feats = [
            feat.permute(1, 2, 0).reshape(image.shape[0], -1, *feat_size)
            for feat, feat_size in zip(image_vision_feats[::-1], feat_sizes[::-1], strict=False)
        ][::-1]
        return feats[0], feats[1], feats[2], raw_vision_feat, raw_vision_pos


class SAM2ImagePreprocessor(nn.Module):
    def __init__(self, image_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.image_size = image_size
        self.dtype = dtype
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, image_rgb: torch.Tensor) -> torch.Tensor:
        image = image_rgb.to(dtype=torch.float32) / 255.0
        image = image.permute(0, 3, 1, 2)
        image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        image = (image - self.mean) / self.std
        return image.to(dtype=self.dtype)


class SAM2PromptEncoder(nn.Module):
    def __init__(self, sam_model: nn.Module) -> None:
        super().__init__()
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.model = sam_model

    def dense_pe_for(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        height, width = image_embeddings.shape[-2], image_embeddings.shape[-1]
        grid = torch.ones((height, width), device=image_embeddings.device, dtype=image_embeddings.dtype)
        y_embed = grid.cumsum(dim=0) - image_embeddings.new_tensor(0.5)
        x_embed = grid.cumsum(dim=1) - image_embeddings.new_tensor(0.5)
        y_embed = y_embed / height
        x_embed = x_embed / width
        image_pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return image_pe.permute(2, 0, 1).unsqueeze(0)

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords.to(dtype=self.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix.dtype)
        point_coords = point_coords + point_coords.new_tensor(0.5)
        padding_point = torch.zeros((point_coords.shape[0], 1, 2), device=point_coords.device, dtype=point_coords.dtype)
        padding_label = -torch.ones((point_labels.shape[0], 1), device=point_labels.device, dtype=point_labels.dtype)
        point_coords = torch.cat([point_coords, padding_point], dim=1)
        point_labels = torch.cat([point_labels, padding_label], dim=1)

        point_coords[:, :, 0] = point_coords[:, :, 0] / self.model.image_size
        point_coords[:, :, 1] = point_coords[:, :, 1] / self.model.image_size
        point_embedding = self._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)
        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (point_labels == -1)
        for index in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[index].weight * (
                point_labels == index
            )
        return point_embedding

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords * coords.new_tensor(2.0) - coords.new_tensor(1.0)
        gaussian = self.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix.to(
            device=coords.device,
            dtype=coords.dtype,
        )
        coords = coords @ gaussian
        coords = coords * coords.new_tensor(2.0 * np.pi)
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def _embed_masks(self, input_masks: torch.Tensor, has_input_masks: torch.Tensor) -> torch.Tensor:
        input_masks = input_masks.to(dtype=self.prompt_encoder.no_mask_embed.weight.dtype)
        has_input_masks = has_input_masks.to(dtype=input_masks.dtype)
        mask_embedding = self.prompt_encoder.mask_downscaling(input_masks)
        no_mask_embedding = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return has_input_masks * mask_embedding + (1.0 - has_input_masks) * no_mask_embedding


class SAM2MemoryAttention(nn.Module):
    def __init__(self, sam_model: nn.Module) -> None:
        super().__init__()
        self.memory_attention = sam_model.memory_attention

    @torch.no_grad()
    def forward(
        self,
        vision_feat: torch.Tensor,
        vision_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
    ) -> torch.Tensor:
        target_dtype = vision_feat.dtype
        conditioned = self.memory_attention(
            curr=[vision_feat],
            curr_pos=[vision_pos.to(dtype=target_dtype)],
            memory=memory.to(dtype=target_dtype),
            memory_pos=memory_pos.to(dtype=target_dtype),
            num_obj_ptr_tokens=0,
        )
        return conditioned


class SAM2MaskDecoder(nn.Module):
    def __init__(self, sam_model: nn.Module, *, multimask_output: bool = False, token_embeddings: bool = False) -> None:
        super().__init__()
        self.model = sam_model
        self.prompt_encoder = SAM2PromptEncoder(sam_model)
        self.multimask_output = multimask_output
        self.token_embeddings = token_embeddings

    @torch.no_grad()
    def forward(
        self,
        image_features_0: torch.Tensor,
        image_features_1: torch.Tensor,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        input_masks: torch.Tensor,
        has_input_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_dtype = image_features_0.dtype
        image_features_1 = image_features_1.to(dtype=target_dtype)
        image_embeddings = image_embeddings.to(dtype=target_dtype)
        if self.token_embeddings:
            embedding_h = image_features_0.shape[-2] // 4
            embedding_w = image_features_0.shape[-1] // 4
            image_embeddings = image_embeddings.permute(1, 2, 0).reshape(
                1, self.model.hidden_dim, embedding_h, embedding_w
            )
        point_coords = point_coords.to(dtype=target_dtype)
        input_masks = input_masks.to(dtype=target_dtype)
        has_input_masks = has_input_masks.to(dtype=target_dtype)
        sparse_embeddings = self.prompt_encoder._embed_points(point_coords, point_labels).to(dtype=target_dtype)
        dense_embeddings = self.prompt_encoder._embed_masks(input_masks, has_input_masks).to(dtype=target_dtype)
        low_res_multimasks, _, _, object_score_logits = self.model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.dense_pe_for(image_embeddings),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            repeat_image=False,
            high_res_features=[image_features_0, image_features_1],
        )
        if self.model.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, -1024.0)

        low_res_multimasks = low_res_multimasks.to(dtype=target_dtype)
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(image_embeddings.shape[-2] * 16, image_embeddings.shape[-1] * 16),
            mode="bilinear",
            align_corners=False,
        )
        return high_res_multimasks, low_res_multimasks, object_score_logits


class SAM2MemoryEncoder(nn.Module):
    def __init__(self, sam_model: nn.Module) -> None:
        super().__init__()
        self.model = sam_model
        self.memory_encoder = sam_model.memory_encoder
        self.hidden_dim = sam_model.hidden_dim

    @torch.no_grad()
    def forward(
        self,
        vision_feat: torch.Tensor,
        pred_masks_high_res: torch.Tensor,
        object_score_logits: torch.Tensor,
        is_mask_from_pts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_dtype = vision_feat.dtype
        pred_masks_high_res = pred_masks_high_res.to(dtype=target_dtype)
        object_score_logits = object_score_logits.to(dtype=target_dtype)
        feature_h = pred_masks_high_res.shape[-2] // 16
        feature_w = pred_masks_high_res.shape[-1] // 16
        pix_feat = vision_feat.permute(1, 2, 0).reshape(1, self.hidden_dim, feature_h, feature_w)
        hard_mask = (pred_masks_high_res > 0).to(dtype=pred_masks_high_res.dtype)
        soft_mask = torch.sigmoid(pred_masks_high_res)
        mask_selector = is_mask_from_pts.to(dtype=pred_masks_high_res.dtype).reshape(1, 1, 1, 1)
        mask_for_mem = mask_selector * hard_mask + (1.0 - mask_selector) * soft_mask
        if self.model.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.model.sigmoid_scale_for_mem_enc
        if self.model.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.model.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        if self.model.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).to(dtype=maskmem_out["vision_features"].dtype)
            maskmem_out["vision_features"] += (1 - is_obj_appearing[..., None, None]) * self.model.no_obj_embed_spatial[
                ..., None, None
            ].expand(*maskmem_out["vision_features"].shape)
        return maskmem_out["vision_features"].flatten(2).permute(2, 0, 1), maskmem_out["vision_pos_enc"][-1].flatten(
            2
        ).permute(2, 0, 1)
