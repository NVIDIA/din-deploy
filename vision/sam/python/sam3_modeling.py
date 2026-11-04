# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-only boundaries around the pinned SAM3.1 implementation."""

import types

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.text = backbone.language_backbone

    def forward(self, token_ids):
        _, memory = self.text.encoder(token_ids)
        return self.text.resizer(memory.transpose(0, 1)), token_ids == 0


class VisionEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.visual = backbone.vision_backbone

    def forward(self, image):
        features, positions, interactive, interactive_pos, propagation, propagation_pos = self.visual(image)
        return (
            tuple(x.tensors for x in features)
            + tuple(positions)
            + tuple(x.tensors for x in interactive)
            + tuple(interactive_pos)
            + tuple(x.tensors for x in propagation)
            + tuple(propagation_pos)
        )


class ConceptDetector(nn.Module):
    def __init__(self, detector, levels):
        super().__init__()
        self.detector = detector
        self.levels = levels
        self.detector.geometry_encoder = EmptyGeometry(detector.geometry_encoder)
        decoder = detector.transformer.decoder
        original_rpb = type(decoder)._get_rpb_matrix

        def static_rpb(instance, reference_boxes, feat_size):
            # The export is static at 1008 / 14. Upstream reconstructs this size
            # from a tensor, which otherwise creates a data-dependent export guard.
            return original_rpb(instance, reference_boxes, (72, 72))

        decoder.compilable_cord_cache = None
        decoder.compilable_stored_size = None
        decoder._get_rpb_matrix = types.MethodType(static_rpb, decoder)

    def forward(self, language_features, language_mask, *features_and_positions):
        from sam3.model.data_misc import FindStage, NestedTensor

        features = features_and_positions[: self.levels]
        positions = features_and_positions[self.levels :]
        backbone = {
            "backbone_fpn": [NestedTensor(x, None) for x in features],
            "vision_pos_enc": list(positions),
            "language_features": language_features,
            "language_mask": language_mask,
        }
        stage = FindStage(
            img_ids=torch.zeros(1, dtype=torch.long, device=language_features.device),
            text_ids=torch.zeros(1, dtype=torch.long, device=language_features.device),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        result = self.detector.forward_grounding(backbone, stage, None, self.detector._get_dummy_prompt())
        return result["pred_masks"], result["pred_boxes_xyxy"], result["pred_logits"], result["presence_logit_dec"]


class EmptyGeometry(nn.Module):
    """Exact empty-prompt branch, avoiding zero-box ROIAlign and host pinning."""

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry

    def forward(self, geo_prompt, img_feats, img_sizes, img_pos_embeds):
        geometry = self.geometry
        features = geometry.cls_embed.weight.reshape(1, 1, geometry.d_model)
        mask = torch.zeros(1, 1, dtype=torch.bool, device=features.device)
        if geometry.final_proj is not None:
            features = geometry.norm(geometry.final_proj(features))
        if geometry.encode is not None:
            for layer in geometry.encode:
                features = layer(tgt=features, memory=img_feats[-1], tgt_key_padding_mask=mask, pos=img_pos_embeds[-1])
            features = geometry.encode_norm(features)
        return features, mask


class ImagePreprocess(nn.Module):
    def forward(self, image_rgb):
        image = image_rgb.float().permute(0, 3, 1, 2)
        image = torch.nn.functional.interpolate(
            image, size=(1008, 1008), mode="bilinear", align_corners=False, antialias=True
        )
        return (image.round().clamp(0, 255) / 127.5 - 1,)


class FixedBucket:
    """One fixed 16-slot bucket; validity stays a tensor input to the graph."""

    multiplex_count = 16
    num_buckets = 1
    total_valid_entries = 16

    def __init__(self, valid):
        self.valid = valid

    def get_valid_object_mask(self):
        return self.valid.reshape(1, 16)

    def mux(self, value):
        return value.unsqueeze(0)

    def demux(self, value):
        return value.squeeze(0)


class SeedDecoder(nn.Module):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def forward(self, feature_0, feature_1, feature_2, masks, valid):
        tracker = self.tracker
        # The multiplex implementation shares high-resolution projections.
        high = [tracker.sam_mask_decoder.conv_s0(feature_0), tracker.sam_mask_decoder.conv_s1(feature_1)]
        feature = feature_2 + tracker.interactivity_no_mem_embed.reshape(1, 256, 1, 1)
        out = tracker._use_mask_as_output(feature, high, masks, FixedBucket(valid))
        return out["low_res_masks"], out["obj_ptr"], out["object_score_logits"]


class TrackingDecoder(nn.Module):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def forward(self, conditioned, feature_0, feature_1, valid):
        tracker = self.tracker
        high = [tracker.sam_mask_decoder.conv_s0(feature_0), tracker.sam_mask_decoder.conv_s1(feature_1)]
        out = tracker._forward_sam_heads(
            conditioned, propagation_high_res_features=high, multiplex_state=FixedBucket(valid), multimask_output=True
        )
        return out["low_res_masks"], out["obj_ptr"], out["object_score_logits"]


class TrackingMemory(nn.Module):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def forward(self, image_features, mask_logits, object_logits, valid, conditioning):
        high = torch.nn.functional.interpolate(
            mask_logits.float(), size=(self.tracker.input_mask_size, self.tracker.input_mask_size),
            mode="bilinear", align_corners=False
        )
        return self.encode(image_features, high, object_logits, valid, conditioning)

    def encode(self, image_features, high, object_logits, valid, conditioning):
        tracker = self.tracker
        masks = (high.sigmoid() * tracker.sigmoid_scale_for_mem_enc + tracker.sigmoid_bias_for_mem_enc) * valid.reshape(
            16, 1, 1, 1
        )
        condition = (conditioning * valid).reshape(1, 16, 1, 1).expand(1, 16, high.shape[-2], high.shape[-1])
        packed = torch.cat([masks.transpose(0, 1), condition], dim=1)
        out = tracker.maskmem_backbone(image_features, packed, skip_mask_sigmoid=True)
        appearing = ((object_logits > tracker.object_score_logit_threshold).float() * valid.reshape(16, 1)).reshape(
            1, 16, 1
        )
        no_object = ((1 - appearing) * tracker.no_obj_embed_spatial.unsqueeze(0)).sum(dim=1)
        # Match native in-place addition, including its BF16 rounding boundary.
        memory = out["vision_features"]
        memory += no_object[..., None, None]
        return memory, out["vision_pos_enc"][0]


class SeedMemory(TrackingMemory):
    def forward(self, image_features, mask_logits, object_logits, valid, conditioning):
        # Consume the actual _encode_new_memory boundary: consolidated high-res
        # logits, not the earlier binary seed prompt. Consolidation uses 1008;
        # direct mask propagation/reconditioning can supply 1152 instead.
        return self.encode(image_features, mask_logits.float(), object_logits, valid, conditioning)


class MemoryAttention(nn.Module):
    def __init__(self, tracker):
        super().__init__()
        self.encoder = tracker.transformer.encoder

    def forward(
        self,
        image_features,
        image_position,
        memory_image,
        memory_image_position,
        memory,
        memory_position,
        object_pointers,
        pointer_position,
    ):
        image = image_features.flatten(2).permute(2, 0, 1)
        position = image_position.flatten(2).permute(2, 0, 1)
        out = self.encoder(
            image=image,
            src=image,
            memory_image=memory_image,
            memory=torch.cat([memory, object_pointers], dim=0),
            image_pos=position,
            src_pos=position,
            memory_image_pos=memory_image_position,
            memory_pos=torch.cat([memory_position, pointer_position], dim=0),
            num_obj_ptr_tokens=object_pointers.shape[0],
        )
        return (out["memory"].permute(1, 2, 0).reshape(1, 256, 72, 72),)


class MemoryEncoder(nn.Module):
    def __init__(self, tracker):
        super().__init__()
        self.encoder = tracker.maskmem_backbone

    def forward(self, image_features, masks):
        out = self.encoder(image_features, masks, skip_mask_sigmoid=True)
        return out["vision_features"], out["vision_pos_enc"][0]
