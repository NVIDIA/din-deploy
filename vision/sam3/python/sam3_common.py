# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream loading shared by reference and export commands."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCK = json.loads((Path(__file__).parents[1] / "upstream.json").read_text(encoding="utf-8"))


def load_upstream() -> Path:
    # Upstream's fused performance kernels force BF16 and are not ONNX operators.
    os.environ["USE_PERFLIB"] = "0"
    source = ROOT / "artifacts/sam3/upstream" / ("sam3-" + LOCK["source_revision"])
    if not (source / "sam3/model_builder.py").is_file():
        raise FileNotFoundError("Run vision/sam3/python/prepare_sam3.py first")
    # Optional task-local dependencies preserve the configured interpreter's packages.
    deps = ROOT / "artifacts/sam3/python_deps"
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    sys.path.insert(0, str(source))
    return source


def checkpoint_path() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        LOCK["checkpoint_repo"], LOCK["checkpoint_file"], revision=LOCK["checkpoint_revision"], local_files_only=True
    )


def build_predictor():
    load_upstream()
    import sam3.model_builder as builder
    import torch
    from sam3.model import decoder, vitdet
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # Upstream forces the Flash-only backend even for FP32 inputs. Math SDPA is
    # required for a full-precision reference and is decomposable by ONNX export.
    decoder.sdpa_kernel = lambda backends: sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    )

    # This fused helper unconditionally casts to BF16 even with USE_PERFLIB=0.
    # Express the same linear + activation mathematically in the requested dtype.
    def unfused_addmm_act(activation, linear, value):
        return activation()(linear(value))

    vitdet.addmm_act = unfused_addmm_act

    # The merged checkpoint is loaded into the completed model below. The upstream
    # builder also tries to load it into the unprefixed tracker first, which cannot
    # match the merged keys and needlessly reads the entire checkpoint twice.
    original = builder.build_sam3_multiplex_video_model

    def unloaded_tracker(*args, **kwargs):
        kwargs["checkpoint_path"] = None
        kwargs["load_from_HF"] = False
        return original(*args, **kwargs)

    builder.build_sam3_multiplex_video_model = unloaded_tracker
    model_type = Sam3MultiplexTrackingWithInteractivity
    own_loader = model_type.__dict__.get("load_state_dict")
    original_loader = model_type.load_state_dict

    def checked_loader(instance, *args, **kwargs):
        result = original_loader(instance, *args, **kwargs)
        missing = [
            key for key in result.missing_keys if not key.endswith((".attn.freqs_cis_real", ".attn.freqs_cis_imag"))
        ]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"Checkpoint coverage mismatch: missing={missing}, unexpected={result.unexpected_keys}")
        return result

    model_type.load_state_dict = checked_loader
    try:
        predictor = builder.build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path(),
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=True,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
        )
    finally:
        builder.build_sam3_multiplex_video_model = original
        if own_loader is None:
            delattr(model_type, "load_state_dict")
        else:
            model_type.load_state_dict = own_loader
    # Upstream enters a process-wide autocast context in its constructor. Explicit
    # precision is required for the FP32 export baseline and reference comparison.
    predictor.bf16_context.__exit__(None, None, None)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    predictor.model.hotstart_delay = 0
    predictor.model.postprocess_batch_size = 1
    predictor.model.batched_grounding_batch_size = 1
    # The pinned base predictor passes this SAM2-only argument to the multiplex
    # implementation, whose init_state does not accept it. State offload remains
    # disabled; fail explicitly if a caller tries to enable it.
    init_state = predictor.model.init_state

    def multiplex_init_state(*args, offload_state_to_cpu=False, **kwargs):
        if offload_state_to_cpu:
            raise ValueError("SAM3.1 multiplex state offload is not supported")
        return init_state(*args, **kwargs)

    predictor.model.init_state = multiplex_init_state
    return predictor
