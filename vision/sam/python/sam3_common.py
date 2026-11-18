# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream loading shared by reference and export commands."""

from __future__ import annotations

import importlib.util
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_REPO = "facebook/sam3.1"
DEFAULT_CHECKPOINT_FILE = "sam3.1_multiplex.pt"
DEFAULT_CHECKPOINT_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"


def add_checkpoint_arguments(parser):
    parser.add_argument(
        "--checkpoint-repo", default=DEFAULT_CHECKPOINT_REPO, help="SAM3 Hugging Face checkpoint repository"
    )
    parser.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT_FILE, help="SAM3 checkpoint filename")
    parser.add_argument("--checkpoint-revision", default=DEFAULT_CHECKPOINT_REVISION, help="SAM3 checkpoint revision")


def checkpoint_options(args):
    return {name: getattr(args, name) for name in ("checkpoint_repo", "checkpoint_file", "checkpoint_revision")}


def upstream_metadata(args):
    with (ROOT / "pyproject.toml").open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["optional-dependencies"]["sam3"]
    source = next(dependency for dependency in dependencies if dependency.startswith("sam3 @ git+"))
    repository, revision = source.split(" @ git+", 1)[1].rsplit("@", 1)
    return {"source_repo": repository.removesuffix(".git"), "source_revision": revision, **checkpoint_options(args)}


def load_upstream() -> Path:
    # Upstream's fused performance kernels force BF16 and are not ONNX operators.
    os.environ["USE_PERFLIB"] = "0"
    package = importlib.util.find_spec("sam3")
    if package is None or package.origin is None:
        raise ModuleNotFoundError('Install the pinned SAM3 dependency with uv pip install -e ".[sam3]"')
    # Preserve the source-root contract used by the standalone tokenizer exporter.
    return Path(package.origin).resolve().parent.parent


def checkpoint_path(
        *,
        checkpoint_repo=DEFAULT_CHECKPOINT_REPO,
        checkpoint_file=DEFAULT_CHECKPOINT_FILE,
        checkpoint_revision=DEFAULT_CHECKPOINT_REVISION,
        local_files_only=True,
) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        checkpoint_repo, checkpoint_file, revision=checkpoint_revision, local_files_only=local_files_only
    )


def build_predictor(
        *,
        checkpoint_repo=DEFAULT_CHECKPOINT_REPO,
        checkpoint_file=DEFAULT_CHECKPOINT_FILE,
        checkpoint_revision=DEFAULT_CHECKPOINT_REVISION,
):
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
            checkpoint_path=checkpoint_path(
                checkpoint_repo=checkpoint_repo,
                checkpoint_file=checkpoint_file,
                checkpoint_revision=checkpoint_revision,
            ),
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
    return predictor
