# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import json
from collections.abc import Iterable
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from modeling import SAM2ImageEncoder, SAM2ImagePreprocessor


@dataclass(frozen=True)
class Sample:
    image: torch.Tensor
    teacher: tuple[torch.Tensor, ...]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    require_modelopt()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    sam_model = load_sam2(args.model, args.device, dtype)
    preprocessor = SAM2ImagePreprocessor(args.image_size, dtype).to(args.device).eval()
    encoder = SAM2ImageEncoder(sam_model).to(args.device).eval()

    rows = load_dataset_rows(args.dataset, args.split, args.calib_samples + args.score_samples)
    images = [preprocess_image(row["image"], preprocessor, args.device) for row in rows]
    samples = build_samples(encoder, images)

    calib_samples = samples[: args.calib_samples]
    score_samples = samples[args.calib_samples : args.calib_samples + args.score_samples]
    amax = collect_amax(encoder, calib_samples)
    disabled_layers = choose_disabled_layers(
        encoder,
        amax,
        boundary_layers=args.boundary_layers,
        small_weight_threshold=args.small_weight_threshold,
        high_amax_percent=args.high_amax_percent,
        disable_attn_proj=args.disable_attn_proj,
    )

    quantized = quantize_encoder_ptq(
        encoder,
        calib_samples,
        disabled_layers,
    )
    metrics = score_feature_drift(quantized, score_samples)
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "calib_samples": len(calib_samples),
        "score_samples": len(score_samples),
        "disabled_layers": disabled_layers,
        "metrics": metrics,
        "artifacts": {
            "modelopt_model": str(args.output / "sam2_encoder_fp8_modelopt.pt"),
            "quant_summary": str(args.output / "quant_summary.txt"),
        },
    }

    save_modelopt(quantized, args.output / "sam2_encoder_fp8_modelopt.pt")
    write_quant_summary(quantized, args.output / "quant_summary.txt", verbose=args.verbose)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))


def load_sam2(model_id: str, device: str, dtype: torch.dtype) -> nn.Module:
    from sam2.build_sam import build_sam2_hf

    return build_sam2_hf(model_id, device=device).to(device=device, dtype=dtype).eval()


def require_modelopt() -> None:
    if importlib.util.find_spec("modelopt") is None:
        raise RuntimeError(
            "SAM2 FP8 quantization requires NVIDIA ModelOpt. Install it with "
            "`python -m pip install -e '.[quantization]'` or `python -m pip install nvidia-modelopt`."
        )


def load_dataset_rows(dataset: str, split: str, count: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Install `datasets` to run SAM2 FP8 calibration.") from error

    data = load_dataset(dataset, split=split, streaming=True)
    rows = []
    for row in data:
        rows.append(row)
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise ValueError(f"Requested {count} rows, but only loaded {len(rows)} from {dataset}/{split}.")
    return rows


def preprocess_image(image: Image.Image, preprocessor: SAM2ImagePreprocessor, device: str) -> torch.Tensor:
    import numpy as np

    rgb = image.convert("RGB")
    tensor = torch.from_numpy(np.array(rgb, copy=True)).to(device=device, dtype=torch.uint8).unsqueeze(0)
    with torch.inference_mode():
        return preprocessor(tensor)


def build_samples(encoder: SAM2ImageEncoder, images: Iterable[torch.Tensor]) -> list[Sample]:
    samples = []
    with torch.inference_mode():
        for image in images:
            teacher = tuple(output.detach() for output in encoder(image))
            samples.append(Sample(image=image, teacher=teacher))
    return samples


def collect_amax(encoder: nn.Module, samples: list[Sample]) -> dict[str, float]:
    amax: dict[str, float] = {}
    hooks = []
    for name, module in quantizable_modules(encoder):
        hooks.append(module.register_forward_hook(lambda _m, _i, out, name=name: record_amax(amax, name, out)))
    try:
        with torch.inference_mode():
            for sample in samples:
                encoder(sample.image)
    finally:
        for hook in hooks:
            hook.remove()
    return amax


def record_amax(amax: dict[str, float], name: str, output: Any) -> None:
    tensors = [item for item in flatten(output) if isinstance(item, torch.Tensor)]
    if not tensors:
        return
    value = max(float(tensor.detach().abs().max().cpu()) for tensor in tensors)
    amax[name] = max(value, amax.get(name, 0.0))


def flatten(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten(item)
    else:
        yield value


def choose_disabled_layers(
    encoder: nn.Module,
    amax: dict[str, float],
    *,
    boundary_layers: int,
    small_weight_threshold: int,
    high_amax_percent: float,
    disable_attn_proj: bool,
) -> list[str]:
    modules = list(quantizable_modules(encoder))
    disabled = set()

    for name, _ in modules[:boundary_layers]:
        disabled.add(name)
    for name, _ in modules[-boundary_layers:]:
        disabled.add(name)

    boundary_terms = ("patch_embed", "stem", "neck", "conv_s0", "conv_s1", "output", "head")
    for name, module in modules:
        if any(term in name for term in boundary_terms):
            disabled.add(name)
        if weight_count(module) < small_weight_threshold:
            disabled.add(name)

    disabled.update(name for name, module in encoder.named_modules() if is_block_norm_or_projection(name, module))
    if disable_attn_proj:
        disabled.update(name for name, _module in quantizable_modules(encoder) if name.endswith(".attn.proj"))

    if amax:
        cutoff_index = max(1, int(len(amax) * high_amax_percent / 100.0))
        for name, _value in sorted(amax.items(), key=lambda item: item[1], reverse=True)[:cutoff_index]:
            disabled.add(name)

    return sorted(f"{name}*" for name in disabled)


def quantizable_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("model.image_encoder.") and isinstance(module, (nn.Conv2d, nn.Linear))
    ]


def is_block_norm_or_projection(name: str, module: nn.Module) -> bool:
    if not name.startswith("model.image_encoder.trunk.blocks."):
        return False
    return isinstance(module, nn.LayerNorm) or is_stage_transition_projection(name)


def is_stage_transition_projection(name: str) -> bool:
    parts = name.split(".")
    return len(parts) >= 6 and parts[-1] == "proj" and parts[-2].isdigit() and parts[-3] == "blocks"


def weight_count(module: nn.Module) -> int:
    weight = getattr(module, "weight", None)
    return 0 if weight is None else int(weight.numel())


def quantize_encoder_ptq(
    encoder: SAM2ImageEncoder,
    calib_samples: list[Sample],
    disabled_layers: list[str],
) -> nn.Module:
    import modelopt.torch.quantization as mtq

    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    config["quant_cfg"].extend(
        {"quantizer_name": pattern, "enable": False}
        for pattern in (
            "model.image_encoder.neck*",
            "model.image_encoder.trunk.patch_embed*",
            "model.image_encoder.trunk.blocks.*.norm*",
        )
    )
    config["quant_cfg"].extend({"quantizer_name": pattern, "enable": False} for pattern in disabled_layers)
    config["quant_cfg"].extend(
        {"quantizer_name": pattern, "enable": False}
        for pattern in (
            "model.sam_prompt_encoder*",
            "model.sam_mask_decoder*",
            "model.memory_*",
            "model.obj_ptr*",
            "model.mask_downsample*",
        )
    )

    def forward_loop(model: nn.Module) -> None:
        model.eval()
        with torch.inference_mode():
            for sample in calib_samples:
                model(sample.image)

    encoder.eval()
    return mtq.quantize(encoder, config, forward_loop)


def feature_loss(output: tuple[torch.Tensor, ...], teacher: tuple[torch.Tensor, ...]) -> torch.Tensor:
    losses = [
        torch.mean((candidate.float() - reference.float()) ** 2)
        for candidate, reference in zip(output, teacher, strict=True)
    ]
    return torch.stack(losses).mean()


def score_feature_drift(encoder: nn.Module, samples: list[Sample]) -> dict[str, float]:
    losses = []
    with torch.inference_mode():
        for sample in samples:
            losses.append(float(feature_loss(tuple(encoder(sample.image)), sample.teacher).cpu()))
    values = torch.tensor(losses, dtype=torch.float64)
    return {
        "feature_mse_mean": float(values.mean()),
        "feature_mse_max": float(values.max()),
    }


def save_modelopt(model: nn.Module, path: Path) -> None:
    import modelopt.torch.opt as mto

    mto.save(model, path)


def write_quant_summary(model: nn.Module, path: Path, *, verbose: bool) -> None:
    import modelopt.torch.quantization as mtq

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        mtq.print_quant_summary(model)
    summary = buffer.getvalue()
    path.write_text(summary, encoding="utf-8")
    if verbose:
        print(summary, end="")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal SAM2 image encoder FP8 prototype.")
    parser.add_argument("--model", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--dataset", default="TNILab/pascal_voc_seg_train_val")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sam2/fp8_encoder"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--score-samples", type=int, default=64)
    parser.add_argument("--boundary-layers", type=int, default=2)
    parser.add_argument("--small-weight-threshold", type=int, default=16_384)
    parser.add_argument("--high-amax-percent", type=float, default=5.0)
    parser.add_argument("--disable-attn-proj", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
