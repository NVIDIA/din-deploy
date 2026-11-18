# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrate conservative SAM encoder FP8 quantization with ModelOpt."""

import argparse
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from sam3_common import add_checkpoint_arguments, checkpoint_options, upstream_metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_checkpoint_arguments(parser)
    parser.add_argument("--model-family", choices=("sam2", "sam3"), required=True)
    parser.add_argument("--model", help="SAM2 Hugging Face model ID; SAM3 uses --checkpoint-* options")
    parser.add_argument(
        "--dataset", default="TNILab/pascal_voc_seg_train_val", help="Hugging Face dataset ID or local image directory"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--score-samples", type=int, default=64)
    parser.add_argument("--boundary-layers", type=int, default=2)
    parser.add_argument("--small-weight-threshold", type=int, default=16_384)
    parser.add_argument("--high-amax-percent", type=float, default=5.0)
    parser.add_argument("--disable-attn-proj", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if min(args.calib_samples, args.score_samples) < 1:
        parser.error("Calibration and scoring sample counts must be positive")
    if min(args.boundary_layers, args.small_weight_threshold) < 0 or not 0 <= args.high_amax_percent <= 100:
        parser.error("Exclusion counts must be nonnegative and high-amax-percent must be in [0, 100]")
    if args.model_family == "sam3" and (args.model or not args.device.startswith("cuda")):
        parser.error("SAM3 requires CUDA and uses --checkpoint-* options; omit --model")
    if args.output.exists():
        parser.error("Use a new output directory")
    return args


def load_encoder(args):
    import torch

    dtype = getattr(torch, args.dtype)
    if args.model_family == "sam2":
        from sam2.build_sam import build_sam2_hf

        from sam2_modeling import SAM2ImageEncoder, SAM2ImagePreprocessor, patch_sam2_fpn_neck_for_fp16_onnx

        model_id = args.model or "facebook/sam2.1-hiera-large"
        model = build_sam2_hf(model_id, device=args.device).eval()
        {"bfloat16": model.bfloat16, "float16": model.half, "float32": model.float}[args.dtype]()
        if args.dtype == "float16":
            patch_sam2_fpn_neck_for_fp16_onnx()
        return (
            SAM2ImageEncoder(model).eval(),
            SAM2ImagePreprocessor(1024, dtype).to(args.device).eval(),
            {"model": model_id, "image_size": 1024},
        )
    from sam3_common import build_predictor
    from sam3_modeling import ImagePreprocess, VisionEncoder

    model = build_predictor(**checkpoint_options(args)).model.to(device=args.device).eval()
    # Match export precision helpers, retaining complex rotary buffers.
    {"bfloat16": model.bfloat16, "float16": model.half, "float32": model.float}[args.dtype]()
    return (
        VisionEncoder(model.detector.backbone).eval(),
        ImagePreprocess(dtype).to(args.device).eval(),
        {**upstream_metadata(args), "image_size": 1008},
    )


def choose_disabled_layers(modules, amax, args):
    names = list(modules)
    disabled = set(names[: args.boundary_layers])
    if args.boundary_layers:
        disabled.update(names[-args.boundary_layers:])
    for name, module in modules.items():
        if any(term in name for term in ("patch_embed", "stem", "neck", "conv_s0", "conv_s1", "output", "head")):
            disabled.add(name)
        if module.weight.numel() < args.small_weight_threshold:
            disabled.add(name)
        if args.disable_attn_proj and name.endswith(".attn.proj"):
            disabled.add(name)
        # SAM2 hierarchical stage-transition projections remain dense.
        if name.endswith(".proj") and name.split(".")[-2].isdigit():
            disabled.add(name)
    if args.high_amax_percent:
        count = max(1, int(len(amax) * args.high_amax_percent / 100))
        disabled.update(sorted(amax, key=amax.get, reverse=True)[:count])
    return disabled


def dataset_rows(dataset, split, count):
    directory = Path(dataset)
    if directory.is_dir():
        from PIL import Image

        directory = directory / split if (directory / split).is_dir() else directory
        paths = sorted(
            path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        for path in paths[:count]:
            with Image.open(path) as image:
                yield {"image": image.convert("RGB")}
    else:
        from datasets import load_dataset

        yield from load_dataset(dataset, split=split, streaming=True).take(count)


def main():
    args = parse_args()
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    import numpy as np
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder, preprocess, provenance = load_encoder(args)
    print(f"Loaded {args.model_family} encoder ({args.dtype}); preparing calibration/scoring images", flush=True)
    prefix = "model.image_encoder." if args.model_family == "sam2" else "visual.trunk."
    modules = {
        name: module
        for name, module in encoder.named_modules()
        if name.startswith(prefix) and isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))
    }
    if not modules:
        raise ValueError("No quantizable encoder modules found")

    def image_tensor(row):
        rgb = np.array(row["image"].convert("RGB"), copy=True)
        tensor = torch.from_numpy(rgb).to(args.device).unsqueeze(0)
        value = preprocess(tensor)
        return value[0] if isinstance(value, tuple) else value

    def encode(model, image):
        # SAM3 export uses autocast for reduced precision, including upstream
        # operations that intentionally retain FP32 parameters/intermediates.
        with torch.autocast(
                "cuda", dtype=getattr(torch, args.dtype),
                enabled=args.model_family == "sam3" and args.dtype != "float32"
        ):
            return model(image)

    # Cache only preprocessed inputs on the host; SAM3's 18 feature outputs make
    # retaining a whole evaluation set's reference activations prohibitively large.
    images, amax = [], {}
    hooks = []
    for name, module in modules.items():
        def record(_module, _inputs, output, name=name):
            value = output.detach().abs().max()
            amax[name] = torch.maximum(amax[name], value) if name in amax else value

        hooks.append(module.register_forward_hook(record))
    count = args.calib_samples + args.score_samples
    try:
        with torch.inference_mode():
            for index, row in enumerate(dataset_rows(args.dataset, args.split, count)):
                if index == args.calib_samples:
                    for hook in hooks:
                        hook.remove()
                image = image_tensor(row)
                if index < args.calib_samples:
                    encode(encoder, image)
                images.append(image.cpu())
                if (index + 1) % 16 == 0 or index + 1 == count:
                    print(f"Prepared {index + 1}/{count} dataset rows", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    if len(images) != count:
        raise ValueError(f"Requested {count} dataset rows, received {len(images)}")
    disabled = choose_disabled_layers(modules, {name: value.item() for name, value in amax.items()}, args)
    selected = sorted(set(modules) - disabled)
    if not selected:
        raise ValueError("Exclusion policy selected no layers")
    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    config["quant_cfg"].append({"quantizer_name": "*", "enable": False})
    for name in selected:
        for kind in ("input", "weight"):
            config["quant_cfg"].append(
                {
                    "quantizer_name": f"{name}.{kind}_quantizer",
                    "cfg": {"num_bits": (4, 3), "axis": None},
                    "enable": True,
                }
            )

    def forward_loop(model):
        with torch.inference_mode():
            for index, image in enumerate(images[: args.calib_samples]):
                encode(model, image.to(args.device))
                if (index + 1) % 16 == 0 or index + 1 == args.calib_samples:
                    print(f"Calibrated {index + 1}/{args.calib_samples} images", flush=True)

    encoder = mtq.quantize(encoder, config, forward_loop)
    quantizers = dict(encoder.named_modules())
    maxima = {}
    for name in selected:
        for kind in ("input", "weight"):
            key = f"{name}.{kind}_quantizer"
            value = quantizers[key].amax.detach().float()
            if value.numel() != 1 or not bool(torch.isfinite(value).all()) or float(value) <= 0:
                raise ValueError(f"Expected positive finite per-tensor amax: {key}")
            maxima[key] = value.item()
    losses = []
    active_quantizers = [quantizers[key] for key in maxima]
    with torch.inference_mode():
        for image in images[args.calib_samples:]:
            image = image.to(args.device)
            # FP8_DEFAULT_CFG is fake quantization: disabling its quantizers
            # gives the dense reference without cloning the model or dataset outputs.
            for quantizer in active_quantizers:
                quantizer.disable()
            teacher = tuple(value.cpu() for value in encode(encoder, image))
            for quantizer in active_quantizers:
                quantizer.enable()
            actual = encode(encoder, image)
            losses.append(
                sum(
                    float((value.float().cpu() - reference.float()).square().mean())
                    for value, reference in zip(actual, teacher, strict=True)
                )
                / len(teacher)
            )
            if len(losses) % 8 == 0 or len(losses) == args.score_samples:
                print(f"Scored {len(losses)}/{args.score_samples} images", flush=True)
    report = {
        "schema_version": 1,
        "model_family": args.model_family,
        "dtype": args.dtype,
        **provenance,
        "dataset": args.dataset,
        "split": args.split,
        "calib_samples": args.calib_samples,
        "score_samples": args.score_samples,
        "selected_layers": selected,
        "disabled_layers": sorted(disabled),
        "amax": maxima,
        "metrics": {"feature_mse_mean": sum(losses) / len(losses), "feature_mse_max": max(losses)},
    }
    args.output.mkdir(parents=True)
    mto.save(encoder, args.output / "encoder_fp8_modelopt.pt")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        mtq.print_quant_summary(encoder)
    (args.output / "quant_summary.txt").write_text(buffer.getvalue(), encoding="utf-8")
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if args.verbose:
        print(buffer.getvalue())
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
