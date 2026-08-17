# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

from modeling import (
    SAM2ImageEncoder,
    SAM2ImagePreprocessor,
    SAM2MaskDecoder,
    SAM2MemoryAttention,
    SAM2MemoryEncoder,
    patch_sam2_fpn_neck_for_fp16_onnx,
    patch_sam2_rope_for_onnx,
)

GRAPH_NAMES = (
    "image_preprocess",
    "image_encoder",
    "sam2_decoder",
    "sam2_decoder_propagate",
    "memory_attention",
    "memory_encoder",
    "mask_postprocess",
)
DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def main() -> None:
    args = parse_args()
    if args.image_size != 1024:
        raise ValueError("This exporter is intentionally static-spatial; --image-size must be 1024")
    if args.max_points < 1:
        raise ValueError("--max-points must be at least 1")
    graphs = tuple(dict.fromkeys(item.strip() for item in args.graphs.split(",") if item.strip()))
    unknown_graphs = sorted(set(graphs) - set(GRAPH_NAMES))
    if unknown_graphs:
        raise ValueError(f"Unknown graph names: {unknown_graphs}. Valid names: {GRAPH_NAMES}")

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    needs_sam2_model = any(graph not in {"image_preprocess", "mask_postprocess"} for graph in graphs)
    model = None
    if needs_sam2_model:
        from sam2.build_sam import build_sam2_hf

        model = build_sam2_hf(args.model, device=args.device).eval()
        model = model.to(device=args.device, dtype=DTYPE_BY_NAME[args.dtype])
        if args.dtype == "float16":
            patch_sam2_fpn_neck_for_fp16_onnx()
    if "image_preprocess" in graphs:
        export_image_preprocess(args)
    if "mask_postprocess" in graphs:
        export_mask_postprocess(args)
    if "image_encoder" in graphs:
        assert model is not None
        export_image_encoder(model, args)
    if "sam2_decoder" in graphs:
        assert model is not None
        export_sam2_decoder(model, args, token_embeddings=False)
    if "sam2_decoder_propagate" in graphs:
        assert model is not None
        export_sam2_decoder(model, args, token_embeddings=True)
    if "memory_attention" in graphs:
        assert model is not None
        patch_sam2_rope_for_onnx(model)
        export_memory_attention(model, args)
    if "memory_encoder" in graphs:
        assert model is not None
        export_memory_encoder(model, args)
    write_metadata(model, args, graphs)


def export_image_preprocess(args: argparse.Namespace) -> None:
    dtype = DTYPE_BY_NAME[args.dtype]
    image_rgb = torch.zeros(1, args.image_size, args.image_size, 3, dtype=torch.uint8, device=args.device)
    export_onnx(
        SAM2ImagePreprocessor(args.image_size, dtype).to(args.device).eval(),
        (image_rgb,),
        args.output / "image_preprocess.onnx",
        input_names=["image_rgb"],
        output_names=["image"],
        args=args,
        opset=args.opset,
        dynamic_shapes={
            "image_rgb": {
                1: torch.export.Dim("height", min=1, max=args.max_input_size),
                2: torch.export.Dim("width", min=1, max=args.max_input_size),
            }
        },
    )


def export_mask_postprocess(args: argparse.Namespace) -> None:
    path = args.output / "mask_postprocess.onnx"
    f4 = args.image_size // 4
    low_res_masks = helper.make_tensor_value_info("low_res_masks", TensorProto.FLOAT16, [1, 1, f4, f4])
    output_size = helper.make_tensor_value_info("output_size", TensorProto.INT64, [2])
    mask_rgb = helper.make_tensor_value_info("mask_rgb", TensorProto.UINT8, [1, "output_height", "output_width", 3])
    initializers = [
        helper.make_tensor("resize_roi", TensorProto.FLOAT, [0], []),
        helper.make_tensor("resize_scales", TensorProto.FLOAT, [0], []),
        helper.make_tensor("resize_prefix_shape", TensorProto.INT64, [2], np.asarray([1, 1], dtype=np.int64)),
        helper.make_tensor("mask_prefix_shape", TensorProto.INT64, [1], np.asarray([1], dtype=np.int64)),
        helper.make_tensor("rgb_channels_shape", TensorProto.INT64, [1], np.asarray([3], dtype=np.int64)),
        helper.make_tensor("zero", TensorProto.FLOAT, [], np.asarray(0.0, dtype=np.float32)),
        helper.make_tensor("mask_value", TensorProto.FLOAT, [], np.asarray(255.0, dtype=np.float32)),
    ]
    nodes = [
        helper.make_node("Concat", ["resize_prefix_shape", "output_size"], ["resize_sizes"], axis=0),
        helper.make_node("Cast", ["low_res_masks"], ["low_res_masks_f32"], to=TensorProto.FLOAT),
        helper.make_node(
            "Resize",
            ["low_res_masks_f32", "resize_roi", "resize_scales", "resize_sizes"],
            ["resized_logits"],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            nearest_mode="floor",
        ),
        helper.make_node("Greater", ["resized_logits", "zero"], ["mask_bool"]),
        helper.make_node("Cast", ["mask_bool"], ["mask_f32"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["mask_f32", "mask_value"], ["mask_scaled"]),
        helper.make_node("Transpose", ["mask_scaled"], ["mask_f32_nhwc"], perm=[0, 2, 3, 1]),
        helper.make_node(
            "Concat", ["mask_prefix_shape", "output_size", "rgb_channels_shape"], ["mask_f32_rgb_shape"], axis=0
        ),
        helper.make_node("Expand", ["mask_f32_nhwc", "mask_f32_rgb_shape"], ["mask_f32_rgb"]),
        helper.make_node("Cast", ["mask_f32_rgb"], ["mask_rgb"], to=TensorProto.UINT8),
    ]
    graph = helper.make_graph(nodes, "SAM2MaskPostprocessor", [low_res_masks, output_size], [mask_rgb], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", args.opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save_model(model, path, save_as_external_data=False)
    print(f"Exported {path}")


def export_image_encoder(model: torch.nn.Module, args: argparse.Namespace) -> None:
    dtype = DTYPE_BY_NAME[args.dtype]
    image = torch.zeros(1, 3, args.image_size, args.image_size, dtype=dtype, device=args.device)
    export_onnx(
        SAM2ImageEncoder(model).eval(),
        (image,),
        args.output / "image_encoder.onnx",
        input_names=["image"],
        output_names=["image_features_0", "image_features_1", "image_embeddings", "vision_feat", "vision_pos"],
        args=args,
        opset=args.opset,
    )


def export_sam2_decoder(model: torch.nn.Module, args: argparse.Namespace, *, token_embeddings: bool) -> None:
    encoder = SAM2ImageEncoder(model).eval()
    dtype = DTYPE_BY_NAME[args.dtype]
    image = torch.zeros(1, 3, args.image_size, args.image_size, dtype=dtype, device=args.device)
    image_features_0, image_features_1, image_embeddings, _, _ = encoder(image)
    if token_embeddings:
        image_embeddings = image_embeddings.flatten(2).permute(2, 0, 1)
    point_count = 1 if token_embeddings else args.max_points
    point_coords = torch.zeros(1, point_count, 2, dtype=dtype, device=args.device)
    point_labels = torch.full((1, point_count), -1, dtype=torch.int32, device=args.device)
    input_masks = torch.zeros(1, 1, args.image_size // 4, args.image_size // 4, dtype=dtype, device=args.device)
    has_input_masks = torch.zeros(1, 1, 1, 1, dtype=dtype, device=args.device)
    name = "sam2_decoder_propagate" if token_embeddings else "sam2_decoder"
    export_onnx(
        SAM2MaskDecoder(model, multimask_output=False, token_embeddings=token_embeddings).eval(),
        (
            image_features_0,
            image_features_1,
            image_embeddings,
            point_coords,
            point_labels,
            input_masks,
            has_input_masks,
        ),
        args.output / f"{name}.onnx",
        input_names=[
            "image_features_0",
            "image_features_1",
            "image_embeddings",
            "point_coords",
            "point_labels",
            "input_masks",
            "has_input_masks",
        ],
        output_names=["masks", "low_res_masks", "object_score_logits"],
        args=args,
        opset=args.opset,
    )


def export_memory_attention(model: torch.nn.Module, args: argparse.Namespace) -> None:
    dtype = DTYPE_BY_NAME[args.dtype]
    hidden_dim = int(model.hidden_dim)
    mem_dim = int(getattr(model, "mem_dim", hidden_dim))
    tokens = (args.image_size // 16) ** 2
    memory_tokens = args.max_memory_frames * tokens
    vision_feat = torch.zeros(tokens, 1, hidden_dim, dtype=dtype, device=args.device)
    vision_pos = torch.zeros(tokens, 1, hidden_dim, dtype=dtype, device=args.device)
    memory = torch.zeros(memory_tokens, 1, mem_dim, dtype=dtype, device=args.device)
    memory_pos = torch.zeros(memory_tokens, 1, mem_dim, dtype=dtype, device=args.device)
    export_onnx(
        SAM2MemoryAttention(model).eval(),
        (vision_feat, vision_pos, memory, memory_pos),
        args.output / "memory_attention.onnx",
        input_names=["vision_feat", "vision_pos", "memory", "memory_pos"],
        output_names=["conditioned_image_embeddings"],
        args=args,
        opset=args.opset,
    )


def export_memory_encoder(model: torch.nn.Module, args: argparse.Namespace) -> None:
    dtype = DTYPE_BY_NAME[args.dtype]
    hidden_dim = int(model.hidden_dim)
    tokens = (args.image_size // 16) ** 2
    vision_feat = torch.zeros(tokens, 1, hidden_dim, dtype=dtype, device=args.device)
    pred_masks_high_res = torch.zeros(1, 1, args.image_size, args.image_size, dtype=dtype, device=args.device)
    object_score_logits = torch.ones(1, 1, dtype=dtype, device=args.device) * 10.0
    is_mask_from_pts = torch.tensor(1.0, dtype=dtype, device=args.device)
    export_onnx(
        SAM2MemoryEncoder(model).eval(),
        (vision_feat, pred_masks_high_res, object_score_logits, is_mask_from_pts),
        args.output / "memory_encoder.onnx",
        input_names=["vision_feat", "pred_masks_high_res", "object_score_logits", "is_mask_from_pts"],
        output_names=["memory_features", "memory_pos"],
        args=args,
        opset=args.opset,
    )


def export_onnx(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    args: argparse.Namespace,
    opset: int,
    dynamic_shapes: dict[str, dict[int, torch.export.Dim]] | None = None,
) -> None:
    torch.onnx.export(
        module,
        inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamo=True,
        optimize=True,
        external_data=False,
        dynamic_shapes=dynamic_shapes,
    )
    print(f"Exported {path}")


def write_metadata(model: torch.nn.Module | None, args: argparse.Namespace, graphs: tuple[str, ...]) -> None:
    metadata_path = args.output / "metadata.json"
    existing = {}
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        graphs = tuple(dict.fromkeys([*existing.get("graphs", []), *graphs]))
    if model is None and not existing:
        raise ValueError("preprocess-only export needs existing metadata; run the full SAM2 export first")

    image_size = int(model.image_size) if model is not None else existing["image_size"]
    max_points = args.max_points if model is not None else existing["max_points"]
    prompt_optimization_shapes = {
        "point_coords": [1, max_points, 2],
        "point_labels": [1, max_points],
        "input_masks": [1, 1, image_size // 4, image_size // 4],
        "has_input_masks": [1, 1, 1, 1],
    }
    decoder_optimization_shapes = existing.get(
        "decoder_optimization_shapes",
        existing.get(
            "decoder_contract",
            {
                "image_features_0": [1, 32, image_size // 4, image_size // 4],
                "image_features_1": [1, 64, image_size // 8, image_size // 8],
            },
        ),
    )
    if model is not None:
        encoder = SAM2ImageEncoder(model).eval()
        dtype = DTYPE_BY_NAME[args.dtype]
        image = torch.zeros(1, 3, image_size, image_size, dtype=dtype, device=args.device)
        with torch.no_grad():
            image_features_0, image_features_1, _, _, _ = encoder(image)
        decoder_optimization_shapes = {
            "image_features_0": [int(v) for v in image_features_0.shape],
            "image_features_1": [int(v) for v in image_features_1.shape],
        }

    metadata = {
        "model": args.model if model is not None else existing.get("model", args.model),
        "graphs": list(graphs),
        "image_size": image_size,
        "spatial_contract": "static_1024",
        "max_points": max_points,
        "max_memory_frames": args.max_memory_frames if model is not None else existing["max_memory_frames"],
        "max_input_size": args.max_input_size
        if model is not None
        else existing.get("max_input_size", args.max_input_size),
        "multimask_output": False,
        "dynamo": True,
        "optimize": True,
        "dtype": args.dtype if model is not None else existing["dtype"],
        "prompt_optimization_shapes": prompt_optimization_shapes,
        "decoder_optimization_shapes": decoder_optimization_shapes,
        "hidden_dim": int(model.hidden_dim) if model is not None else existing["hidden_dim"],
        "mem_dim": int(getattr(model, "mem_dim", model.hidden_dim)) if model is not None else existing["mem_dim"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SAM2 split ONNX graphs.")
    parser.add_argument("--model", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sam2/onnx"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--graphs",
        default="image_preprocess,image_encoder,sam2_decoder,sam2_decoder_propagate,memory_attention,memory_encoder,mask_postprocess",
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=3)
    parser.add_argument("--max-memory-frames", type=int, default=7)
    parser.add_argument("--max-input-size", type=int, default=4096)
    parser.add_argument("--opset", type=int, default=23)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
