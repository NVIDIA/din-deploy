# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export SAM3.1 tensor graphs with explicit, versioned runtime contracts."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import time
from pathlib import Path

from sam3_common import LOCK, build_predictor


def expand_attention_masks(path):
    """Make broadcast query axes explicit for Attention consumers such as ORT CPU."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(path, load_external_data=False)
    values = {value.name: value for value in (*model.graph.input, *model.graph.value_info)}
    nodes = []
    changed = False
    for node in model.graph.node:
        if node.op_type == "Attention" and len(node.input) > 3 and node.input[3]:
            query = values.get(node.input[0])
            mask = values.get(node.input[3])
            if query is not None and mask is not None:
                query_shape = [dim.dim_value for dim in query.type.tensor_type.shape.dim]
                mask_shape = [dim.dim_value for dim in mask.type.tensor_type.shape.dim]
                if len(query_shape) == len(mask_shape) == 4 and mask_shape[-2] == 1 and query_shape[-2] > 1 and all(mask_shape):
                    mask_shape[-2] = query_shape[-2]
                    prefix = node.name + "_broadcast_mask"
                    shape_name, output_name = prefix + "_shape", prefix + "_output"
                    model.graph.initializer.append(numpy_helper.from_array(np.array(mask_shape, dtype=np.int64), shape_name))
                    nodes.append(helper.make_node("Expand", [node.input[3], shape_name], [output_name], name=prefix))
                    model.graph.value_info.append(helper.make_tensor_value_info(output_name, mask.type.tensor_type.elem_type, mask_shape))
                    node.input[3] = output_name
                    changed = True
        nodes.append(node)
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        onnx.save(model, path)
    return changed


def export_graph(module, inputs, path, input_names, output_names, fixtures_only=False, opset_version=23):
    import onnx
    import torch

    print(f"Exporting {path.name}", flush=True)
    module.eval()
    with torch.inference_mode():
        expected = module(*inputs)
    dynamic_shapes = None
    if path.stem == "image_preprocess":
        dynamic_shapes = {
            "image_rgb": {1: torch.export.Dim("height", min=1, max=8192), 2: torch.export.Dim("width", min=1, max=8192)}
        }
    if path.stem == "memory_attention":
        memories = 5184 * torch.export.Dim("memory_frames", min=1, max=10)
        pointers = 16 * torch.export.Dim("pointer_frames", min=1, max=16)
        dynamic_shapes = (
            {},
            {},
            {0: memories},
            {0: memories},
            {0: memories},
            {0: memories},
            {0: pointers},
            {0: pointers},
        )
    if not fixtures_only:
        from antialias_resize import tensor_antialias

        rewrite = (
            tensor_antialias()
            if path.stem in ("seed_decoder", "seed_memory", "tracking_memory", "memory_encoder")
            else contextlib.nullcontext()
        )
        with rewrite:
            program = torch.onnx.export(
                module,
                inputs,
                input_names=input_names,
                output_names=output_names,
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
                opset_version=opset_version,
                external_data=True,
                optimize=False,
                verbose=False,
            )
        if path.stem in (
            "tracking_decoder",
            "memory_attention",
            "seed_decoder",
            "seed_memory",
            "tracking_memory",
            "memory_encoder",
        ):
            program.optimize()
        program.save(path, external_data=True)
        from lower_sequences import lower_sequences

        lower_sequences(path)
        expand_attention_masks(path)
    onnx.checker.check_model(str(path))
    import numpy as np

    np.savez(path.with_suffix(".inputs.npz"), **{k: v.cpu().numpy() for k, v in zip(input_names, inputs, strict=True)})
    np.savez(
        path.with_suffix(".outputs.npz"), **{k: v.cpu().numpy() for k, v in zip(output_names, expected, strict=True)}
    )
    raw_dir = path.parent / (path.stem + ".fixtures")
    raw_dir.mkdir(exist_ok=True)
    for names, values in ((input_names, inputs), (output_names, expected)):
        for name, value in zip(names, values, strict=True):
            value.cpu().numpy().tofile(raw_dir / (name + ".bin"))
    graph = onnx.load(path, load_external_data=False).graph
    if path.stem in ('seed_memory', 'tracking_memory') and not any(
        'valid' in node.input for node in graph.node
    ):
        raise ValueError('Memory export lost the validity input; check for aliased export fixtures')

    def describe(values):
        return [
            {
                "name": value.name,
                "shape": [dimension.dim_param or dimension.dim_value for dimension in value.type.tensor_type.shape.dim],
                "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type).lower(),
            }
            for value in values
        ]

    return {
        "file": path.name,
        "inputs": describe(graph.input),
        "outputs": describe(graph.output),
        "fixture_inputs": {name: list(value.shape) for name, value in zip(input_names, inputs, strict=True)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sam3/onnx-fp32"))
    graph_names = [
        "image_preprocess",
        "text_encoder",
        "vision_encoder",
        "concept_detector",
        "memory_encoder",
        "seed_decoder",
        "seed_memory",
        "tracking_decoder",
        "tracking_memory",
        "memory_attention",
    ]
    parser.add_argument("--graphs", nargs="+", choices=graph_names, default=graph_names)
    parser.add_argument("--fixtures-only", action="store_true", help="Regenerate PyTorch fixtures for existing graphs")
    parser.add_argument("--opset", type=int, default=23, help="ONNX opset (23+ preserves native attention)")
    parser.add_argument(
        "--fixture-image", type=Path, help="Use a real RGB image for graph reference/calibration inputs"
    )
    args = parser.parse_args()
    import torch
    from export_tokenizer import export_tokenizer

    from modeling import (
        ConceptDetector,
        ImagePreprocess,
        MemoryAttention,
        MemoryEncoder,
        SeedDecoder,
        SeedMemory,
        TextEncoder,
        TrackingDecoder,
        TrackingMemory,
        VisionEncoder,
    )

    predictor = build_predictor()
    model = predictor.model
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "metadata.json"
    metadata = {
        "schema_version": 1,
        "opset_version": args.opset,
        "model_family": "sam3.1",
        **LOCK,
        "dtype": "float32",
        "image_size": 1008,
        "backbone_stride": 14,
        "max_objects": 16,
        "multiplex_count": 16,
        "hotstart_delay": 0,
        "graphs": {},
        "capabilities": ["image_segmentation"],
        "reference_tf32": False,
        "preprocessing": {
            "input": "RGB uint8 NHWC",
            "resize": "bilinear antialias",
            "size": [1008, 1008],
            "align_corners": False,
            "round_before_normalize": True,
            "normalization": "x / 127.5 - 1",
            "native_gpu_implementation": "CUDA separable triangle filter",
        },
        "tokenizer": {"file": "tokenizer.json", "fixtures": "tokenizer_tests.json", "context_length": 32},
        "tracking_status": "exported boundaries; native sequence controller not validated",
    }
    metadata["environment"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "onnx", "onnxscript", "ftfy", "regex", "huggingface_hub")
    }
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("source_revision", "checkpoint_revision", "dtype", "schema_version"):
            if previous[key] != metadata[key]:
                raise ValueError(f"Existing export has incompatible {key}; use a new output directory")
        metadata["graphs"] = previous["graphs"]
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        text_encoder = TextEncoder(model.detector.backbone)
        tokenizer = text_encoder.text.tokenizer
        token_ids = tokenizer(["bicycle"], context_length=text_encoder.text.context_length).cuda()
        text = text_encoder(token_ids)
        metadata["text_context_length"] = token_ids.shape[1]
        export_tokenizer(args.output, tokenizer, token_ids.shape[1])
        jobs = {}
        if "image_preprocess" in args.graphs:
            jobs["image_preprocess"] = (
                ImagePreprocess(),
                (torch.zeros(1, 720, 1280, 3, dtype=torch.uint8, device="cuda"),),
                ["image_rgb"],
                ["image"],
            )
        if "text_encoder" in args.graphs:
            jobs["text_encoder"] = (text_encoder, (token_ids,), ["token_ids"], ["language_features", "language_mask"])
        if any(name in args.graphs for name in graph_names if name not in ("image_preprocess", "text_encoder")):
            image = torch.zeros(1, 3, 1008, 1008, device="cuda")
            if args.fixture_image:
                import numpy as np
                from PIL import Image

                rgb = torch.from_numpy(np.array(Image.open(args.fixture_image).convert("RGB"))).cuda().unsqueeze(0)
                image = ImagePreprocess()(rgb)[0]
            vision = VisionEncoder(model.detector.backbone)
            visual = vision(image)
            levels = len(visual) // 6
            vision_names = [
                f"{branch}_{kind}_{i}"
                for branch in ("detector", "interactive", "propagation")
                for kind in ("feature", "position")
                for i in range(levels)
            ]
            metadata["feature_levels"] = levels
            if "vision_encoder" in args.graphs:
                jobs["vision_encoder"] = (vision, (image,), ["image"], vision_names)
            if "concept_detector" in args.graphs:
                jobs["concept_detector"] = (
                    ConceptDetector(model.detector, levels),
                    (*text, *visual[: levels * 2]),
                    ["language_features", "language_mask", *vision_names[: levels * 2]],
                    ["mask_logits", "boxes_xyxy", "class_logits", "presence_logits"],
                )
            if "memory_encoder" in args.graphs:
                tracker = model.tracker.model
                memory = MemoryEncoder(tracker)
                jobs["memory_encoder"] = (
                    memory,
                    (visual[4 * levels + levels - 1], torch.zeros(1, 32, 1008, 1008, device="cuda")),
                    ["image_features", "masks"],
                    ["memory_features", "memory_position"],
                )
            tracker = model.tracker.model
            tracking_config = {}
            for key in (
                "image_size",
                "input_mask_size",
                "low_res_mask_size",
                "num_maskmem",
                "max_cond_frames_in_attn",
                "max_obj_ptrs_in_encoder",
                "memory_temporal_stride_for_eval",
                "use_memory_selection",
                "keep_first_cond_frame",
                "use_maskmem_tpos_v2",
                "add_tpos_enc_to_obj_ptrs",
                "use_signed_tpos_enc_to_obj_ptrs",
                "only_obj_ptrs_in_the_past_for_eval",
                "sigmoid_scale_for_mem_enc",
                "sigmoid_bias_for_mem_enc",
                "apply_sigmoid_to_mask_logits_for_mem_enc",
                "non_overlap_masks_for_mem_enc",
                "condition_as_mask_input_fg",
                "condition_as_mask_input_bg",
                "add_object_conditional_embeddings",
                "add_object_unconditional_embeddings",
                "object_score_logit_threshold",
            ):
                value = getattr(tracker, key, None)
                if isinstance(value, int | float | bool | str) or value is None:
                    tracking_config[key] = value
            for key in (
                "score_threshold_detection",
                "det_nms_thresh",
                "det_nms_use_iom",
                "assoc_iou_thresh",
                "trk_assoc_iou_thresh",
                "new_det_thresh",
                "init_trk_keep_alive",
                "max_trk_keep_alive",
                "min_trk_keep_alive",
                "suppress_overlapping_based_on_recent_occlusion_threshold",
                "suppress_det_close_to_boundary",
                "recondition_every_nth_frame",
                "iou_thresh_recondition",
                "fill_hole_area",
                "sprinkle_removal_area",
            ):
                value = getattr(model, key, None)
                if isinstance(value, int | float | bool | str) or value is None:
                    tracking_config[key] = value
            metadata["tracking_config"] = tracking_config
            projection = tracker.obj_ptr_tpos_proj
            temporal_constants = {
                "schema_version": 1,
                "mask_memory_temporal_embeddings": tracker.maskmem_tpos_enc.detach().cpu().reshape(tracker.num_maskmem, -1).tolist(),
                "pointer_sine_dimension": tracker.hidden_dim if tracker.proj_tpos_enc_in_obj_ptrs else tracker.mem_dim,
                "pointer_projection": type(projection).__name__,
            }
            if isinstance(projection, torch.nn.Linear):
                temporal_constants["pointer_projection_weight"] = projection.weight.detach().cpu().tolist()
                temporal_constants["pointer_projection_bias"] = projection.bias.detach().cpu().tolist() if projection.bias is not None else None
            elif not isinstance(projection, torch.nn.Identity):
                raise ValueError("Unsupported pointer temporal projection")
            (args.output / "temporal_constants.json").write_text(json.dumps(temporal_constants), encoding="utf-8")
            metadata["temporal_constants"] = "temporal_constants.json"
            valid = torch.ones(16, device="cuda")
            interactive = visual[2 * levels : 3 * levels]
            propagation = visual[4 * levels : 5 * levels]
            if "seed_decoder" in args.graphs:
                jobs["seed_decoder"] = (
                    SeedDecoder(tracker),
                    (
                        *interactive,
                        torch.zeros(16, 1, tracker.input_mask_size, tracker.input_mask_size, device="cuda"),
                        valid,
                    ),
                    ["feature_0", "feature_1", "feature_2", "masks", "valid"],
                    ["mask_logits", "object_pointers", "object_logits"],
                )
            if "seed_memory" in args.graphs:
                jobs["seed_memory"] = (
                    SeedMemory(tracker),
                    (
                        propagation[2],
                        torch.zeros(16, 1, tracker.input_mask_size, tracker.input_mask_size, device="cuda"),
                        torch.zeros(16, 1, device="cuda"),
                        valid,
                        valid.clone(),  # Distinct input storage: conditioning is not an alias of validity.
                    ),
                    ["image_features", "seed_masks", "object_logits", "valid", "conditioning"],
                    ["memory_features", "memory_position"],
                )
            if "tracking_decoder" in args.graphs:
                jobs["tracking_decoder"] = (
                    TrackingDecoder(tracker),
                    (propagation[2], *propagation[:2], valid),
                    ["conditioned", "feature_0", "feature_1", "valid"],
                    ["mask_logits", "object_pointers", "object_logits"],
                )
            if "tracking_memory" in args.graphs:
                jobs["tracking_memory"] = (
                    TrackingMemory(tracker),
                    (
                        propagation[2],
                        torch.zeros(16, 1, 288, 288, device="cuda"),
                        torch.zeros(16, 1, device="cuda"),
                        valid,
                        torch.zeros_like(valid),  # Tracking frames are valid but not conditioning frames.
                    ),
                    ["image_features", "mask_logits", "object_logits", "valid", "conditioning"],
                    ["memory_features", "memory_position"],
                )
            if "memory_attention" in args.graphs:
                memories = [torch.zeros(5184 * 2, 1, 256, device="cuda") for _ in range(4)]
                pointers = [torch.zeros(16 * 2, 1, 256, device="cuda") for _ in range(2)]
                jobs["memory_attention"] = (
                    MemoryAttention(tracker),
                    (propagation[2], visual[6 * levels - 1], *memories, *pointers),
                    [
                        "image_features",
                        "image_position",
                        "memory_image",
                        "memory_image_position",
                        "memory",
                        "memory_position",
                        "object_pointers",
                        "pointer_position",
                    ],
                    ["conditioned"],
                )
        for name, (module, inputs, in_names, out_names) in jobs.items():
            start = time.perf_counter()
            contract = export_graph(
                module, inputs, args.output / f"{name}.onnx", in_names, out_names, args.fixtures_only, args.opset
            )
            metadata["graphs"][name] = contract
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Exported {name} in {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
