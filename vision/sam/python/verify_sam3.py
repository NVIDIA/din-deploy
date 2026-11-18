# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare pinned SAM3.1 PyTorch and ORT networks on an image or image sequence."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from sam3_common import add_checkpoint_arguments, build_predictor, checkpoint_options, upstream_metadata
from sam_verify_common import (
    ModelComponent,
    OnnxBundle,
    add_runtime_arguments,
    check_instance_ids,
    compare_instances,
    run_native,
    select_components,
    set_component_backends,
    validate_arguments,
    write_json,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_COMPONENTS = {"image_preprocess", "text_encoder", "vision_encoder", "concept_detector"}
TRACKING_COMPONENTS = IMAGE_COMPONENTS | {
    "seed_decoder",
    "seed_memory",
    "tracking_decoder",
    "tracking_memory",
    "memory_attention",
}
GAPS = [
    "Both component backends use the exported single 16-object bucket and bounded controller, not upstream association/reconditioning.",
    "The upstream video reference uses its official BF16 autocast; graph comparisons use the bundle's dtype.",
    "The full upstream predictor remains a separate quality reference; export/ONNX comparisons share one pipeline.",
]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "family": "sam3",
        "provider": args.provider,
        "passed": False,
        "gaps": GAPS,
        "model_dir": str(args.onnx_dir),
        "device": args.device,
        "thresholds": {"min_iou": args.min_iou, "atol": args.atol, "rtol": args.rtol},
    }
    bundle = None
    try:
        bundle = OnnxBundle(args, "sam3")
        if not args.validate_only:
            args.selected_onnx = select_components(
                args.onnx_components, TRACKING_COMPONENTS if args.frames_dir else IMAGE_COMPONENTS, bundle.graphs
            )
        report["structure"] = bundle.validate(None if args.validate_only else args.selected_onnx)
        if not all(row["passed"] for row in report["structure"].values()):
            raise ValueError("Graph validation failed; see structure results")
        if args.validate_only:
            report["passed"] = True
            report["verification"] = "export fixtures" if args.fixtures else "structure only"
        else:
            if args.frames_dir and bundle.metadata.get("memory_boundary_version") != 2:
                raise ValueError("Tracking requires memory_boundary_version=2; re-export the memory graphs")
            for key, value in upstream_metadata(args).items():
                if bundle.metadata.get(key) != value:
                    raise ValueError(f"Bundle {key} differs from the configured SAM3 source/checkpoint")
            if torch.device(args.device).type != "cuda":
                raise ValueError("The pinned upstream SAM3 predictor requires CUDA, including with --provider cpu")
            predictor = build_predictor(**checkpoint_options(args))
            report["reference_precision"] = "float32" if args.image else "bfloat16"
            report["export_reference_precision"] = bundle.metadata["dtype"]
            report["onnx_precision"] = bundle.metadata["dtype"]
            report["text"] = args.text
            report["records"] = (
                run_image(args, predictor, bundle) if args.image else run_sequence(args, predictor, bundle)
            )
            report["graphs"] = bundle.comparisons
            report["graph_calls"] = bundle.calls
            report["passed"] = bool(report["records"]) and all(row["passed"] for row in report["records"])
            report["passed"] &= all(
                row["passed"] for outputs in bundle.comparisons.values() for row in outputs.values()
            )
    except Exception as error:
        report["error"] = str(error)
    if bundle is not None and not args.validate_only:
        report.setdefault("records", bundle.frame_comparisons)
        if "error" in report:
            report["error_graph"] = bundle.active_graph
        report["graphs"] = bundle.comparisons
        report["component_backends"] = bundle.component_backends
        report["component_calls"] = bundle.component_calls
        report["export_reference_calls"] = bundle.export_reference_calls
        report["graph_calls"] = bundle.calls
        report["unexecuted_graphs"] = sorted(set(bundle.graphs) - set(bundle.calls))
    write_json(args.output / "comparison.json", report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


def run_image(args, predictor, bundle):
    from sam3.model.sam3_image_processor import Sam3Processor

    image = Image.open(args.image).convert("RGB")
    processor = Sam3Processor(predictor.model.detector, device=args.device, confidence_threshold=args.threshold)
    state = processor.set_image(image)
    result = processor.set_text_prompt(args.text, state)
    # In the pinned revision masks_logits contains probabilities, not raw logits.
    reference = save_outputs(
        args.output / "pytorch",
        0,
        np.arange(1, len(result["scores"]) + 1),
        result["masks_logits"],
        result["scores"],
        result["boxes"],
    )
    del state, result
    candidate = Sam3Pipeline(bundle, predictor.model)
    export = candidate.run_frame(image, tracking=False, frame_idx=0, length=1)
    export_masks = save_outputs(args.output / "export", 0, *export)
    del export
    bundle.export_reference_calls = bundle.component_calls
    candidate.reset(args.selected_onnx)
    actual = candidate.run_frame(image, tracking=False, frame_idx=0, length=1)
    save_outputs(args.output / "candidate", 0, *actual)
    row = compare_instances(reference, actual[1].detach().float().cpu().numpy()[:, 0] > 0.5, args.min_iou)
    row["export"] = compare_instances(export_masks, actual[1].detach().float().cpu().numpy()[:, 0] > 0.5, args.min_iou)
    row["passed"] &= row["export"]["passed"]
    row["frame_idx"] = 0
    bundle.frame_comparisons.append(row)
    if args.din_sam:
        native = run_native(args, "sam3", args.image, text=args.text)
        compare_native(row, native, 0, args.output / "candidate", args.min_iou)
    return [row]


def run_sequence(args, predictor, bundle):
    frames = sorted(path for path in args.frames_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit_frames is not None:
        frames = frames[: args.limit_frames]
    if not frames:
        raise ValueError("No image frames found")
    # Stage exactly the selected frames for both the upstream session and native CLI.
    staged = args.output / "frames"
    if staged.exists():
        raise FileExistsError(f"Use a fresh --output directory: {staged}")
    staged.mkdir()
    geometry = None
    for index, path in enumerate(frames):
        image = Image.open(path).convert("RGB")
        if geometry is not None and image.size != geometry:
            raise ValueError("C++ tracking requires constant frame geometry")
        geometry = image.size
        image.save(staged / f"{index:06d}.png")
    reference = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        session = start_video_session(predictor, staged)
        session_id = session["session_id"]
        try:
            predictor.handle_request(
                {"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": args.text}
            )
            for result in predictor.handle_stream_request(
                    {"type": "propagate_in_video", "session_id": session_id, "propagation_direction": "forward"}
            ):
                output, index = result["outputs"], int(result["frame_index"])
                # The upstream video API returns binary masks, not probabilities.
                reference[index] = save_outputs(
                    args.output / "pytorch",
                    index,
                    np.asarray(output["out_obj_ids"]) + 1,
                    output["out_binary_masks"],
                    output["out_probs"],
                )
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
    if set(reference) != set(range(len(frames))):
        raise ValueError("Upstream reference did not produce every selected frame")
    candidate = Sam3Pipeline(bundle, predictor.model)
    export_reference = {}
    for index, path in enumerate(sorted(staged.glob("*.png"))):
        export = candidate.run_frame(
            Image.open(path).convert("RGB"), tracking=True, frame_idx=index, length=len(frames)
        )
        export_reference[index] = (export[0], save_outputs(args.output / "export", index, *export))
    del export
    bundle.export_reference_calls = bundle.component_calls
    candidate.reset(args.selected_onnx)
    records = bundle.frame_comparisons
    identity_mapping = {}
    export_mapping = {}
    for index, path in enumerate(sorted(staged.glob("*.png"))):
        actual = candidate.run_frame(
            Image.open(path).convert("RGB"), tracking=True, frame_idx=index, length=len(frames)
        )
        save_outputs(args.output / "candidate", index, *actual)
        row = compare_instances(reference[index], actual[1].detach().float().cpu().numpy()[:, 0] > 0.5, args.min_iou)
        reference_records = json.loads((args.output / "pytorch" / f"{index:06d}.json").read_text(encoding="utf-8"))
        check_instance_ids(row, [item["id"] for item in reference_records], actual[0], identity_mapping, args.min_iou)
        export_ids, export_masks = export_reference[index]
        row["export"] = compare_instances(
            export_masks, actual[1].detach().float().cpu().numpy()[:, 0] > 0.5, args.min_iou
        )
        check_instance_ids(row["export"], export_ids, actual[0], export_mapping, args.min_iou)
        row["passed"] &= row["export"]["passed"]
        records.append({"frame_idx": index, **row})
    if args.din_sam:
        native = run_native(args, "sam3", staged, text=args.text, tracking=True)
        native_mapping = {}
        for index, row in enumerate(records):
            compare_native(row, native, index, args.output / "candidate", args.min_iou, native_mapping)
    return records


def start_video_session(predictor, resource):
    """Bridge the pinned session wrapper's newer default kwargs to its multiplex model.

    Only unsupported default offload/loader arguments are adapted. Do not catch
    arbitrary TypeErrors or alter inference settings to hide upstream failures.
    """
    model = predictor.model
    original = model.init_state
    supported = inspect.signature(original).parameters
    previous_override = model.__dict__.get("init_state")

    def init_state(*args, **kwargs):
        if "offload_state_to_cpu" not in supported and kwargs.pop("offload_state_to_cpu", False):
            raise ValueError("The pinned multiplex model does not support state offloading")
        if "video_loader_type" not in supported and "video_loader_type" in kwargs:
            loader = kwargs.pop("video_loader_type")
            if loader != "cv2":
                raise ValueError(f"Unsupported pinned video loader: {loader}")
            if "use_cv2" in supported:
                kwargs["use_cv2"] = True
        return original(*args, **kwargs)

    model.init_state = init_state
    try:
        return predictor.handle_request({"type": "start_session", "resource_path": str(resource.resolve())})
    finally:
        if previous_override is None:
            del model.init_state
        else:
            model.init_state = previous_override


def as_numpy(value):
    return value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def save_outputs(output, frame, ids, probabilities, scores, boxes=None):
    """Save the same instance masks, label maps and combined masks as the SAM3 reference."""
    output.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(as_numpy(ids), dtype=np.int64)
    probabilities = np.asarray(as_numpy(probabilities), dtype=np.float32)
    if probabilities.ndim == 4:
        probabilities = probabilities[:, 0]
    scores = as_numpy(scores).reshape(-1)
    if probabilities.ndim != 3 or len(ids) != len(probabilities) or len(ids) != len(scores):
        raise ValueError("Inconsistent instance outputs")
    if len(ids) > 16 or len(set(ids.tolist())) != len(ids) or np.any(ids <= 0):
        raise ValueError("Expected up to sixteen unique positive instance IDs")
    if not np.isfinite(probabilities).all() or not np.isfinite(scores).all():
        raise ValueError("Non-finite instance output")
    order = np.argsort(ids, kind="stable")
    ids, probabilities, scores = ids[order], probabilities[order], scores[order]
    labels = np.zeros(probabilities.shape[-2:], dtype=np.uint32)
    union = np.zeros_like(labels, dtype=bool)
    if len(ids):
        best = probabilities.argmax(axis=0)
        union = probabilities.max(axis=0) > 0.5
        labels[union] = ids[best[union]]
    prefix = f"{frame:06d}"
    np.save(output / f"{prefix}_labels.npy", labels)
    np.savez(output / f"{prefix}_instances.npz", ids=ids, probabilities=probabilities, scores=scores)
    Image.fromarray(union.astype(np.uint8) * 255).save(output / f"{prefix}_combined.png")
    records = []
    for index, obj_id in enumerate(ids):
        filename = f"{prefix}_id{obj_id}.png"
        Image.fromarray((probabilities[index] > 0.5).astype(np.uint8) * 255).save(output / filename)
        row = {"id": int(obj_id), "score": float(scores[index]), "mask": filename}
        if boxes is not None:
            row["box"] = as_numpy(boxes)[order[index]].tolist()
        records.append(row)
    write_json(output / f"{prefix}.json", records)
    return probabilities > 0.5


def compare_native(row, native, frame, candidate, threshold, mapping=None):
    def load_masks(directory):
        # Native and Python writers both use six-digit frame/instance filenames.
        paths = sorted(directory.glob(f"{frame:06d}_id*.png"))
        if not paths:
            raise FileNotFoundError(f"No instance masks for frame {frame:06d} in {directory}")
        masks = [np.asarray(Image.open(path).convert("L")) > 0 for path in paths]
        return masks, [int(path.stem.rsplit("_id", 1)[1]) for path in paths]

    reference_masks, reference_ids = load_masks(candidate)
    actual_masks, actual_ids = load_masks(native)
    row["native"] = compare_instances(reference_masks, actual_masks, threshold)
    check_instance_ids(row["native"], reference_ids, actual_ids, {} if mapping is None else mapping, threshold)
    row["passed"] &= row["native"]["passed"]


class Sam3Pipeline:
    """One pipeline of exporter modeling components, each with an export or ONNX backend."""

    def __init__(self, bundle, model):
        from sam3_modeling import (
            ConceptDetector,
            ImagePreprocess,
            MemoryAttention,
            SeedDecoder,
            SeedMemory,
            TextEncoder,
            TrackingDecoder,
            TrackingMemory,
            VisionEncoder,
        )

        self.bundle = bundle
        self.device, self.dtype = torch.device(bundle.args.device), getattr(torch, bundle.metadata["dtype"])
        # Keep complex rotary buffers intact, matching the exporter.
        {torch.float32: model.float, torch.float16: model.half, torch.bfloat16: model.bfloat16}[self.dtype]()
        tracker = model.tracker.model
        levels = bundle.metadata.get("feature_levels", 3)
        visual_names = tuple(
            f"{branch}_{kind}_{i}"
            for branch in ("detector", "interactive", "propagation")
            for kind in ("feature", "position")
            for i in range(levels)
        )
        specs = {
            "image_preprocess": (ImagePreprocess(self.dtype), ("image",)),
            "text_encoder": (TextEncoder(model.detector.backbone), ("language_features", "language_mask")),
            "vision_encoder": (VisionEncoder(model.detector.backbone), visual_names),
            "concept_detector": (
                ConceptDetector(model.detector, levels),
                ("mask_logits", "boxes_xyxy", "class_logits", "presence_logits"),
            ),
            "seed_decoder": (SeedDecoder(tracker), ("mask_logits", "object_pointers", "object_logits")),
            "tracking_decoder": (TrackingDecoder(tracker), ("mask_logits", "object_pointers", "object_logits")),
            "seed_memory": (SeedMemory(tracker), ("memory_features", "memory_position")),
            "tracking_memory": (TrackingMemory(tracker), ("memory_features", "memory_position")),
            "memory_attention": (MemoryAttention(tracker), ("conditioned",)),
        }
        self.components = {
            name: ModelComponent(
                bundle,
                name,
                module,
                outputs,
                ("language_features", "language_mask", *visual_names[: 2 * levels])
                if name == "concept_detector"
                else None,
            )
            for name, (module, outputs) in specs.items()
        }
        self.temporal = tracker.maskmem_tpos_enc.detach().reshape(tracker.num_maskmem, -1)
        self.pointer_projection = tracker.obj_ptr_tpos_proj
        self.reset()

    def reset(self, onnx_components=()):
        self.text = None
        self.history = []
        self.valid = torch.zeros(16, device=self.device, dtype=self.dtype)
        set_component_backends(self.bundle, self.components, onnx_components)

    def network(self, name, values):
        return self.components[name](values)

    def encode(self, image):
        rgb = torch.from_numpy(np.array(image, copy=True)).unsqueeze(0).to(self.device)
        tensor = self.network("image_preprocess", {"image_rgb": rgb})["image"]
        return self.network("vision_encoder", {"image": tensor})

    def detect(self, visual, tracking):
        if self.text is None:
            encoder = self.components["text_encoder"].module.text
            tokens = encoder.tokenizer(
                [self.bundle.args.text], context_length=self.bundle.metadata["text_context_length"]
            )
            self.text = self.network("text_encoder", {"token_ids": tokens.to(self.device)})
        values = dict(self.text)
        values.update({key: value for key, value in visual.items() if key.startswith("detector_")})
        output = self.network("concept_detector", values)
        scores = output["class_logits"].float().sigmoid().reshape(-1) * output[
            "presence_logits"
        ].float().sigmoid().reshape(())
        selected = torch.where(scores > max(0.65 if tracking else 0.0, self.bundle.args.threshold))[0]
        if len(selected) > 16:
            raise ValueError(f"Object capacity exceeded: {len(selected)} > 16")
        masks = output["mask_logits"].reshape(-1, 1, 288, 288)[selected]
        return masks, scores[selected], output["boxes_xyxy"].reshape(-1, 4)[selected]

    def pack_memory(self, visual, frame_idx, length):
        spatial = self.history if len(self.history) <= 7 else [self.history[0], *self.history[-6:]]
        values = {"image_features": visual["propagation_feature_2"], "image_position": visual["propagation_position_2"]}
        for name, key in (
                ("memory_image", "image"),
                ("memory_image_position", "position"),
                ("memory", "memory"),
                ("memory_position", "memory_position"),
        ):
            pieces = []
            for index, entry in enumerate(spatial):
                value = entry[key]
                age = frame_idx - entry["frame_idx"]
                temporal_index = 6 if age >= 7 else 6 - age if index == 0 else age - 1
                if name.endswith("position"):
                    value = value + self.temporal[temporal_index][None, :, None, None]
                pieces.append(value.flatten(2).permute(2, 0, 1))
            values[name] = torch.cat(pieces)
        pointers, positions = [], []
        for entry in self.history:
            distance = (frame_idx - entry["frame_idx"]) / max(1, min(16, length) - 1)
            dimensions = torch.arange(256, device=self.device)
            phase = distance / (10000.0 ** (2 * ((dimensions % 128) // 2).float() / 128))
            sine = torch.where(dimensions < 128, phase.sin(), phase.cos()).to(self.dtype)
            position = self.pointer_projection(sine).reshape(1, 1, 256).expand(16, 1, 256)
            pointers.append(entry["pointers"].reshape(16, 1, 256))
            positions.append(position)
        values["object_pointers"], values["pointer_position"] = torch.cat(pointers), torch.cat(positions)
        return values

    def run_frame(self, image, *, tracking, frame_idx, length):
        visual = self.encode(image)
        boxes = None
        if not tracking:
            masks, scores, boxes = self.detect(visual, False)
            ids = np.arange(1, len(masks) + 1)
            probabilities = F.interpolate(
                masks.float(), (image.height, image.width), mode="bilinear", align_corners=False
            ).sigmoid()
            scale = boxes.new_tensor([image.width, image.height, image.width, image.height])
            return ids, probabilities, scores, boxes * scale
        seed = not self.history
        if seed:
            masks, _, _ = self.detect(visual, True)
            count = len(masks)
            if not count:
                return (
                    np.array([], dtype=np.int64),
                    torch.empty(0, 1, image.height, image.width, device=self.device),
                    torch.empty(0, device=self.device),
                )
            self.valid[:count] = 1
            padded = masks.new_full((16, 1, 288, 288), -1024)
            padded[:count] = masks
            seed_masks = (F.interpolate(padded.float(), (1152, 1152), mode="bilinear", align_corners=False) > 0).to(
                self.dtype
            )
            decoded = self.network(
                "seed_decoder",
                {
                    **{f"feature_{i}": visual[f"interactive_feature_{i}"] for i in range(3)},
                    "masks": seed_masks,
                    "valid": self.valid,
                },
            )
            high = F.interpolate(decoded["mask_logits"].float(), (1008, 1008), mode="bilinear", align_corners=False)
            high = torch.where(self.valid[:, None, None, None] > 0, high, -1024)
            winner = high.argmax(dim=0, keepdim=True)
            slots = torch.arange(16, device=self.device)[:, None, None, None]
            memory_masks = torch.where(slots == winner, high, high.clamp(max=-10))
        else:
            conditioned = self.network("memory_attention", self.pack_memory(visual, frame_idx, length))["conditioned"]
            decoded = self.network(
                "tracking_decoder",
                {
                    "conditioned": conditioned,
                    "feature_0": visual["propagation_feature_0"],
                    "feature_1": visual["propagation_feature_1"],
                    "valid": self.valid,
                },
            )
            memory_masks = decoded["mask_logits"]
        encoded = self.network(
            "seed_memory" if seed else "tracking_memory",
            {
                "image_features": visual["propagation_feature_2"],
                "mask_logits": memory_masks,
                "object_logits": decoded["object_logits"],
                "valid": self.valid,
                "conditioning": self.valid if seed else torch.zeros_like(self.valid),
            },
        )
        self.history.append(
            {
                "frame_idx": frame_idx,
                "image": visual["propagation_feature_2"],
                "position": visual["propagation_position_2"],
                "memory": encoded["memory_features"],
                "memory_position": encoded["memory_position"],
                "pointers": decoded["object_pointers"] * self.valid[:, None],
            }
        )
        if len(self.history) > 16:
            del self.history[1]
        active = (self.valid > 0) & (decoded["object_logits"].reshape(-1) > 0)
        ids = torch.where(active)[0].cpu().numpy() + 1
        logits = decoded["mask_logits"][active].float()
        probabilities = F.interpolate(
            logits, (image.height, image.width), mode="bilinear", align_corners=False
        ).sigmoid()
        return ids, probabilities, decoded["object_logits"].reshape(-1)[active].float().sigmoid()


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Components: " + ", ".join(sorted(TRACKING_COMPONENTS))
    )
    add_runtime_arguments(parser)
    add_checkpoint_arguments(parser)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--text")
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    validate_arguments(parser, args)
    if not args.validate_only and (
            (args.image is None) == (args.frames_dir is None) or not args.text or not args.text.strip()
    ):
        parser.error("Specify --text and exactly one of --image or --frames-dir")
    if args.limit_frames is not None and args.limit_frames <= 0 or not 0 <= args.threshold <= 1:
        parser.error("--limit-frames must be positive and --threshold must be in [0, 1]")
    return args


if __name__ == "__main__":
    with torch.inference_mode():
        main()
