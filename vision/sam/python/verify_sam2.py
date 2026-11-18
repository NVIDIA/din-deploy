# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam_verify_common import (
    ModelComponent,
    OnnxBundle,
    add_runtime_arguments,
    mask_iou,
    run_native,
    select_components,
    set_component_backends,
    validate_arguments,
    write_json,
)

IMAGE_COMPONENTS = {"image_preprocess", "image_encoder", "sam2_decoder"}
TRACKING_COMPONENTS = IMAGE_COMPONENTS | {"sam2_decoder_propagate", "memory_attention", "memory_encoder"}

GAPS = [
    "The exported decoder is single-mask and has no IoU predictions or object pointers.",
    "Both component backends use the C++ seven-slot memory ring without pointer attention or upstream memory selection.",
    "The full upstream predictor remains a separate quality reference; export/ONNX comparisons share one pipeline.",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PromptSet:
    obj_id: int
    frame_idx: int
    points: np.ndarray
    labels: np.ndarray
    box: np.ndarray | None
    multimask_output: bool


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "family": "sam2",
        "provider": args.provider,
        "passed": False,
        "gaps": GAPS,
        "model_dir": str(args.onnx_dir),
        "device": args.device,
        "thresholds": {"min_iou": args.min_iou, "atol": args.atol, "rtol": args.rtol},
    }
    bundle = None
    try:
        bundle = OnnxBundle(args, "sam2")
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
            args.model = args.model or bundle.metadata["model"]
            if args.model != bundle.metadata["model"]:
                raise ValueError("--model must match the ONNX bundle's model")
            prompts = load_prompt_sets(args.prompts, default_multimask_output=args.multimask_output)
            if not prompts:
                raise ValueError("At least one prompt is required")
            if any(prompt.multimask_output for prompt in prompts):
                raise ValueError(
                    "Current ONNX decoders export multimask_output=False; multimask verification needs an export change"
                )
            if args.frames_dir and (len(prompts) != 1 or prompts[0].frame_idx != 0):
                raise ValueError(
                    "The C++ tracking contract supports one object prompted at frame 0; later prompts require controller changes"
                )
            if args.din_sam and (len(prompts) != 1 or args.image and prompts[0].frame_idx != 0):
                raise ValueError("--din-sam supports one prompt at frame 0")
            report["model"] = args.model
            report["reference_precision"] = "float32"
            report["export_reference_precision"] = bundle.metadata["dtype"]
            report["onnx_precision"] = bundle.metadata["dtype"]
            report["records"] = run_image(args, prompts, bundle) if args.image else run_sequence(args, prompts, bundle)
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


def run_image(args: argparse.Namespace, prompts: list[PromptSet], bundle) -> list[dict]:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image = np.array(Image.open(args.image).convert("RGB"), copy=True)
    predictor = SAM2ImagePredictor.from_pretrained(args.model, device=args.device)
    predictor.set_image(image)
    records = []
    for index, prompt in enumerate(prompts):
        masks, ious, low_res_masks = predictor.predict(
            point_coords=prompt.points if len(prompt.points) else None,
            point_labels=prompt.labels if len(prompt.labels) else None,
            box=prompt.box,
            multimask_output=prompt.multimask_output,
            return_logits=True,
            normalize_coords=True,
        )
        best_index = int(np.argmax(ious))
        mask_path = args.output / "pytorch" / f"image_obj{prompt.obj_id}_mask{index}.png"
        logits_path = args.output / "pytorch" / f"image_obj{prompt.obj_id}_low_res{index}.npy"
        save_mask_png(masks[best_index], mask_path)
        np.save(logits_path, low_res_masks[best_index])
        records.append(
            {
                "kind": "image",
                "image": str(args.image),
                "prompt": {
                    "obj_id": prompt.obj_id,
                    "frame_idx": prompt.frame_idx,
                    "points": prompt.points.tolist(),
                    "labels": prompt.labels.tolist(),
                    "box": None if prompt.box is None else prompt.box.tolist(),
                    "multimask_output": prompt.multimask_output,
                },
                "best_mask_index": best_index,
                "iou_predictions": ious.tolist(),
                "mask_path": str(mask_path),
                "low_res_logits_path": str(logits_path),
            }
        )
    (args.output / "reference_outputs.json").write_text(
        json.dumps({"model": args.model, "records": records}, indent=2), encoding="utf-8"
    )
    candidate = Sam2Pipeline(bundle, predictor.model, predictor._transforms)
    encoded = candidate.encode(image)
    for prompt, reference in zip(prompts, records, strict=True):
        logits, _ = candidate.decode(encoded, prompt, image.shape[:2])
        save_mask_png(logits.float().cpu().numpy(), args.output / "export" / Path(reference["mask_path"]).name)
    bundle.export_reference_calls = bundle.component_calls
    candidate.reset(args.selected_onnx)
    encoded = candidate.encode(image)
    comparisons = bundle.frame_comparisons
    for index, (prompt, reference) in enumerate(zip(prompts, records, strict=True)):
        logits, _ = candidate.decode(encoded, prompt, image.shape[:2])
        comparisons.append(
            compare_mask(
                args,
                Path(reference["mask_path"]),
                logits,
                args.output / "candidate" / Path(reference["mask_path"]).name,
                index,
                prompt.obj_id,
                args.output / "export" / Path(reference["mask_path"]).name,
            )
        )
    if args.din_sam:
        native_prompt = args.output / "native_prompt.json"
        write_json(native_prompt, prompt_record(prompts[0], bundle.metadata["max_points"]))
        native = run_native(args, "sam2", args.image, prompt=native_prompt)
        compare_native(comparisons[0], native / "000000_id1.png", args.min_iou)
    return comparisons


def run_sequence(args: argparse.Namespace, prompts: list[PromptSet], bundle) -> list[dict]:
    from sam2.build_sam import build_sam2_video_predictor_hf

    predictor = build_sam2_video_predictor_hf(args.model, device=args.device).eval()
    frames = sorted(path for path in args.frames_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not frames:
        raise FileNotFoundError(f"No image frames found in {args.frames_dir}")
    if args.limit_frames is not None:
        frames = frames[: args.limit_frames]
    if any(prompt.frame_idx >= len(frames) for prompt in prompts):
        raise ValueError("Prompt frame is outside the selected sequence")
    staged_frames_dir = stage_frames_for_sam2(frames, args.output / "_sam2_frames", args.limit_frames)
    inference_state = predictor.init_state(video_path=str(staged_frames_dir))
    for prompt in sorted(prompts, key=lambda item: (item.frame_idx, item.obj_id)):
        predictor.add_new_points_or_box(
            inference_state,
            frame_idx=prompt.frame_idx,
            obj_id=prompt.obj_id,
            points=prompt.points if len(prompt.points) else None,
            labels=prompt.labels if len(prompt.labels) else None,
            box=prompt.box,
            normalize_coords=True,
        )

    records = []
    for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
            inference_state,
            max_frame_num_to_track=args.limit_frames,
    ):
        masks = video_res_masks.detach().cpu()
        for obj_index, obj_id in enumerate(obj_ids):
            mask_path = args.output / "pytorch" / f"frame_{frame_idx:06d}_obj{obj_id}.png"
            save_mask_png(masks[obj_index].numpy(), mask_path)
            records.append(
                {
                    "kind": "sequence",
                    "frame_idx": int(frame_idx),
                    "frame_path": str(frames[frame_idx]) if frame_idx < len(frames) else None,
                    "obj_id": int(obj_id),
                    "mask_path": str(mask_path),
                }
            )
    payload = {
        "model": args.model,
        "frames_dir": str(args.frames_dir),
        "prompts": [
            {
                "obj_id": prompt.obj_id,
                "frame_idx": prompt.frame_idx,
                "points": prompt.points.tolist(),
                "labels": prompt.labels.tolist(),
                "box": None if prompt.box is None else prompt.box.tolist(),
                "multimask_output": prompt.multimask_output,
            }
            for prompt in prompts
        ],
        "records": records,
    }
    (args.output / "reference_outputs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    from sam2.utils.transforms import SAM2Transforms

    candidate = Sam2Pipeline(bundle, predictor, SAM2Transforms(resolution=predictor.image_size, mask_threshold=0.0))
    # Decode the same JPEG pixels as the upstream video loader, not the pre-conversion originals.
    candidate_frames = sorted(staged_frames_dir.glob("*.jpg"))
    by_frame = {record["frame_idx"]: record for record in records}
    if set(by_frame) != set(range(len(frames))):
        raise ValueError("Upstream reference did not produce every selected frame")
    geometry = None
    for index, frame in enumerate(candidate_frames):
        image = np.array(Image.open(frame).convert("RGB"), copy=True)
        if geometry is not None and geometry != image.shape:
            raise ValueError("C++ tracking requires constant frame geometry")
        geometry = image.shape
        logits, _ = candidate.decode(candidate.encode(image), prompts[0], image.shape[:2], tracking=True)
        save_mask_png(logits.float().cpu().numpy(), args.output / "export" / Path(by_frame[index]["mask_path"]).name)
    bundle.export_reference_calls = bundle.component_calls
    candidate.reset(args.selected_onnx)
    comparisons = bundle.frame_comparisons
    geometry = None
    for index, frame in enumerate(candidate_frames):
        image = np.array(Image.open(frame).convert("RGB"), copy=True)
        if geometry is not None and geometry != image.shape:
            raise ValueError("C++ tracking requires constant frame geometry")
        geometry = image.shape
        logits, _ = candidate.decode(candidate.encode(image), prompts[0], image.shape[:2], tracking=True)
        reference = Path(by_frame[index]["mask_path"])
        comparisons.append(
            compare_mask(
                args,
                reference,
                logits,
                args.output / "candidate" / reference.name,
                index,
                prompts[0].obj_id,
                args.output / "export" / reference.name,
            )
        )
    if args.din_sam:
        native_prompt = args.output / "native_prompt.json"
        write_json(native_prompt, prompt_record(prompts[0], bundle.metadata["max_points"]))
        native = run_native(
            args,
            "sam2",
            stage_native_frames(candidate_frames, args.output / "native_frames"),
            prompt=native_prompt,
            tracking=True,
        )
        for index, row in enumerate(comparisons):
            compare_native(row, native / f"{index:06d}_id1.png", args.min_iou)
    return comparisons


def prompt_record(prompt, max_points):
    count = len(prompt.points) + (2 if prompt.box is not None else 0)
    if count > max_points:
        raise ValueError(f"Prompt uses {count} points, but the bundle supports {max_points}")
    return {
        "points": prompt.points.tolist() + [[0, 0]] * (max_points - count),
        "labels": prompt.labels.tolist() + [-1] * (max_points - count),
        "box": None if prompt.box is None else prompt.box.tolist(),
    }


def stage_native_frames(frames, output):
    output.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.open(frame).convert("RGB").save(output / f"{index:06d}.png")
    return output


def compare_native(row, path, threshold):
    actual = np.asarray(Image.open(path).convert("L")) > 0
    expected = np.asarray(Image.open(row["candidate_mask"]).convert("L")) > 0
    row["native_iou"] = mask_iou(expected, actual)
    row["passed"] &= row["native_iou"] >= threshold


def compare_mask(args, reference, logits, output, frame_idx, obj_id, export_reference):
    save_mask_png(logits.detach().float().cpu().numpy(), output)
    np.save(output.with_suffix(".npy"), logits.detach().float().cpu().numpy())
    first = np.asarray(Image.open(reference).convert("L")) > 0
    second = np.asarray(Image.open(output).convert("L")) > 0
    iou = mask_iou(first, second)
    export_iou = mask_iou(np.asarray(Image.open(export_reference).convert("L")) > 0, second)
    return {
        "frame_idx": frame_idx,
        "obj_id": obj_id,
        "reference_mask": str(reference),
        "candidate_mask": str(output),
        "iou": iou,
        "export_reference_mask": str(export_reference),
        "export_iou": export_iou,
        "passed": iou >= args.min_iou and export_iou >= args.min_iou,
    }


class Sam2Pipeline:
    """One pipeline of exporter modeling components, each with an export or ONNX backend.

    The full upstream predictor runs independently above. Its missing IoU/pointer
    outputs are never fabricated to make the current graphs look interchangeable.
    """

    def __init__(self, bundle, model, transforms):
        from sam2_modeling import (
            SAM2ImageEncoder,
            SAM2ImagePreprocessor,
            SAM2MaskDecoder,
            SAM2MemoryAttention,
            SAM2MemoryEncoder,
            patch_sam2_fpn_neck_for_fp16_onnx,
            patch_sam2_rope_for_onnx,
        )

        self.bundle, self.transforms = bundle, transforms
        self.dtype = getattr(torch, bundle.metadata["dtype"])
        self.device = torch.device(bundle.args.device)
        # Module precision helpers preserve complex rotary buffers, unlike .to(dtype=...).
        {torch.float32: model.float, torch.float16: model.half, torch.bfloat16: model.bfloat16}[self.dtype]()
        patch_sam2_rope_for_onnx(model)
        if self.dtype == torch.float16:
            patch_sam2_fpn_neck_for_fp16_onnx()
        specs = {
            "image_preprocess": (SAM2ImagePreprocessor(model.image_size, self.dtype).to(self.device), ("image",)),
            "image_encoder": (
                SAM2ImageEncoder(model),
                ("image_features_0", "image_features_1", "image_embeddings", "vision_feat", "vision_pos"),
            ),
            "sam2_decoder": (SAM2MaskDecoder(model), ("masks", "low_res_masks", "object_score_logits")),
            "sam2_decoder_propagate": (
                SAM2MaskDecoder(model, token_embeddings=True),
                ("masks", "low_res_masks", "object_score_logits"),
            ),
            "memory_attention": (SAM2MemoryAttention(model), ("conditioned_image_embeddings",)),
            "memory_encoder": (SAM2MemoryEncoder(model), ("memory_features", "memory_pos")),
        }
        self.components = {
            name: ModelComponent(bundle, name, module, outputs) for name, (module, outputs) in specs.items()
        }
        self.reset()

    def reset(self, onnx_components=()):
        self.history = []
        self.next_slot = 0
        set_component_backends(self.bundle, self.components, onnx_components)

    def network(self, name, values):
        return self.components[name](values)

    def encode(self, image):
        rgb = torch.from_numpy(np.array(image, copy=True)).unsqueeze(0).to(self.device)
        tensor = self.network("image_preprocess", {"image_rgb": rgb})["image"]
        return self.network("image_encoder", {"image": tensor})

    def decode(self, encoded, prompt, original_hw, tracking=False):
        seed = not tracking or not self.history
        record = (
            prompt_record(prompt, self.bundle.metadata["max_points"])
            if seed
            else {"points": [[0, 0]], "labels": [-1], "box": None}
        )
        points, labels = list(record["points"]), list(record["labels"])
        if record["box"] is not None:
            points = np.asarray(record["box"]).reshape(2, 2).tolist() + points
            labels = [2, 3] + labels
        coords = torch.tensor(points, device=self.device, dtype=self.dtype)[None]
        coords *= coords.new_tensor([1024 / original_hw[1], 1024 / original_hw[0]])
        embedding = encoded["image_embeddings"]
        if not seed:
            memory = torch.cat([entry[0] for entry in self.history])
            positions = torch.cat([entry[1] for entry in self.history])
            embedding = self.network(
                "memory_attention",
                {
                    "vision_feat": encoded["vision_feat"],
                    "vision_pos": encoded["vision_pos"],
                    "memory": memory,
                    "memory_pos": positions,
                },
            )["conditioned_image_embeddings"]
        decoded = self.network(
            "sam2_decoder" if seed else "sam2_decoder_propagate",
            {
                "image_features_0": encoded["image_features_0"],
                "image_features_1": encoded["image_features_1"],
                "image_embeddings": embedding,
                "point_coords": coords,
                "point_labels": torch.tensor(labels, device=self.device, dtype=torch.int32)[None],
                "input_masks": torch.zeros(1, 1, 256, 256, device=self.device, dtype=self.dtype),
                "has_input_masks": torch.zeros(1, 1, 1, 1, device=self.device, dtype=self.dtype),
            },
        )
        if tracking:
            memory = self.network(
                "memory_encoder",
                {
                    "vision_feat": encoded["vision_feat"],
                    "pred_masks_high_res": decoded["masks"],
                    "object_score_logits": decoded["object_score_logits"],
                    "is_mask_from_pts": torch.tensor(float(seed), device=self.device, dtype=self.dtype),
                },
            )
            entry = (memory["memory_features"], memory["memory_pos"])
            if len(self.history) < 7:
                self.history.append(entry)
            else:
                self.history[self.next_slot] = entry
            self.next_slot = (self.next_slot + 1) % 7
        logits = self.transforms.postprocess_masks(decoded["low_res_masks"].float(), original_hw)
        return logits[0, 0], decoded["object_score_logits"]


def load_prompt_sets(path: Path, *, default_multimask_output: bool = False) -> list[PromptSet]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("prompts", [payload])
    prompts: list[PromptSet] = []
    for index, record in enumerate(records):
        points = np.asarray(record.get("points", []), dtype=np.float32).reshape(-1, 2)
        labels = np.asarray(record.get("labels", []), dtype=np.int32).reshape(-1)
        if len(points) != len(labels):
            raise ValueError(f"Prompt {index} has {len(points)} points but {len(labels)} labels")
        box_value = record.get("box")
        box = None if box_value is None else np.asarray(box_value, dtype=np.float32).reshape(4)
        if len(points) == 0 and box is None:
            raise ValueError(f"Prompt {index} needs points or box")
        prompts.append(
            PromptSet(
                obj_id=int(record.get("obj_id", 1)),
                frame_idx=int(record.get("frame_idx", 0)),
                points=points,
                labels=labels,
                box=box,
                multimask_output=bool(record.get("multimask_output", default_multimask_output)),
            )
        )
    return prompts


def stage_frames_for_sam2(frames: list[Path], output: Path, limit_frames: int | None) -> Path:
    if output.exists():
        raise FileExistsError(f"Use a fresh --output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selected_frames = frames[:limit_frames] if limit_frames is not None else frames
    for index, frame in enumerate(selected_frames):
        staged = output / f"{index:05d}.jpg"
        # The upstream video reader requires JPEG. Convert pixels rather than
        # renaming PNG/WebP data to .jpg (which can fail in its accelerated loader).
        Image.open(frame).convert("RGB").save(staged, quality=100, subsampling=0)
    return output


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_2d = np.squeeze(mask)
    image = mask_2d.astype(np.uint8) * 255 if mask_2d.dtype == np.bool_ else (mask_2d > 0).astype(np.uint8) * 255
    Image.fromarray(image).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SAM2 export/PyTorch and selected ONNX components.",
        epilog="Components: " + ", ".join(sorted(TRACKING_COMPONENTS)),
    )
    add_runtime_arguments(parser)
    parser.add_argument("--model", help="Defaults to the model recorded in the ONNX bundle")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--multimask-output", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    validate_arguments(parser, args)
    if not args.validate_only and ((args.image is None) == (args.frames_dir is None) or not args.prompts):
        parser.error("Specify --prompts and exactly one of --image or --frames-dir")
    if args.limit_frames is not None and args.limit_frames <= 0:
        parser.error("--limit-frames must be positive")
    return args


if __name__ == "__main__":
    with torch.inference_mode():
        main()
