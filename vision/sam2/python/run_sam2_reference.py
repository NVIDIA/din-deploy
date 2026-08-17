# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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
    prompts = load_prompt_sets(args.prompts, default_multimask_output=args.multimask_output)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.image is not None:
        run_image(args, prompts)
    else:
        run_sequence(args, prompts)


def run_image(args: argparse.Namespace, prompts: list[PromptSet]) -> None:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image = np.asarray(Image.open(args.image).convert("RGB"))
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
        mask_path = args.output / f"image_obj{prompt.obj_id}_mask{index}.png"
        logits_path = args.output / f"image_obj{prompt.obj_id}_low_res{index}.npy"
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


def run_sequence(args: argparse.Namespace, prompts: list[PromptSet]) -> None:
    from sam2.build_sam import build_sam2_video_predictor_hf

    predictor = build_sam2_video_predictor_hf(args.model, device=args.device).eval()
    frames = sorted(path for path in args.frames_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not frames:
        raise FileNotFoundError(f"No image frames found in {args.frames_dir}")
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
            mask_path = args.output / f"frame_{frame_idx:06d}_obj{obj_id}.png"
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
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    selected_frames = frames[:limit_frames] if limit_frames is not None else frames
    for index, frame in enumerate(selected_frames):
        staged = output / f"{index:05d}.jpg"
        try:
            os.symlink(frame.resolve(), staged)
        except OSError:
            shutil.copyfile(frame, staged)
    return output


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_2d = np.squeeze(mask)
    image = mask_2d.astype(np.uint8) * 255 if mask_2d.dtype == np.bool_ else (mask_2d > 0).astype(np.uint8) * 255
    Image.fromarray(image).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM2 Torch reference on an image or image sequence.")
    parser.add_argument("--model", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--multimask-output", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if (args.image is None) == (args.frames_dir is None):
        raise ValueError("Specify exactly one of --image or --frames-dir")
    return args


if __name__ == "__main__":
    with torch.inference_mode():
        main()
