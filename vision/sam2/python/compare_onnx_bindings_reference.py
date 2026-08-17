# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frames_dir = args.output / "frames"
    extract_frames(args.video, frames_dir, args.frames)
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No frames extracted from {args.video}")

    if args.ep_cache is None:
        args.ep_cache = args.output / "trt_rtx_cache"
        shutil.rmtree(args.ep_cache, ignore_errors=True)
    if args.ep_context_dir is None:
        args.ep_context_dir = args.output / "ep_context"
        shutil.rmtree(args.ep_context_dir, ignore_errors=True)

    prompt = prompt_from_args(args, frames[0])
    image_candidates = None
    video_candidates = None
    if args.mode in {"image", "both"}:
        image_candidates = [run_candidate_frame(args, frame, prompt, index) for index, frame in enumerate(frames)]
    if args.mode in {"video", "both"}:
        video_candidates = run_candidate_video(args, frames, prompt)

    sections = {}
    if args.mode in {"image", "both"}:
        assert image_candidates is not None
        records = [compare_reference_frame(args, candidate, prompt) for candidate in image_candidates]
        sections["image"] = summarize_records(records)
    if args.mode in {"video", "both"}:
        assert video_candidates is not None
        records = compare_reference_video(args, frames, video_candidates, prompt)
        sections["video"] = summarize_records(records)

    summary = {
        "video": str(args.video),
        "onnx_dir": str(args.onnx_dir),
        "model": args.model,
        "mode": args.mode,
        "sections": sections,
    }
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, section in sections.items():
        print(f"{name}: frames={section['frames']} mean_iou={section['mean_iou']} min_iou={section['min_iou']}")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ious = np.asarray([record["iou"] for record in records], dtype=np.float64)
    return {
        "frames": len(records),
        "mean_iou": float(ious.mean()) if len(ious) else None,
        "min_iou": float(ious.min()) if len(ious) else None,
        "records": records,
    }


def run_candidate_frame(args: argparse.Namespace, frame: Path, prompt: dict[str, Any], index: int) -> dict[str, Any]:
    prompt_path = args.output / "candidate_image_prompt.json"
    write_cli_prompt(prompt, args.onnx_dir / "metadata.json", prompt_path)
    output_dir = args.output / "candidate_image_cli" / f"frame_{index:06d}"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(args.din_sam2),
        "--model-dir",
        str(args.onnx_dir),
        "--ep-cache",
        str(args.ep_cache),
        "--ep-context-dir",
        str(args.ep_context_dir),
        "--provider",
        args.provider,
        "--dump-masks",
        str(frame),
        str(prompt_path),
        str(output_dir),
    ]
    subprocess.run(command, check=True, env=cli_env(args))

    cli_mask = output_dir / "frame_000000_fast.png"
    if not cli_mask.exists():
        raise RuntimeError(f"SAM2 CLI did not write expected mask: {cli_mask}")
    candidate_path = args.output / "candidate" / f"frame_{index:06d}.png"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cli_mask, candidate_path)
    return {"frame": frame, "candidate_mask": candidate_path}


def compare_reference_frame(
    args: argparse.Namespace, candidate: dict[str, Any], prompt: dict[str, Any]
) -> dict[str, Any]:
    frame = candidate["frame"]
    index = int(Path(candidate["candidate_mask"]).stem.split("_")[-1])
    image = np.array(Image.open(frame).convert("RGB"), copy=True)
    reference_mask = run_reference_image(args.model, args.device, image, prompt)
    candidate_mask = np.asarray(Image.open(candidate["candidate_mask"]).convert("RGB"))[:, :, 0] > 0

    reference_path = args.output / "reference" / f"frame_{index:06d}.png"
    save_binary_mask(reference_mask, reference_path)

    iou = mask_iou(reference_mask, candidate_mask)
    return {
        "frame": str(frame),
        "reference_mask": str(reference_path),
        "candidate_mask": str(candidate["candidate_mask"]),
        "iou": iou,
    }


def run_candidate_video(args: argparse.Namespace, frames: list[Path], prompt: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_path = args.output / "candidate_video_prompt.json"
    write_cli_prompt(prompt, args.onnx_dir / "metadata.json", prompt_path)
    output_dir = args.output / "candidate_video"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(args.din_sam2),
        "--model-dir",
        str(args.onnx_dir),
        "--ep-cache",
        str(args.ep_cache),
        "--ep-context-dir",
        str(args.ep_context_dir),
        "--provider",
        args.provider,
        "--propagate",
        "--dump-masks",
        "--max-frames",
        str(len(frames)),
        str(args.output / "frames"),
        str(prompt_path),
        str(output_dir),
    ]
    subprocess.run(command, check=True, env=cli_env(args))

    candidates = []
    for index, frame in enumerate(frames):
        path = output_dir / f"frame_{index:06d}_fast.png"
        if path.exists():
            candidates.append({"frame": frame, "candidate_mask": path, "frame_idx": index})
    return candidates


def compare_reference_video(
    args: argparse.Namespace,
    frames: list[Path],
    candidates: list[dict[str, Any]],
    prompt: dict[str, Any],
) -> list[dict[str, Any]]:
    from sam2.build_sam import build_sam2_video_predictor_hf

    predictor = build_sam2_video_predictor_hf(args.model, device=args.device).eval()
    staged_frames = stage_frames_for_sam2(frames, args.output / "reference_video_frames")
    inference_state = predictor.init_state(video_path=str(staged_frames))
    predictor.add_new_points_or_box(
        inference_state,
        frame_idx=0,
        obj_id=1,
        points=prompt["points"] if len(prompt["points"]) else None,
        labels=prompt["labels"] if len(prompt["labels"]) else None,
        box=prompt["box"],
        normalize_coords=True,
    )

    reference_masks = {}
    for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
        inference_state,
        max_frame_num_to_track=len(frames),
    ):
        masks = video_res_masks.detach().cpu().numpy()
        if 1 in obj_ids:
            obj_index = list(obj_ids).index(1)
            reference_masks[int(frame_idx)] = np.squeeze(masks[obj_index]) > 0

    records = []
    for candidate in candidates:
        frame_idx = int(candidate["frame_idx"])
        if frame_idx not in reference_masks:
            continue
        reference_path = args.output / "reference_video" / f"frame_{frame_idx:06d}.png"
        save_binary_mask(reference_masks[frame_idx], reference_path)
        candidate_mask = np.asarray(Image.open(candidate["candidate_mask"]).convert("RGB"))[:, :, 0] > 0
        records.append(
            {
                "frame": str(candidate["frame"]),
                "reference_mask": str(reference_path),
                "candidate_mask": str(candidate["candidate_mask"]),
                "iou": mask_iou(reference_masks[frame_idx], candidate_mask),
            }
        )
    return records


def run_reference_image(model: str, device: str, image: np.ndarray, prompt: dict[str, Any]) -> np.ndarray:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(model, device=device)
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=prompt["points"] if len(prompt["points"]) else None,
        point_labels=prompt["labels"] if len(prompt["labels"]) else None,
        box=prompt["box"],
        multimask_output=False,
        return_logits=True,
        normalize_coords=True,
    )
    best = int(np.argmax(scores))
    return np.squeeze(masks[best]) > 0


def write_cli_prompt(prompt: dict[str, Any], metadata_path: Path, output: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    max_points = int(metadata["max_points"])
    points = np.asarray(prompt["points"], dtype=np.float32).reshape(-1, 2).tolist()
    labels = np.asarray(prompt["labels"], dtype=np.int32).reshape(-1).tolist()
    if prompt["box"] is not None:
        raise ValueError("video CLI comparison currently expects --point, not --box")
    while len(labels) < max_points:
        points.append([0.0, 0.0])
        labels.append(-1)
    output.write_text(
        json.dumps({"points": points[:max_points], "labels": labels[:max_points]}, indent=2), encoding="utf-8"
    )


def stage_frames_for_sam2(frames: list[Path], output: Path) -> Path:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        staged = output / f"{index:05d}.jpg"
        try:
            os.symlink(frame.resolve(), staged)
        except OSError:
            shutil.copyfile(frame, staged)
    return output


def cli_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    lib_dirs = [str(args.build_dir)]
    ep_dir = args.build_dir.parent.parent / "onnxruntime_trt_rtx_ep" / args.build_dir.name
    if ep_dir.exists():
        lib_dirs.append(str(ep_dir))
    existing = env.get("LD_LIBRARY_PATH")
    if existing:
        lib_dirs.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs)
    return env


def extract_frames(video: Path, output: Path, count: int) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required to extract frames from the sample .mov: pip install av") from error

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index >= count:
                break
            image = frame.to_image().convert("RGB")
            image.save(output / f"frame_{index:06d}.png")


def prompt_from_args(args: argparse.Namespace, frame: Path) -> dict[str, Any]:
    image = Image.open(frame)
    width, height = image.size
    if args.box is not None:
        box = np.asarray(args.box, dtype=np.float32)
        points = np.empty((0, 2), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int32)
    else:
        point = args.point if args.point is not None else [width * 0.5, height * 0.5]
        box = None
        points = np.asarray([point], dtype=np.float32)
        labels = np.asarray([1], dtype=np.int32)
    return {"points": points, "labels": labels, "box": box}


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.squeeze(mask) > 0).astype(np.uint8) * 255).save(path)


def mask_iou(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.squeeze(reference) > 0
    candidate = np.squeeze(candidate) > 0
    if reference.shape != candidate.shape:
        raise ValueError(f"Mask shape mismatch: {reference.shape} vs {candidate.shape}")
    intersection = np.logical_and(reference, candidate).sum()
    union = np.logical_or(reference, candidate).sum()
    return 1.0 if union == 0 else float(intersection / union)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare upstream SAM2 masks with DIN SAM2 ONNX paths.")
    parser.add_argument("--mode", choices=("image", "video", "both"), default="both")
    parser.add_argument("--video", type=Path, default=Path("assets/jhh-sample.mov"))
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("out/sam2_cli_compare"))
    parser.add_argument("--din-sam2", type=Path, default=Path("out/build/linux-x64/bin/Debug/din_sam2"))
    parser.add_argument("--build-dir", type=Path, default=Path("out/build/linux-x64/bin/Debug"))
    parser.add_argument("--model", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--provider", default="trt-rtx")
    parser.add_argument("--ep-cache", type=Path)
    parser.add_argument("--ep-context-dir", type=Path)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--point", type=float, nargs=2)
    parser.add_argument("--box", type=float, nargs=4)
    args = parser.parse_args()
    if args.point is not None and args.box is not None:
        raise ValueError("Use either --point or --box, not both")
    return args


if __name__ == "__main__":
    main()
