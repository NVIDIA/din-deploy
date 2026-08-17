# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    args = parse_args()
    frames_dir = args.output / "frames"
    prompt_dir = args.output / "prompts"
    image_ref_dir = args.output / "reference" / "image"
    video_ref_dir = args.output / "reference" / "video"
    for path in (frames_dir, prompt_dir, image_ref_dir, video_ref_dir):
        path.mkdir(parents=True, exist_ok=True)

    frames = extract_frames(args.video, frames_dir, args.frames)
    if args.onnx_dir is not None:
        copy_onnx_dir(args.onnx_dir, args.output / "onnx")

    width, height = Image.open(frames[0]).size
    point = args.point or [width * 0.5, height * 0.5]
    max_points = load_max_points(args.onnx_dir) if args.onnx_dir is not None else args.max_points
    write_prompt(prompt_dir / "center_point.json", point, max_points)

    reference_image = run_reference_image(args.model, args.device, frames[0], point)
    save_binary_mask(reference_image, image_ref_dir / "frame_000000_fast.png")

    reference_video = run_reference_video(args.model, args.device, frames, point)
    for frame_idx, mask in sorted(reference_video.items()):
        save_binary_mask(mask, video_ref_dir / f"frame_{frame_idx:06d}_fast.png")


def extract_frames(video: Path, output: Path, count: int) -> list[Path]:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required: pip install av") from error

    for old_frame in output.glob("frame_*.png"):
        old_frame.unlink()

    frames = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index >= count:
                break
            path = output / f"frame_{index:06d}.png"
            frame.to_image().convert("RGB").save(path)
            frames.append(path)
    if not frames:
        raise RuntimeError(f"No frames extracted from {video}")
    return frames


def copy_onnx_dir(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def load_max_points(onnx_dir: Path | None) -> int:
    if onnx_dir is None:
        raise RuntimeError("--onnx-dir is required to load max_points")
    metadata = json.loads((onnx_dir / "metadata.json").read_text(encoding="utf-8"))
    return int(metadata["max_points"])


def write_prompt(path: Path, point: list[float], max_points: int) -> None:
    points = [[float(point[0]), float(point[1])]]
    labels = [1]
    while len(labels) < max_points:
        points.append([0.0, 0.0])
        labels.append(-1)
    path.write_text(
        json.dumps({"points": points[:max_points], "labels": labels[:max_points]}, indent=2),
        encoding="utf-8",
    )


def run_reference_image(model: str, device: str, frame: Path, point: list[float]) -> np.ndarray:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image = np.asarray(Image.open(frame).convert("RGB"))
    predictor = SAM2ImagePredictor.from_pretrained(model, device=device)
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray([point], dtype=np.float32),
        point_labels=np.asarray([1], dtype=np.int32),
        multimask_output=False,
        return_logits=True,
        normalize_coords=True,
    )
    return np.squeeze(masks[int(np.argmax(scores))]) > 0


def run_reference_video(model: str, device: str, frames: list[Path], point: list[float]) -> dict[int, np.ndarray]:
    from sam2.build_sam import build_sam2_video_predictor_hf

    predictor = build_sam2_video_predictor_hf(model, device=device).eval()
    staged_frames = stage_frames_for_sam2(frames, frames[0].parent.parent / "reference_video_frames")
    inference_state = predictor.init_state(video_path=str(staged_frames))
    predictor.add_new_points_or_box(
        inference_state,
        frame_idx=0,
        obj_id=1,
        points=np.asarray([point], dtype=np.float32),
        labels=np.asarray([1], dtype=np.int32),
        normalize_coords=True,
    )

    masks_by_frame = {}
    for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
        inference_state,
        max_frame_num_to_track=len(frames),
    ):
        if 1 not in obj_ids:
            continue
        obj_index = list(obj_ids).index(1)
        masks = video_res_masks.detach().cpu().numpy()
        masks_by_frame[int(frame_idx)] = np.squeeze(masks[obj_index]) > 0
    return masks_by_frame


def stage_frames_for_sam2(frames: list[Path], output: Path) -> Path:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.open(frame).convert("RGB").save(output / f"{index:05d}.jpg", quality=95)
    return output


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((np.squeeze(mask) > 0).astype(np.uint8) * 255).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SAM2 C++ integration test artifacts.")
    parser.add_argument("--video", type=Path, default=Path("assets/jhh-sample.mov"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--point", type=float, nargs=2)
    parser.add_argument("--onnx-dir", type=Path)
    parser.add_argument("--max-points", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
