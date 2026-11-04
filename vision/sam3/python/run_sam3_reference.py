# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run pinned SAM3.1 image or causal video inference and save instance outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sam3_common import LOCK, build_predictor


def save_outputs(output: Path, frame: int, ids, probabilities, scores, boxes=None):
    import numpy as np
    from PIL import Image

    output.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(ids, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim == 4:
        probabilities = probabilities[:, 0]
    if len(ids) != len(probabilities) or len(ids) != len(scores):
        raise ValueError("Inconsistent instance outputs")
    order = np.argsort(ids, kind="stable")
    ids, probabilities = ids[order], probabilities[order]
    scores = np.asarray(scores)[order]
    height, width = probabilities.shape[-2:]
    labels = np.zeros((height, width), dtype=np.uint32)
    union = np.zeros((height, width), dtype=bool)
    if len(ids):
        # np.argmax chooses the first (lowest ID) on exact ties.
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
        record = {"id": int(obj_id), "score": float(scores[index]), "mask": filename}
        if boxes is not None:
            record["box"] = np.asarray(boxes)[order[index]].tolist()
        records.append(record)
    (output / f"{prefix}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text must not be empty")
    import torch
    from PIL import Image

    start = time.perf_counter()
    predictor = build_predictor()
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    # The upstream video orchestrator stores BF16 memories and has nested BF16
    # autocast regions. Keep the official video precision rather than label that
    # execution FP32. Image/export correctness is established independently in FP32.
    with torch.inference_mode(), torch.autocast("cuda", enabled=args.video, dtype=torch.bfloat16):
        if not args.video:
            from sam3.model.sam3_image_processor import Sam3Processor

            processor = Sam3Processor(predictor.model.detector)
            state = processor.set_image(Image.open(args.input).convert("RGB"))
            result = processor.set_text_prompt(args.text, state)
            count = len(result["scores"])
            if count > 16:
                raise RuntimeError(f"Object capacity exceeded: {count} > 16")
            # Despite its name, masks_logits contains sigmoid probabilities in this revision.
            save_outputs(
                args.output,
                0,
                list(range(1, count + 1)),
                result["masks_logits"].cpu().numpy(),
                result["scores"].cpu().numpy(),
                result["boxes"].cpu().numpy(),
            )
        else:
            session = predictor.handle_request({"type": "start_session", "resource_path": str(args.input.resolve())})
            session_id = session["session_id"]
            try:
                predictor.handle_request(
                    {"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": args.text}
                )
                for result in predictor.handle_stream_request(
                    {"type": "propagate_in_video", "session_id": session_id, "propagation_direction": "forward"}
                ):
                    out = result["outputs"]
                    # Upstream video API returns binary masks; retain that fact in metadata.
                    save_outputs(
                        args.output,
                        result["frame_index"],
                        out["out_obj_ids"] + 1,
                        out["out_binary_masks"],
                        out["out_probs"],
                    )
            finally:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
    torch.cuda.synchronize()
    metadata = {
        **LOCK,
        "precision": "bfloat16" if args.video else "float32",
        "hotstart_delay": 0,
        "causal": True,
        "text": args.text,
        "video_masks_are_binary": args.video,
        "initialization_seconds": init_seconds,
        "inference_seconds": time.perf_counter() - start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reference.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
