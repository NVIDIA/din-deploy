# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def main() -> None:
    args = parse_args()
    if args.reference_mask is not None:
        if args.candidate_mask is None:
            raise ValueError("--candidate-mask is required with --reference-mask")
        pairs = [(args.reference_mask, args.candidate_mask)]
    else:
        if args.reference_json is None or args.candidate_json is None:
            raise ValueError("Use either --reference-mask/--candidate-mask or --reference-json/--candidate-json")
        reference = json.loads(args.reference_json.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate_json.read_text(encoding="utf-8"))
        reference_records = records_by_frame(reference)
        candidate_records = records_by_frame(candidate)
        common_frames = sorted(set(reference_records) & set(candidate_records))
        if not common_frames and "mask_path" in candidate and reference.get("records"):
            pairs = [(Path(reference["records"][0]["mask_path"]), Path(candidate["mask_path"]))]
        else:
            pairs = [
                (Path(reference_records[frame]["mask_path"]), Path(candidate_records[frame]["mask_path"]))
                for frame in common_frames
            ]

    records = []
    for reference, candidate in pairs:
        reference_mask = np.asarray(Image.open(reference).convert("RGB"))[:, :, 0] > 0
        candidate_mask = np.asarray(Image.open(candidate).convert("RGB"))[:, :, 0] > 0
        if reference_mask.shape != candidate_mask.shape:
            raise ValueError(
                f"Mask shape mismatch: {reference} {reference_mask.shape} vs {candidate} {candidate_mask.shape}"
            )
        intersection = np.logical_and(reference_mask, candidate_mask).sum()
        union = np.logical_or(reference_mask, candidate_mask).sum()
        iou = 1.0 if union == 0 else float(intersection / union)
        records.append({"reference": str(reference), "candidate": str(candidate), "iou": iou, "error": 1.0 - iou})

    errors = np.asarray([record["error"] for record in records], dtype=np.float64)
    ious = np.asarray([record["iou"] for record in records], dtype=np.float64)
    if errors.size == 0:
        summary = {"count": 0, "mean_error": None, "min_error": None, "max_error": None, "mean_iou": None}
    else:
        summary = {
            "count": int(errors.size),
            "mean_error": float(errors.mean()),
            "min_error": float(errors.min()),
            "max_error": float(errors.max()),
            "mean_iou": float(ious.mean()),
            "min_iou": float(ious.min()),
            "max_iou": float(ious.max()),
        }
    result = {"summary": summary, "records": records}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "count={count} mean_error={mean_error} min_error={min_error} max_error={max_error} "
        "mean_iou={mean_iou} min_iou={min_iou} max_iou={max_iou}".format(**summary)
    )


def records_by_frame(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if "records" in payload:
        return {int(record.get("frame_idx", index)): record for index, record in enumerate(payload["records"])}
    if "mask_path" in payload:
        return {int(payload.get("frame_idx", 0)): payload}
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SAM2 reference and candidate binary masks.")
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--candidate-json", type=Path)
    parser.add_argument("--reference-mask", type=Path)
    parser.add_argument("--candidate-mask", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
