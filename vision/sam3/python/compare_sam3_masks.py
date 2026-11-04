# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare instance masks with one-to-one IoU matching and check output consistency."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


def load_frame(root, path):
    records = json.loads(path.read_text(encoding="utf-8"))
    combined = np.asarray(Image.open(root / f"{path.stem}_combined.png")) > 0
    labels = np.load(root / f"{path.stem}_labels.npy")
    masks = [np.asarray(Image.open(root / item["mask"])) > 0 for item in records]
    expected = np.logical_or.reduce(masks) if masks else np.zeros_like(combined)
    if not np.array_equal(expected, combined) or not np.array_equal(labels > 0, combined):
        raise ValueError(f"Inconsistent combined mask/labels in {path}")
    known_ids = {int(item["id"]) for item in records}
    if len(known_ids) != len(records) or 0 in known_ids or not set(np.unique(labels)).issubset(known_ids | {0}):
        raise ValueError(f"Invalid instance labels in {path}")
    for record, mask in zip(records, masks, strict=True):
        if np.any((labels == record["id"]) & ~mask):
            raise ValueError(f"Label assigned outside its instance in {path}")
    return records, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--min-iou", type=float, default=0.99)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    reference_files = sorted(args.reference.glob("[0-9]*.json"))
    actual_files = sorted(args.actual.glob("[0-9]*.json"))
    if not reference_files or [p.name for p in reference_files] != [p.name for p in actual_files]:
        raise ValueError("Frame inventories differ or are empty")
    report = {"passed": True, "frames": []}
    for reference, actual in zip(reference_files, actual_files, strict=True):
        ref_records, ref_masks = load_frame(args.reference, reference)
        actual_records, actual_masks = load_frame(args.actual, actual)
        if len(ref_masks) != len(actual_masks):
            raise ValueError(f"Instance count mismatch at {reference.stem}: {len(ref_masks)} vs {len(actual_masks)}")
        iou = np.zeros((len(ref_masks), len(actual_masks)))
        for i, ref in enumerate(ref_masks):
            for j, candidate in enumerate(actual_masks):
                if ref.shape != candidate.shape:
                    raise ValueError("Mask dimensions differ")
                union = np.count_nonzero(ref | candidate)
                iou[i, j] = np.count_nonzero(ref & candidate) / union if union else 1
        rows, cols = linear_sum_assignment(-iou)
        pairs = [
            {"reference_id": ref_records[i]["id"], "actual_id": actual_records[j]["id"], "iou": float(iou[i, j])}
            for i, j in zip(rows, cols, strict=True)
        ]
        report["frames"].append({"frame": reference.stem, "matches": pairs})
        report["passed"] &= all(pair["iou"] >= args.min_iou for pair in pairs)
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
