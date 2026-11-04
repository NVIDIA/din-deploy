# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare saved PyTorch graph fixtures against ONNX Runtime CPU outputs."""

import argparse
import json
import time
from pathlib import Path


def main():
    import numpy as np
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--graphs", nargs="+")
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--rtol", type=float, default=0.002)
    parser.add_argument("--disable-optimization", action="store_true")
    args = parser.parse_args()
    metadata = json.loads((args.model_dir / "metadata.json").read_text())
    report = {}
    for name in args.graphs or metadata["graphs"]:
        print(f"Checking {name}", flush=True)
        started = time.perf_counter()
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 8
            if args.disable_optimization:
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(
                str(args.model_dir / f"{name}.onnx"), options, providers=["CPUExecutionProvider"]
            )
            with np.load(args.model_dir / f"{name}.inputs.npz") as inputs:
                result = session.run(None, {item.name: inputs[item.name] for item in session.get_inputs()})
            comparisons = {}
            with np.load(args.model_dir / f"{name}.outputs.npz") as reference:
                for item, value in zip(session.get_outputs(), result, strict=True):
                    expected = reference[item.name]
                    finite = bool(np.isfinite(value).all())
                    same_shape = value.shape == expected.shape
                    good = same_shape and finite and bool(np.allclose(value, expected, atol=args.atol, rtol=args.rtol))
                    error = np.abs(value.astype(np.float64) - expected.astype(np.float64)) if same_shape else None
                    comparisons[item.name] = {
                        "passed": good,
                        "finite": finite,
                        "same_shape": same_shape,
                        "max_absolute_error": float(error.max()) if error is not None and error.size else 0,
                        "mean_absolute_error": float(error.mean()) if error is not None and error.size else 0,
                    }
            report[name] = {"passed": all(item["passed"] for item in comparisons.values()), "outputs": comparisons}
            del session, result
        except Exception as error:
            report[name] = {"passed": False, "error": str(error)}
        report[name]["seconds"] = time.perf_counter() - started
        print(json.dumps(report[name]), flush=True)
        (args.model_dir / "graph_parity.json").write_text(json.dumps(report, indent=2))
    if not all(item["passed"] for item in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
