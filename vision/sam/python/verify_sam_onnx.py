# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check SAM2/SAM3 ONNX bundles; optionally compare saved PyTorch fixtures on CPU."""

import argparse
import json
from pathlib import Path


def verify_graph(path, fixtures, atol, rtol):
    import onnx

    # Path-based checking resolves external weight sidecars without loading all
    # model weights into Python or rewriting the bundle.
    onnx.checker.check_model(str(path), full_check=True)
    result = {"passed": True, "onnx_checked": True}
    if not fixtures:
        return result

    inputs_path = path.with_suffix(".inputs.npz")
    outputs_path = path.with_suffix(".outputs.npz")
    if not inputs_path.is_file() or not outputs_path.is_file():
        raise FileNotFoundError(f"Saved input/output fixtures are required for {path.stem}")

    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    with np.load(inputs_path) as saved:
        inputs = {item.name: saved[item.name] for item in session.get_inputs()}
    actual = session.run(None, inputs)
    comparisons = {}
    with np.load(outputs_path) as expected:
        for info, value in zip(session.get_outputs(), actual, strict=True):
            reference = expected[info.name]
            same_shape = value.shape == reference.shape
            finite = bool(np.isfinite(value).all())
            good = same_shape and finite and bool(np.allclose(value, reference, atol=atol, rtol=rtol))
            error = np.abs(value.astype(np.float64) - reference.astype(np.float64)) if same_shape else None
            comparisons[info.name] = {
                "passed": good,
                "same_shape": same_shape,
                "finite": finite,
                "max_absolute_error": float(error.max()) if error is not None and error.size else 0.0,
            }
    result.update(passed=all(item["passed"] for item in comparisons.values()), outputs=comparisons)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--graphs", nargs="+", help="Graph names; defaults to every graph in metadata.json")
    parser.add_argument("--fixtures", action="store_true", help="Require saved input/output fixtures and compare CPU outputs")
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--rtol", type=float, default=0.002)
    parser.add_argument("--report", type=Path, help="Optional JSON output; the model bundle is never modified")
    args = parser.parse_args()
    directory = args.model_dir.resolve()
    if args.report and args.report.resolve().is_relative_to(directory):
        parser.error("Place verification reports outside the model bundle")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8-sig"))
    contracts = metadata["graphs"]
    names = args.graphs or list(contracts)
    if not names or set(names) - set(contracts):
        parser.error("Select at least one graph listed in metadata.json")
    report = {}
    for name in names:
        filename = contracts[name].get("file", name + ".onnx") if isinstance(contracts, dict) else name + ".onnx"
        path = (directory / filename).resolve()
        try:
            if not path.is_relative_to(directory):
                raise ValueError("Graph path escapes the model bundle")
            report[name] = verify_graph(path, args.fixtures, args.atol, args.rtol)
        except Exception as error:
            report[name] = {"passed": False, "error": str(error)}
        print(json.dumps({"graph": name, **report[name]}), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(int(not all(item["passed"] for item in report.values())))


if __name__ == "__main__":
    main()
