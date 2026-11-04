# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a separate mixed-FP16/BF16 artifact set, retaining FP32 graph interfaces.

ModelOpt uses saved reference inputs to retain sensitive operations in FP32.
Conversion is not a parity certificate: run graph and end-to-end mask validation.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def validate_converted_graph(path):
    import onnx

    # ModelOpt may leave stale intermediate type annotations after inserting
    # casts. Discard only those annotations, then infer/check actual node types.
    # Graph inputs, outputs, operators and weights remain unchanged.
    model = onnx.load(path)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    # The installed ModelOpt version emits invalid mixed types for the detector's
    # ConstantOfShape branch. Keep detection and tracking in FP32 by default.
    parser.add_argument("--graphs", nargs="+", default=["text_encoder", "vision_encoder"])
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Use a separate output directory to preserve the FP32 baseline")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.source / "metadata.json").read_text())
    unknown = set(args.graphs) - set(metadata["graphs"])
    if unknown:
        parser.error(f"Unknown graphs: {unknown}")
    precision = "mixed_bfloat16" if args.precision == "bf16" else "mixed_float16"
    metadata["precision"] = precision
    metadata["precision_validation"] = "pending"
    metadata["graph_precision"] = {}
    metadata["fp32_operator_exceptions"] = ["ConvTranspose"] if args.precision == "bf16" else []
    for name, contract in metadata["graphs"].items():
        if name in args.graphs:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modelopt.onnx.autocast",
                    "--onnx_path",
                    str(args.source / contract["file"]),
                    "--output_path",
                    str(args.output / contract["file"]),
                    "--low_precision_type",
                    args.precision,
                    "--keep_io_types",
                    "--calibration_data",
                    str(args.source / f"{name}.inputs.npz"),
                    "--providers",
                    "cpu",
                    *(["--op_types_to_exclude", "ConvTranspose"] if args.precision == "bf16" else []),
                ],
                check=True,
            )
            validate_converted_graph(args.output / contract["file"])
            metadata["graph_precision"][name] = precision
        else:
            shutil.copy2(args.source / contract["file"], args.output / contract["file"])
            sidecar = args.source / (contract["file"] + ".data")
            if sidecar.exists():
                shutil.copy2(sidecar, args.output / sidecar.name)
            metadata["graph_precision"][name] = "float32"
        for suffix in ("inputs.npz", "outputs.npz"):
            shutil.copy2(args.source / f"{name}.{suffix}", args.output / f"{name}.{suffix}")
        fixtures = args.source / f"{name}.fixtures"
        if fixtures.exists():
            shutil.copytree(fixtures, args.output / fixtures.name)
    for filename in ("tokenizer.json", "tokenizer_tests.json"):
        shutil.copy2(args.source / filename, args.output / filename)
    if "temporal_constants" in metadata:
        shutil.copy2(args.source / metadata["temporal_constants"], args.output / metadata["temporal_constants"])
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
