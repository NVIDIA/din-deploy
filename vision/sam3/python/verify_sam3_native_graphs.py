# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive the native graph parity harness and preserve each graph's diagnostics."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--provider", choices=["cpu", "trt-rtx"], default="trt-rtx")
    parser.add_argument("--graphs", nargs="+")
    args = parser.parse_args()
    metadata = json.loads((args.model_dir / "metadata.json").read_text())
    names = args.graphs or list(metadata["graphs"])
    report = {}
    for name in names:
        if name == "image_preprocess" and args.provider == "trt-rtx":
            report[name] = {"skipped": "Native GPU path uses the tested CUDA preprocessing kernel"}
            continue
        print(f"Checking native {args.provider}: {name}", flush=True)
        with (args.model_dir / f"{name}.{args.provider}.log").open("w") as log:
            result = subprocess.run(
                [str(args.executable.resolve()), str(args.model_dir.resolve()), name, args.provider],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        report[name] = {"passed": result.returncode == 0, "exit_code": result.returncode}
        print(json.dumps(report[name]), flush=True)
        (args.model_dir / f"native_{args.provider}_parity.json").write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if all(item.get("passed", True) for item in report.values()) else 1)


if __name__ == "__main__":
    main()
