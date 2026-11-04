# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a new fixed-attention bundle without changing the deployed baseline."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from static_attention import cpu_mask_compatibility, specialize_fixtures, specialize_memory_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("sam2", "sam3"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--memory-overlay", type=Path, help="Corrected FP32 SAM3 memory export to overlay on the FP32 reference bundle")
    parser.add_argument("--cpu-compatible", action="store_true", help="Chunk CPU attention queries for ORT broadcast-mask compatibility")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or (output.exists() and any(output.iterdir())):
        parser.error("Output must be a new or empty directory distinct from source")
    metadata = json.loads((source / "metadata.json").read_text())
    overlay = None
    if args.memory_overlay:
        if args.family != "sam3":
            parser.error("Memory overlays are only supported for SAM3 reference bundles")
        overlay = json.loads((args.memory_overlay / "metadata.json").read_text())
        for key in ("memory_boundary_version", "memory_boundaries", "memory_boundary_validation"):
            if key in overlay:
                metadata[key] = overlay[key]
    if args.family == "sam3" and metadata.get("memory_boundary_version") != 2:
        parser.error("SAM3 requires corrected memory boundary v2")
    for name in metadata["graphs"]:
        directory = args.memory_overlay if overlay and name in ("seed_memory", "tracking_memory", "memory_encoder") else source
        if not (directory / (name + ".onnx")).is_file():
            parser.error(f"Source bundle is incomplete: missing {name}.onnx")
    output.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and (path.name.endswith((".onnx", ".onnx.data", ".json", ".npz"))):
            shutil.copy2(path, output / path.name)
        elif path.is_dir() and path.name.endswith(".fixtures"):
            shutil.copytree(path, output / path.name)
    if overlay:
        for name in ("seed_memory", "tracking_memory", "memory_encoder"):
            for path in args.memory_overlay.glob(name + ".*"):
                if path.is_file():
                    shutil.copy2(path, output / path.name)
                elif path.is_dir() and path.name.endswith(".fixtures"):
                    shutil.copytree(path, output / path.name, dirs_exist_ok=True)
            metadata["graphs"][name] = overlay["graphs"][name]
        metadata["memory_overlay_source"] = str(args.memory_overlay.resolve())
    contract = specialize_memory_attention(output / "memory_attention.onnx", args.family)
    specialize_fixtures(output / "memory_attention.onnx", args.family)
    if args.cpu_compatible:
        cpu_mask_compatibility(output / "memory_attention.onnx", args.family)
        metadata["cpu_attention_query_chunk"] = 128
    metadata["attention_contract"] = contract
    metadata["static_state_source"] = str(source)
    metadata["static_attention_validation"] = "pending"
    if "precision_validation" in metadata:
        metadata["source_precision_validation"] = metadata["precision_validation"]
    metadata["precision_validation"] = "Source precision policy preserved; fixed-attention validation pending; full native PyTorch parity is not established"
    if args.family == "sam3":
        import onnx
        graph = onnx.load(output / "memory_attention.onnx", load_external_data=False).graph
        def describe(value):
            return {"name": value.name, "shape": [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim],
                    "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type).lower()}
        metadata["graphs"]["memory_attention"]["inputs"] = [describe(v) for v in graph.input]
        fixtures = output / "memory_attention.inputs.npz"
        if fixtures.exists():
            import numpy as np
            with np.load(fixtures) as saved:
                metadata["graphs"]["memory_attention"]["fixture_inputs"] = {n: list(v.shape) for n, v in saved.items()}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    manifest = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                manifest.append({"file": path.name, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()})
    (output / "static_state_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(output), "contract": contract}, indent=2))


if __name__ == "__main__":
    main()
