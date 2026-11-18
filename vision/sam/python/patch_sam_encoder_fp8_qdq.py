# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy a SAM bundle and insert calibrated encoder FP8 E4M3 Q/DQ."""

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path


def encoder_modules(model, family):
    """Map exported operators through weight/bias names, independent of node IDs."""
    producers = {out: node for node in model.graph.node for out in node.output}
    initializers = {value.name for value in model.graph.initializer}
    modules = {}
    for index, node in enumerate(model.graph.node):
        if node.op_type not in ("MatMul", "Conv") or len(node.input) < 2:
            continue
        weight = node.input[1]
        producer = producers.get(weight)
        if producer is not None and producer.op_type == "Transpose":
            weight = producer.input[0]
        if family == "sam3" and weight in initializers:
            # Folding transposes replaces weight names; export scopes retain the
            # original module identity without relying on generated node IDs.
            scopes = next((p.value for p in node.metadata_props if p.key == "pkg.torch.onnx.name_scopes"), "")
            matches = re.findall(r"visual\.trunk\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.fc[12])", scopes)
            if matches:
                modules[index] = matches[-1]
                continue
        if weight not in initializers or not weight.endswith(".weight"):
            continue
        module = weight.removesuffix(".weight")
        if family == "sam2":
            module = module.removeprefix("model.")
            if module.startswith("image_encoder."):
                modules[index] = "model." + module
        elif re.fullmatch(r"visual\.trunk\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.fc[12])", module):
            modules[index] = module
    # Some SAM2 exports fold weight transposes into unnamed initializers.
    if family == "sam2":
        by_output = {node.output[0]: index for index, node in enumerate(model.graph.node) if node.output}
        for node in model.graph.node:
            if node.op_type != "Add":
                continue
            bias = next((name for name in node.input if name in initializers and name.endswith(".bias")), None)
            if bias is None:
                continue
            module = bias.removesuffix(".bias").removeprefix("model.")
            if not module.startswith("image_encoder."):
                continue
            for name in node.input:
                index = by_output.get(name)
                if index is not None and model.graph.node[index].op_type == "MatMul":
                    modules[index] = "model." + module
    return modules


def validate_calibration(metadata, calibration, family):
    if calibration.get("schema_version") != 1 or calibration.get("model_family") != family:
        raise ValueError("Calibration schema/model family mismatch")
    keys = (
        ("dtype", "image_size", "model")
        if family == "sam2"
        else (
            "dtype",
            "image_size",
            "source_repo",
            "source_revision",
            "checkpoint_repo",
            "checkpoint_file",
            "checkpoint_revision",
        )
    )
    for key in keys:
        if key not in metadata or metadata[key] != calibration.get(key):
            raise ValueError(f"Calibration/export mismatch: {key}")
    if metadata["dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported bundle precision")


def insert_qdq(model, calibration, family, *, weight_only=False):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnx.reference.ops import load_op

    if any(node.op_type in ("QuantizeLinear", "DequantizeLinear") for node in model.graph.node):
        raise ValueError("Encoder already contains quantization nodes")
    if not any(op.domain == "" and op.version >= 23 for op in model.opset_import):
        raise ValueError("FP8 patching requires an opset-23 or newer export")
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    types = {
        v.name: v.type.tensor_type.elem_type for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    types.update({v.name: v.data_type for v in model.graph.initializer})
    modules = encoder_modules(model, family)
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Transpose" and node.input[0] in constants:
            perm = next((list(a.ints) for a in node.attribute if a.name == "perm"), None)
            constants[node.output[0]] = constants[node.input[0]].transpose(perm)
    selected = calibration["selected_layers"]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Calibration must select unique encoder layers")
    missing = set(selected) - set(modules.values())
    if missing:
        raise ValueError(f"Selected layers missing from exported graph: {sorted(missing)}")
    selected = set(selected)
    dtype = {"float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16, "bfloat16": TensorProto.BFLOAT16}[
        calibration["dtype"]
    ]
    if any(v.type.tensor_type.elem_type != dtype for v in model.graph.input):
        raise ValueError("Encoder input precision does not match calibration")
    names = {v.name for v in model.graph.initializer} | {
        name for n in model.graph.node for name in (*n.input, *n.output)
    }
    nodes = []
    for index, node in enumerate(model.graph.node):
        module = modules.get(index)
        if module in selected:
            for input_index, kind in ((0, "input"), (1, "weight")):
                if weight_only and kind == "input":
                    continue
                key = f"{module}.{kind}_quantizer"
                amax = calibration["amax"].get(key)
                if not isinstance(amax, (int, float)) or not np.isfinite(amax) or amax <= 0:
                    raise ValueError(f"Invalid or missing amax: {key}")
                original = node.input[input_index]
                tensor_dtype = types.get(original)
                if tensor_dtype != dtype:
                    raise ValueError(f"Layer precision differs from calibration: {module}/{kind}")
                scale = np.asarray(amax / 448.0, dtype=helper.tensor_dtype_to_np_dtype(tensor_dtype))
                if not np.isfinite(float(scale)) or float(scale) <= 0:
                    raise ValueError(f"Scale is not representable in the graph precision: {key}")
                prefix = f"sam_fp8_{index}_{kind}"
                if any(prefix + suffix in names for suffix in ("_scale", "_zero", "_q", "_dq")):
                    raise ValueError(f"Generated name collision: {prefix}")
                model.graph.initializer.extend(
                    [
                        numpy_helper.from_array(scale, prefix + "_scale"),
                        helper.make_tensor(prefix + "_zero", TensorProto.FLOAT8E4M3FN, [], bytes([0]), raw=True),
                    ]
                )
                if kind == "weight":
                    if original not in constants:
                        raise ValueError(f"Cannot fold nonconstant encoder weight: {module}")
                    zero = np.zeros((), dtype=helper.tensor_dtype_to_np_dtype(TensorProto.FLOAT8E4M3FN))
                    quantized = load_op("", "QuantizeLinear", 23).eval(constants[original], scale, zero)
                    model.graph.initializer.append(numpy_helper.from_array(quantized, prefix + "_q"))
                else:
                    nodes.append(
                        helper.make_node(
                            "QuantizeLinear",
                            [original, prefix + "_scale", prefix + "_zero"],
                            [prefix + "_q"],
                            name=prefix + "_Q",
                        )
                    )
                nodes.append(
                    helper.make_node(
                        "DequantizeLinear",
                        [prefix + "_q", prefix + "_scale", prefix + "_zero"],
                        [prefix + "_dq"],
                        name=prefix + "_DQ",
                    )
                )
                node.input[input_index] = prefix + "_dq"
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=("sam2", "sam3"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Source model bundle")
    parser.add_argument(
        "--calibration", type=Path, required=True, help="Exact calibration.json from ModelOpt calibration"
    )
    parser.add_argument("--output", type=Path, required=True, help="New output bundle")
    parser.add_argument("--weight-only", action="store_true")
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Use a new output directory separate from the source bundle")
    import onnx

    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    validate_calibration(metadata, calibration, args.model_family)
    graph = "image_encoder" if args.model_family == "sam2" else "vision_encoder"
    if graph not in metadata["graphs"]:
        raise ValueError(f"Bundle does not contain {graph}")
    entry = metadata["graphs"][graph] if isinstance(metadata["graphs"], dict) else {}
    filename = entry.get("file", graph + ".onnx")
    path = (source / filename).resolve()
    if not path.is_relative_to(source):
        raise ValueError("Encoder path escapes source bundle")
    model = insert_qdq(onnx.load(path), calibration, args.model_family, weight_only=args.weight_only)
    # Use a new sidecar name to avoid appending into copied dense weights.
    sidecar = path.name + ".fp8.data"
    if (path.parent / sidecar).exists():
        raise ValueError(f"Output sidecar already exists: {sidecar}")
    manifest = {
        "format": "FP8 E4M3FN Q/DQ",
        "graph": graph,
        "weight_only": args.weight_only,
        "calibration": calibration,
        "validation": "runtime parity and performance unverified",
    }
    # Publish only a checked bundle; failures clean up this command's staging area.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sam-fp8-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        shutil.copytree(source, staging)
        target = staging / path.relative_to(source)
        onnx.save_model(
            model,
            target,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=sidecar,
            size_threshold=1024,
        )
        onnx.checker.check_model(str(target), full_check=True)
        from sam_onnx_optimization import fold_graph

        manifest["constant_folding"] = fold_graph(target)
        if isinstance(metadata["graphs"], dict):
            metadata["graphs"][graph]["constant_folding"] = manifest["constant_folding"]
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (staging / "fp8_quantization.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )
        staging.rename(output)
    print(f"Patched {len(calibration['selected_layers'])} encoder layers: {output}")


if __name__ == "__main__":
    main()
