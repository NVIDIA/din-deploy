# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

MAX_FP8_E4M3 = 448.0


def main() -> None:
    args = parse_args()
    quantizers = load_quant_summary(args.quant_summary)
    model = onnx.load(args.input)
    patched = insert_qdq(model, quantizers, patch_weights=args.patch_weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(patched)
    onnx.save_model(patched, args.output, save_as_external_data=False)
    print(f"Inserted {count_qdq_nodes(patched)} official ONNX FP8 QDQ nodes")
    print(f"Exported {args.output}")


def load_quant_summary(path: Path) -> dict[str, float]:
    quantizers: dict[str, float] = {}
    pattern = re.compile(r"amax=(?P<amax>[0-9.eE+-]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        if "TensorQuantizer(disabled)" in line or " quant" not in line:
            continue
        match = pattern.search(line)
        if match:
            quantizers[line.split()[0]] = float(match.group("amax"))
    return quantizers


def insert_qdq(model: onnx.ModelProto, quantizers: dict[str, float], *, patch_weights: bool) -> onnx.ModelProto:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    nodes_by_output = {output: node for node in model.graph.node for output in node.output}
    linear_modules = linear_modules_by_matmul_output(model, nodes_by_output)

    new_nodes: list[onnx.NodeProto] = []
    inserted = 0
    for node in model.graph.node:
        if node.op_type == "MatMul" and node.output and node.output[0] in linear_modules:
            module = linear_modules[node.output[0]]
            inserted += patch_node_input(model, new_nodes, node, 0, module, "input", quantizers)
            if patch_weights:
                inserted += patch_node_input(model, new_nodes, node, 1, module, "weight", quantizers)
        elif node.op_type == "Conv" and len(node.input) >= 2 and node.input[1].endswith(".weight"):
            module = node.input[1].removeprefix("image_encoder.").removesuffix(".weight")
            module = f"model.image_encoder.{module}"
            inserted += patch_node_input(model, new_nodes, node, 0, module, "input", quantizers)
            if patch_weights and node.input[1] in initializer_names:
                inserted += patch_node_input(model, new_nodes, node, 1, module, "weight", quantizers)
        new_nodes.append(node)

    if inserted == 0:
        raise RuntimeError("No QDQ nodes were inserted; check quant summary/model naming.")
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    ensure_opset(model, "", 23)
    return model


def linear_modules_by_matmul_output(
    model: onnx.ModelProto, nodes_by_output: dict[str, onnx.NodeProto]
) -> dict[str, str]:
    modules: dict[str, str] = {}
    for node in model.graph.node:
        if node.op_type != "Add" or len(node.input) < 2:
            continue
        bias = next((name for name in node.input if name.endswith(".bias")), None)
        if bias is None:
            continue
        matmul_output = next((name for name in node.input if nodes_by_output.get(name) is not None), None)
        if matmul_output is None or nodes_by_output[matmul_output].op_type != "MatMul":
            continue
        module = bias.removeprefix("image_encoder.").removesuffix(".bias")
        modules[matmul_output] = f"model.image_encoder.{module}"
    return modules


def patch_node_input(
    model: onnx.ModelProto,
    new_nodes: list[onnx.NodeProto],
    node: onnx.NodeProto,
    input_index: int,
    module: str,
    kind: str,
    quantizers: dict[str, float],
) -> int:
    quantizer_name = f"{module}.{kind}_quantizer"
    amax = quantizers.get(quantizer_name)
    if amax is None:
        return 0

    original = node.input[input_index]
    prefix = sanitize(f"{node.name or node.op_type}_{input_index}_{kind}_fp8")
    scale_name = f"{prefix}_scale"
    zero_name = f"{prefix}_zero_point"
    quantized = f"{prefix}_quantized"
    dequantized = f"{prefix}_dequantized"

    model.graph.initializer.append(
        helper.make_tensor(
            scale_name,
            TensorProto.FLOAT16,
            [],
            raw=True,
            vals=np.asarray(amax / MAX_FP8_E4M3, dtype=np.float16).tobytes(),
        )
    )
    model.graph.initializer.append(
        helper.make_tensor(zero_name, TensorProto.FLOAT8E4M3FN, [], raw=True, vals=bytes([0]))
    )
    new_nodes.append(
        helper.make_node(
            "QuantizeLinear", [original, scale_name, zero_name], [quantized], name=f"{prefix}_QuantizeLinear"
        )
    )
    new_nodes.append(
        helper.make_node(
            "DequantizeLinear", [quantized, scale_name, zero_name], [dequantized], name=f"{prefix}_DequantizeLinear"
        )
    )
    node.input[input_index] = dequantized
    return 2


def ensure_opset(model: onnx.ModelProto, domain: str, version: int) -> None:
    for opset in model.opset_import:
        if opset.domain == domain:
            opset.version = max(opset.version, version)
            return
    model.opset_import.append(helper.make_opsetid(domain, version))


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def count_qdq_nodes(model: onnx.ModelProto) -> int:
    return sum(node.op_type in {"QuantizeLinear", "DequantizeLinear"} for node in model.graph.node)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insert official ONNX FP8 QDQ into SAM2 encoder ONNX.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/sam2/onnx/image_encoder.onnx"))
    parser.add_argument("--quant-summary", type=Path, default=Path("artifacts/sam2/fp8_encoder/quant_summary.txt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/sam2/onnx_fp8_qdq/image_encoder.onnx"))
    parser.add_argument("--patch-weights", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
