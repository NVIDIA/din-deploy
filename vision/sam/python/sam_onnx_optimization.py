# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constant-fold SAM graphs while preserving runtime inputs and Q/DQ boundaries."""

from collections import Counter


def _fold_static_split_and_bf16_matmul(model):
    """Cover evaluator gaps in ONNXScript without changing dynamic computation."""
    import numpy as np
    import onnx
    from onnx import numpy_helper
    from onnx.reference import ReferenceEvaluator

    inputs = {value.name for value in model.graph.input}
    constants = {tensor.name: tensor for tensor in model.graph.initializer if tensor.name not in inputs}
    opsets = {op.domain: op.version for op in model.opset_import}
    nodes = []
    count = 0
    for node in model.graph.node:
        if (
                node.domain
                or node.op_type not in ("Split", "MatMul")
                or any(name not in constants for name in node.input if name)
        ):
            nodes.append(node)
            continue
        if node.op_type == "MatMul" and any(
                constants[name].data_type != onnx.TensorProto.BFLOAT16 for name in node.input
        ):
            nodes.append(node)
            continue
        values = {name: numpy_helper.to_array(constants[name]) for name in node.input if name}
        if node.op_type == "Split":
            outputs = ReferenceEvaluator(node, opsets=opsets).run(None, values)
        else:
            import ml_dtypes
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tensors = [
                torch.from_numpy(np.array(values[name].view(np.uint16), copy=True)).view(torch.bfloat16).to(device)
                for name in node.input
            ]
            with torch.inference_mode():
                value = (tensors[0] @ tensors[1]).cpu().view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
            outputs = [value]
        for name, value in zip(node.output, outputs, strict=True):
            tensor = numpy_helper.from_array(value, name)
            model.graph.initializer.append(tensor)
            constants[name] = tensor
        count += 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return count


def fold_graph(path):
    import onnx
    from onnx_ir.passes import common
    from onnxscript import ir, optimizer

    def fold(model):
        model_ir = ir.serde.deserialize_model(model)
        for _ in range(3):
            optimizer.fold_constants(
                model_ir, onnx_shape_inference=True, input_size_limit=1 << 26, output_size_limit=1 << 26
            )
            common.LiftConstantsToInitializersPass(lift_all_constants=True, size_limit=0)(model_ir)
            common.RemoveUnusedNodesPass()(model_ir)
        common.NameFixPass()(model_ir)
        return ir.serde.serialize_model(model_ir)

    header = onnx.load(path, load_external_data=False)
    previous_sidecars = {
        entry.value for tensor in header.graph.initializer for entry in tensor.external_data if entry.key == "location"
    }
    model = onnx.load(path)
    inputs = [(value.name, value.type.SerializeToString()) for value in model.graph.input]
    outputs = [(value.name, value.type.SerializeToString()) for value in model.graph.output]
    before = Counter(node.op_type for node in model.graph.node)
    # The default 8192-element input limit skips positional tensors and weights.
    model = fold(model)
    while _fold_static_split_and_bf16_matmul(model):
        model = fold(model)
    after = Counter(node.op_type for node in model.graph.node)
    if inputs != [(value.name, value.type.SerializeToString()) for value in model.graph.input] or outputs != [
        (value.name, value.type.SerializeToString()) for value in model.graph.output
    ]:
        raise ValueError(f"Constant folding changed graph interfaces: {path}")
    for op in ("Attention", "QuantizeLinear", "DequantizeLinear"):
        if before[op] != after[op]:
            raise ValueError(f"Constant folding changed {op} boundaries: {path}")
    # Use a fresh sidecar so saving never appends to the original external data.
    sidecar = path.name + ".folded.data"
    target = path.with_name(sidecar)
    if target.exists():
        target.unlink()
    onnx.save_model(
        model, path, save_as_external_data=True, all_tensors_to_one_file=True, location=sidecar, size_threshold=1024
    )
    onnx.checker.check_model(str(path), full_check=True)
    for location in previous_sidecars - {sidecar}:
        previous = path.parent / location
        if previous.parent.resolve() == path.parent.resolve() and previous.name.startswith(path.name + "."):
            previous.unlink(missing_ok=True)
    print(f"  Constant folding {path.name}: {sum(before.values())} -> {sum(after.values())} nodes", flush=True)
    return {
        "nodes_before": sum(before.values()),
        "nodes_after": sum(after.values()),
        "operators_before": dict(before),
        "operators_after": dict(after),
    }
