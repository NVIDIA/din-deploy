# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable fixed-capacity SAM attention contract, preserving native Attention nodes.

Validity is per history frame, not per query: no quadratic Q x K mask is built.
The source model's weights, rotary positions and precision boundaries are retained.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CONTRACT_VERSION = 1


def memory_cross_attentions(model, memory_names):
    inputs = {value.name for value in model.graph.input}
    producers = {name: node for node in model.graph.node for name in node.output}
    cache = {}

    def sources(name):
        if name not in cache:
            node = producers.get(name)
            if node is None:
                cache[name] = {name} if name in inputs else set()
            elif node.op_type == "Attention":
                cache[name] = set()
            else:
                cache[name] = set().union(*(sources(i) for i in node.input))
        return cache[name]

    return [n for n in model.graph.node if n.op_type == "Attention" and sources(n.input[2]) & memory_names]


def specialize_memory_attention(path: Path, family: str) -> dict:
    if family not in ("sam2", "sam3"):
        raise ValueError(f"Unknown SAM family: {family}")
    model = onnx.load(path)
    graph = model.graph
    if any(v.name == "memory_valid" for v in graph.input):
        raise ValueError("Model already has a validity contract")
    spatial_tokens = 4096 if family == "sam2" else 5184
    memory_names = {"memory", "memory_pos"} if family == "sam2" else {
        "memory_image", "memory_image_position", "memory", "memory_position",
    }
    pointer_names = {"object_pointers", "pointer_position"} if family == "sam3" else set()
    inputs = {value.name: value for value in graph.input}
    if not memory_names.issubset(inputs) or not pointer_names.issubset(inputs):
        raise ValueError("Unexpected source memory-attention interface")
    for name in memory_names | pointer_names:
        inputs[name].type.tensor_type.shape.dim[0].dim_value = (
            256 if name in pointer_names else 7 * spatial_tokens
        )

    prefix = "sam_static_"
    nodes = []

    def constant(name, data):
        graph.initializer.append(numpy_helper.from_array(np.asarray(data), prefix + name))
        return prefix + name

    def node(op, args, name, **attrs):
        result = prefix + name
        nodes.append(helper.make_node(op, args, [result], name=result, **attrs))
        return result

    graph.input.append(helper.make_tensor_value_info("memory_valid", TensorProto.BOOL, [7]))
    spatial = node("Gather", ["memory_valid", constant("spatial_indices", np.repeat(np.arange(7, dtype=np.int64), spatial_tokens))], "spatial", axis=0)
    valid = spatial
    if family == "sam3":
        graph.input.append(helper.make_tensor_value_info("pointer_valid", TensorProto.BOOL, [16]))
        pointers = node("Gather", ["pointer_valid", constant("pointer_indices", np.repeat(np.arange(16, dtype=np.int64), 16))], "pointers", axis=0)
        valid = node("Concat", [spatial, pointers], "valid", axis=0)

    # A safe sentinel avoids an all-masked softmax for speculative propagation.
    # The runtime must discard that candidate while the sequence is uninitialized.
    count = node("ReduceSum", [node("Cast", [valid], "valid_int", to=TensorProto.INT64)], "valid_count", keepdims=0)
    empty = node("Equal", [count, constant("zero", np.array(0, dtype=np.int64))], "empty")
    token_count = 7 * spatial_tokens + (256 if family == "sam3" else 0)
    sentinel = np.zeros(token_count, dtype=np.bool_)
    sentinel[0] = True
    safe = node("Or", [valid, node("And", [empty, constant("sentinel", sentinel)], "sentinel_valid")], "safe")
    mask = node("Reshape", [safe, constant("mask_shape", np.array([1, 1, 1, token_count], dtype=np.int64))], "mask")
    cross_count = 0
    for attention in memory_cross_attentions(model, memory_names):
        while len(attention.input) < 4:
            attention.input.append("")
        if attention.input[3]:
            raise ValueError("Source cross-attention already has a mask; refusing to overwrite it")
        attention.input[3] = mask
        cross_count += 1
    if cross_count != 4:
        raise ValueError(f"Expected four memory cross-attention nodes, found {cross_count}")
    original = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes + original)
    # Original symbolic intermediate shapes describe the dynamic source model.
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    # TODO: Investigate alternative history/graph I/O precision only with sequence
    # parity checks and explicit NVIDIA EP / portable ONNX consumer support. Keep
    # the existing FP16 SAM2 and FP32 SAM3 interfaces unchanged for this contract.
    sidecar = path.with_name(path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()  # Loaded above; do not append another copy of the weights.
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=path.name + ".data", size_threshold=1024)
    return {"version": CONTRACT_VERSION, "spatial_frames": 7, "spatial_tokens_per_frame": spatial_tokens,
            "pointer_frames": 16 if family == "sam3" else 0, "pointer_tokens_per_frame": 16 if family == "sam3" else 0,
            "mask": "boolean broadcast [1,1,1,K]; true means attend; safe first-token sentinel for empty history",
            "masked_attention_nodes": cross_count}


def specialize_fixtures(path: Path, family: str) -> None:
    """Pad existing reference inputs; valid-token reference outputs stay unchanged."""
    fixture = path.with_suffix(".inputs.npz")
    if not fixture.exists():
        return
    with np.load(fixture) as values:
        inputs = {name: value.copy() for name, value in values.items()}
    spatial_tokens = 4096 if family == "sam2" else 5184
    n = inputs["memory"].shape[0] // spatial_tokens
    if not 1 <= n <= 7:
        raise ValueError("Fixture memory occupancy must be within capacity")
    memory_names = ("memory", "memory_pos") if family == "sam2" else (
        "memory_image", "memory_image_position", "memory", "memory_position",
    )
    for name in memory_names:
        value = inputs[name]
        inputs[name] = np.pad(value, ((0, (7 - n) * spatial_tokens), (0, 0), (0, 0)))
    inputs["memory_valid"] = np.arange(7) < n
    if family == "sam3":
        p = inputs["object_pointers"].shape[0] // 16
        for name in ("object_pointers", "pointer_position"):
            inputs[name] = np.pad(inputs[name], ((0, (16 - p) * 16), (0, 0), (0, 0)))
        inputs["pointer_valid"] = np.arange(16) < p
    np.savez(fixture, **inputs)
    raw = path.parent / (path.stem + ".fixtures")
    raw.mkdir(exist_ok=True)
    for name, value in inputs.items():
        value.tofile(raw / (name + ".bin"))


def cpu_mask_compatibility(path: Path, family: str, chunk_size: int = 128) -> None:
    """Work around ORT CPU requiring mask Q length to equal the query length.

    Chunk the query instead of expanding a full quadratic mask. GPU bundles do
    not use this compatibility lowering; their fused Attention is untouched.
    """
    model = onnx.load(path)
    q_count = 4096 if family == "sam2" else 5184
    k_count = 7 * q_count + (256 if family == "sam3" else 0)
    shapes = {v.name: v.type.tensor_type.shape for v in (*model.graph.input, *model.graph.value_info)}
    nodes = []
    for attention in model.graph.node:
        if attention.op_type != "Attention" or len(attention.input) < 4 or attention.input[3] != "sam_static_mask":
            nodes.append(attention)
            continue
        query_shape = shapes.get(attention.input[0])
        query_count = query_shape.dim[2].dim_value if query_shape and len(query_shape.dim) == 4 else q_count
        query_count = query_count or q_count
        chunks = []
        for start in range(0, query_count, chunk_size):
            count = min(chunk_size, query_count - start)
            prefix = attention.name + f"_cpu_q{start}"
            def constant(suffix, data, prefix=prefix):
                name = prefix + suffix
                model.graph.initializer.append(numpy_helper.from_array(np.array(data, dtype=np.int64), name))
                return name
            query = prefix + "_query"
            mask = prefix + "_mask"
            nodes.append(helper.make_node("Slice", [attention.input[0], constant("_start", [start]),
                constant("_end", [start + count]), constant("_axis", [2])], [query]))
            nodes.append(helper.make_node("Expand", [attention.input[3], constant("_shape", [1, 1, count, k_count])], [mask]))
            part = onnx.NodeProto()
            part.CopyFrom(attention)
            part.name = prefix
            part.input[0], part.input[3] = query, mask
            if len(part.output) != 1:
                raise ValueError("CPU compatibility lowering requires one Attention output")
            part.output[0] = prefix + "_output"
            nodes.append(part)
            chunks.append(part.output[0])
        nodes.append(helper.make_node("Concat", chunks, [attention.output[0]], axis=2))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    sidecar = path.with_name(path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=path.name + ".data", size_threshold=1024)
