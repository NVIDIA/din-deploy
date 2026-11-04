# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lower constant-index ONNX split sequences into tensor-only Slice/Gather nodes."""

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def lower_sequences(path: Path) -> int:
    model = onnx.load(path, load_external_data=False)
    constants = {}
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            constants[tensor.name] = numpy_helper.to_array(tensor)
    for node in model.graph.node:
        if node.op_type == "Constant":
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if "value" in attrs:
                constants[node.output[0]] = numpy_helper.to_array(attrs["value"])
    splits = {node.output[0]: node for node in model.graph.node if node.op_type == "SplitToSequence"}
    if not splits:
        return 0
    for node in model.graph.node:
        if any(name in splits for name in node.input) and node.op_type != "SequenceAt":
            raise ValueError(f"Unsupported sequence consumer: {node.op_type}")
    nodes = []
    lowered = 0
    for node in model.graph.node:
        if node.op_type == "SplitToSequence":
            continue
        if node.op_type != "SequenceAt":
            nodes.append(node)
            continue
        split = splits[node.input[0]]
        index = int(constants[node.input[1]].item())
        if index < 0:
            raise ValueError("Negative sequence indices need an explicit split length")
        attrs = {a.name: helper.get_attribute_value(a) for a in split.attribute}
        axis, keepdims = attrs.get("axis", 0), attrs.get("keepdims", 1)
        chunks = constants[split.input[1]] if len(split.input) > 1 and split.input[1] else np.array(1)
        if chunks.ndim == 0:
            start = index * int(chunks)
            end = start + int(chunks)
        else:
            start = int(chunks[:index].sum())
            end = start + int(chunks[index])
        prefix = f"din_split_{lowered}"
        if not keepdims:
            if end - start != 1:
                raise ValueError("Cannot squeeze a split with length other than one")
            idx_name = prefix + "_index"
            model.graph.initializer.append(numpy_helper.from_array(np.array(start, dtype=np.int64), idx_name))
            nodes.append(
                helper.make_node("Gather", [split.input[0], idx_name], list(node.output), axis=axis, name=prefix)
            )
        else:
            names = []
            for suffix, value in (("start", start), ("end", end), ("axis", axis)):
                name = prefix + "_" + suffix
                model.graph.initializer.append(numpy_helper.from_array(np.array([value], dtype=np.int64), name))
                names.append(name)
            nodes.append(helper.make_node("Slice", [split.input[0], *names], list(node.output), name=prefix))
        lowered += 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    value_info = [value for value in model.graph.value_info if value.name not in splits]
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)
    temporary = path.with_suffix(".lowered.onnx")
    onnx.save(model, temporary)
    onnx.checker.check_model(str(temporary))
    temporary.replace(path)
    return lowered


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", type=Path, nargs="+")
    for graph in parser.parse_args().graphs:
        print(f"{graph.name}: lowered {lower_sequences(graph)} sequence accesses")
