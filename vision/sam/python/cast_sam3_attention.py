# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a separate native-BF16 attention variant while preserving graph interfaces."""

import argparse
import json
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def cast_attention(source, output):
    model = onnx.load(source)
    if not any(node.op_type == "Attention" for node in model.graph.node):
        shutil.copy2(source, output)
        sidecar = source.with_name(source.name + ".data")
        if sidecar.exists():
            shutil.copy2(sidecar, output.with_name(output.name + ".data"))
        return 0
    model = onnx.shape_inference.infer_shapes(model)
    values = {value.name: value for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
    types = {name: value.type.tensor_type.elem_type for name, value in values.items()}
    types.update({value.name: value.data_type for value in model.graph.initializer})
    nodes = []
    count = 0
    for node in model.graph.node:
        if node.op_type != "Attention":
            nodes.append(node)
            continue
        count += 1
        original_type = types[node.input[0]]
        for index in range(min(4, len(node.input))):
            name = node.input[index]
            if not name or types[name] in (TensorProto.BOOL, TensorProto.BFLOAT16):
                continue
            if types[name] not in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE):
                raise ValueError(f"Unexpected Attention input type for {name}: {types[name]}")
            cast_name = node.name + f"_bf16_input_{index}"
            nodes.append(helper.make_node("Cast", [name], [cast_name], to=TensorProto.BFLOAT16, name=cast_name))
            node.input[index] = cast_name
        if original_type != TensorProto.BFLOAT16:
            original_output = node.output[0]
            bf16_output = node.name + "_bf16_output"
            node.output[0] = bf16_output
            nodes.append(node)
            nodes.append(helper.make_node("Cast", [bf16_output], [original_output], to=original_type, name=node.name + "_restore_dtype"))
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Re-infer intermediates after inserting precision boundaries.
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(model, output, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=output.name + ".data", size_threshold=1024)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Use a separate output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    counts = {}
    for source in sorted(args.source.glob("*.onnx")):
        counts[source.stem] = cast_attention(source, args.output / source.name)
        print(source.name, counts[source.stem], flush=True)
    for filename in ("metadata.json", "tokenizer.json", "tokenizer_tests.json", "temporal_constants.json"):
        if (args.source / filename).exists():
            shutil.copy2(args.source / filename, args.output / filename)
    metadata_path = args.output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["attention_precision"] = "bfloat16; original graph interfaces and surrounding precision retained"
    metadata.setdefault("graph_precision", {}).update(
        {name: "mixed_bfloat16" for name, count in counts.items() if count}
    )
    metadata["precision_validation"] = "pending"
    metadata["attention_nodes"] = counts
    metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
