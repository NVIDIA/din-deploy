# SPDX-License-Identifier: Apache-2.0
"""Export Nagisa's word segmenter as an FP32 ONNX LSTM; no POS model is needed."""

import gzip
import hashlib
import json
import pickle
import re
from array import array
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper

REVISION = "3c4bb48d3ba7451e3314b35337c79f1256ade0cf"
HASHES = {
    "dict": "968ac9e6c7a53051ef24d8561673dd31de81b2feb9b5bff01b1d3b6b2473113c",
    "hp": "6737f76b588315fe2fe05d05c99939d3142a0f1c6468f49f4c23e7003192f204",
    "model": "9db9abc06a927c56e18af8d485e20a14908138752be459a83f0dc7ac85368c1b",
}


def export_japanese(output, source=None):
    source = Path(source or Path(torch.hub.get_dir()) / "nagisa" / REVISION)
    source.mkdir(parents=True, exist_ok=True)
    for suffix, digest in HASHES.items():
        path = source / f"nagisa_v001.{suffix}"
        if not path.exists():
            torch.hub.download_url_to_file(
                f"https://raw.githubusercontent.com/taishi-i/nagisa/{REVISION}/nagisa/data/{path.name}",
                str(path),
                hash_prefix=digest,
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unexpected Nagisa asset: {path}")
    # Only load the authenticated, pinned upstream dictionaries above.
    vocabs = pickle.loads(gzip.decompress((source / "nagisa_v001.dict").read_bytes()))
    hp = pickle.loads(gzip.decompress((source / "nagisa_v001.hp").read_bytes()))
    parameters, lookups = [], []
    with (source / "nagisa_v001.model").open("rb") as file:
        while header := file.readline():
            match = re.fullmatch(rb"#(Parameter|LookupParameter)# (\S+) \{([\d,]+)\} (\d+) ZERO_GRAD\s*", header)
            if not match:
                raise ValueError("Unsupported Nagisa parameter layout")
            kind, name, dimensions, count = match.groups()
            shape = [int(n) for n in dimensions.split(b",")]
            data = torch.tensor(array("f", (float(n) for n in file.read(int(count)).split())), dtype=torch.float32)
            if len(shape) == 2:
                data = data.reshape(shape[::-1])
                if kind == b"Parameter":
                    data = data.T
            (parameters if kind == b"Parameter" else lookups).append((name.decode(), data.contiguous()))

    nodes, constants = [], []

    def const(name, data):
        data = data.contiguous().clone()
        tensor = TensorProto(name=name, data_type=TensorProto.INT64 if data.dtype == torch.int64 else TensorProto.FLOAT)
        tensor.dims.extend(data.shape)
        tensor.raw_data = bytes(data.untyped_storage())
        constants.append(tensor)
        return name

    def node(op, inputs, name, **attributes):
        nodes.append(helper.make_node(op, inputs, [name], **attributes))
        return name

    width = hp["WINDOW_SIZE"]
    inputs, pieces = [], []
    word_table = torch.cat([lookups[2][1], torch.zeros(1, hp["DIM_WORD"])])
    const("word_table", word_table)
    const("axis1", torch.tensor([1]))
    for name, table in [("unigrams", lookups[0][1]), ("bigrams", lookups[1][1]), ("types", lookups[3][1])]:
        inputs.append(helper.make_tensor_value_info(name, TensorProto.INT64, ["characters", width]))
        gathered = node("Gather", [const(name + "_table", table), name], name + "_vectors", axis=0)
        pieces.append(node("Flatten", [gathered], name + "_flat", axis=1))
    for name in ("word_starts", "word_ends"):
        inputs.append(helper.make_tensor_value_info(name, TensorProto.INT64, ["characters", 8]))
        gathered = node("Gather", ["word_table", name], name + "_vectors", axis=0)
        pieces.append(node("ReduceSum", [gathered, "axis1"], name + "_sum", keepdims=0))
    x = node("Concat", pieces, "features", axis=1)
    x = node("Unsqueeze", [x, "axis1"], "sequence")
    hidden = hp["DIM_HIDDEN"] // 2
    # DyNet gates i,f,o,g -> ONNX gates i,o,f,g; DyNet adds +1 to the forget bias.
    order = torch.cat([torch.arange(i * hidden, (i + 1) * hidden) for i in (0, 2, 1, 3)])
    for layer in range(hp["LAYERS"]):
        weights, recurrent, biases = [], [], []
        for direction in range(2):
            offset = (2 * layer + direction) * 3
            wx, wh, bias = [p[1] for p in parameters[offset : offset + 3]]
            bias = bias.clone()
            bias[hidden : 2 * hidden] += 1
            weights.append(wx[order])
            recurrent.append(wh[order])
            biases.append(torch.cat([bias[order], torch.zeros(4 * hidden)]))
        name = f"lstm_{layer}"
        x = node(
            "LSTM",
            [
                x,
                const(name + "_w", torch.stack(weights)),
                const(name + "_r", torch.stack(recurrent)),
                const(name + "_b", torch.stack(biases)),
            ],
            name,
            hidden_size=hidden,
            direction="bidirectional",
        )
        x = node("Transpose", [x], name + "_ordered", perm=[0, 2, 1, 3])
        x = node("Reshape", [x, const(name + "_shape", torch.tensor([-1, 1, 2 * hidden]))], name + "_flat")
    plain = [p for p in parameters if p[0].count("/") == 1]
    x = node("Squeeze", [x, "axis1"], "hidden")
    x = node("MatMul", [x, const("projection", plain[0][1].T)], "projected")
    node("Add", [x, const("bias", plain[1][1])], "emissions")
    graph = helper.make_graph(
        nodes,
        "nagisa_word_segmentation",
        inputs,
        [helper.make_tensor_value_info("emissions", TensorProto.FLOAT, ["characters", 6])],
        constants,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "japanese.onnx"
    path.with_suffix(".onnx.data").unlink(missing_ok=True)
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="japanese.onnx.data",
        size_threshold=1024,
    )
    onnx.checker.check_model(str(path))
    metadata = {
        "revision": REVISION,
        "window": width,
        "unigrams": vocabs[0],
        "bigrams": vocabs[1],
        "words": vocabs[2],
        "padding_word": len(word_table) - 1,
        "transitions": lookups[5][1].tolist(),
    }
    (output / "japanese.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    (output / "japanese.LICENSE.txt").write_text(
        Path(__file__).with_name("nagisa.LICENSE.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )
