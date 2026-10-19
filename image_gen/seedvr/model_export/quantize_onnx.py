# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantize SeedVR2 DiT linear operations using Comfy-Org FP8/NVFP4 weights.

Consumes the named Gemm weights from export_seedvr.py (opset >= 23).
FP8 is weight-only, with unit weight scales and unchanged activations.
NVFP4 preserves checkpoint weight
scales and uses a unit global activation scale when none is saved.
Nonlinear parameters, biases, VAE models, and conditioning remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import onnx
import torch
from onnx import TensorProto as T, helper
from safetensors import safe_open

REPO = "Comfy-Org/SeedVR2"
REVISION = "df48879708206a403d2a61acd55578c2e80fd233"
FILES = {"fp8_wo": "seedvr2_3b_fp8_e4m3fn.safetensors", "nvfp4": "seedvr2_3b_nvfp4.safetensors"}
DEFAULT_MODEL_DIR = Path("models/SeedVR2-3B-onnx")
DTYPES = {T.FLOAT16: torch.float16, T.BFLOAT16: torch.bfloat16, T.FLOAT: torch.float32}
PREFIX = "__seedvr_quant."


def tensor_proto(name, tensor, dtype=None, shape=None):
    tensor = tensor.detach().cpu().contiguous()
    dtype = dtype or {torch.float32: T.FLOAT, torch.float16: T.FLOAT16,
                      torch.bfloat16: T.BFLOAT16, torch.float8_e4m3fn: T.FLOAT8E4M3FN}[tensor.dtype]
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return helper.make_tensor(name, dtype, list(tensor.shape) if shape is None else shape, raw, raw=True)


class Writer:
    """Stream one tensor at a time; never load the baseline's multi-GB data."""
    def __init__(self, stream, location):
        self.stream, self.location = stream, location

    def reference(self, tensor, offset, length):
        tensor.ClearField("raw_data")
        del tensor.external_data[:]
        tensor.data_location = T.EXTERNAL
        for key, value in (("location", self.location), ("offset", str(offset)), ("length", str(length))):
            tensor.external_data.add(key=key, value=value)

    def write(self, tensor):
        offset = self.stream.tell()
        length = self.stream.write(tensor.raw_data)
        self.reference(tensor, offset, length)
        return tensor

    def copy(self, tensor, input_path):
        if tensor.data_location != T.EXTERNAL:
            return
        fields = {v.key: v.value for v in tensor.external_data}
        if "location" not in fields or "length" not in fields:
            raise ValueError(f"Incomplete external data for {tensor.name}")
        length = int(fields["length"])
        offset = self.stream.tell()
        with (input_path.parent / fields["location"]).open("rb") as source:
            source.seek(int(fields.get("offset", 0)))
            remaining = length
            while remaining:
                chunk = source.read(min(remaining, 16 * 1024 * 1024))
                if not chunk:
                    raise EOFError(f"Truncated external data for {tensor.name}")
                self.stream.write(chunk)
                remaining -= len(chunk)
        self.reference(tensor, offset, length)


def select_weights(model, checkpoint, quant):
    keys = set(checkpoint.keys())
    if quant == "nvfp4":
        selected = set()
        for key in keys:
            if key.endswith(".comfy_quant"):
                descriptor = json.loads(bytes(checkpoint.get_tensor(key).tolist()).decode())
                if descriptor != {"format": "nvfp4"}:
                    raise ValueError(f"Unsupported quantization descriptor: {key}: {descriptor}")
                selected.add(key.removesuffix(".comfy_quant") + ".weight")
    else:
        selected = {k for k in keys if k.endswith(".weight") and
                    len(checkpoint.get_slice(k).get_shape()) == 2}
    if not selected:
        raise ValueError("Checkpoint contains no quantizable linear weights")
    initializers = {x.name: x for x in model.graph.initializer}
    mapping = {}
    for key in sorted(selected):
        name = "diffusion_model." + key
        if name not in initializers:
            raise ValueError(f"Missing named ONNX weight {name}; use export_seedvr.py's DiT export")
        initializer = initializers[name]
        shape = list(checkpoint.get_slice(key).get_shape())
        expected_dtype = "F8_E4M3" if quant == "fp8" else "U8"
        if checkpoint.get_slice(key).get_dtype() != expected_dtype:
            raise ValueError(f"Wrong checkpoint dtype for {key}, expected {expected_dtype}")
        if quant == "nvfp4":
            shape[-1] *= 2
        if list(initializer.dims) != shape or initializer.data_type not in DTYPES:
            raise ValueError(f"Shape/dtype mismatch for {name}: ONNX {list(initializer.dims)}, checkpoint {shape}")
        mapping[name] = key
    uses = Counter()
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name not in mapping:
                continue
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if (node.domain or node.op_type != "Gemm" or index != 1 or
                    attrs.get("transB", 0) != 1 or attrs.get("transA", 0) != 0):
                raise ValueError(f"Unsupported consumer of {name}: {node.name}; expected Gemm(transB=1, transA=0)")
            uses[name] += 1
    if any(x.name in mapping for x in list(model.graph.input) + list(model.graph.output)):
        raise ValueError("Quantized weights must not be graph inputs or outputs")
    if set(uses) != set(mapping):
        raise ValueError(f"Unused checkpoint weights: {set(mapping) - set(uses)}")
    return mapping, uses


def rewrite(model, checkpoint, quant, writer, input_path):
    if not any(o.domain in ("", "ai.onnx") and o.version >= 23 for o in model.opset_import):
        raise ValueError("Requires an ONNX opset >= 23 export")
    if any(x.name.startswith(PREFIX) for x in model.graph.initializer):
        raise ValueError("Model has already been quantized")
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for n in model.graph.node for a in n.attribute):
        raise ValueError("Nested graphs are unsupported")
    mapping, uses = select_weights(model, checkpoint, quant)
    originals = {x.name: x for x in model.graph.initializer}
    new_initializers, weight_nodes = [], []

    def constant(name, value, dtype=torch.float32):
        new_initializers.append(tensor_proto(name, torch.tensor(value, dtype=dtype)))
        return name

    def node(nodes, op, inputs, output, **attrs):
        nodes.append(helper.make_node(op, inputs, [output], name=output + ".node", **attrs))
        return output

    for name, key in mapping.items():
        p = PREFIX + key
        original = originals[name]
        dtype = original.data_type
        weight = checkpoint.get_tensor(key)
        if quant == "fp8":
            new_initializers.append(writer.write(tensor_proto(p + ".packed", weight)))
            scale = constant(p + ".scale", 1.0, DTYPES[dtype])
            node(weight_nodes, "DequantizeLinear", [p + ".packed", scale], name)
        else:
            from comfy_kitchen.float_utils import from_blocked, swap_nibbles
            rows, cols = original.dims
            if cols % 16:
                raise ValueError(f"NVFP4 input width must be a multiple of 16: {name}")
            base = key.removesuffix(".weight")
            global_scale = checkpoint.get_tensor(base + ".weight_scale_2")
            if global_scale.numel() != 1 or not torch.isfinite(global_scale).all() or global_scale.item() <= 0:
                raise ValueError(f"Invalid global weight scale: {base}")
            blocked = checkpoint.get_tensor(base + ".weight_scale")
            if blocked.dtype != torch.float8_e4m3fn:
                raise ValueError(f"Expected E4M3 block scales: {base}")
            scales = from_blocked(blocked, num_rows=rows, num_cols=cols // 16).contiguous()
            if not torch.isfinite(scales.float()).all() or (scales.float() < 0).any():
                raise ValueError(f"Invalid block scales: {base}")
            new_initializers.extend([
                writer.write(tensor_proto(p + ".packed", swap_nibbles(weight), T.FLOAT4E2M1, [rows, cols])),
                writer.write(tensor_proto(p + ".block_scale", scales)),
                tensor_proto(p + ".global_scale", global_scale.float().reshape(())),
            ])
            scale = node(weight_nodes, "DequantizeLinear", [p + ".block_scale", p + ".global_scale"], p + ".scale")
            dq = node(weight_nodes, "DequantizeLinear", [p + ".packed", scale], p + ".dq", axis=-1, block_size=16)
            node(weight_nodes, "Cast", [dq], name, to=dtype)

    rewritten = list(weight_nodes)
    for index, original_node in enumerate(model.graph.node):
        if quant == "nvfp4" and original_node.op_type == "Gemm" and original_node.input[1] in mapping:
            weight_name = original_node.input[1]
            p = PREFIX + f"activation.{index}"
            x = original_node.input[0]
            dtype = originals[weight_name].data_type
            # RTX requires this global scale to be an initializer. Block
            # scales are still computed dynamically by the FP4 operator.
            scale_key = mapping[weight_name].removesuffix(".weight") + ".input_scale"
            value = checkpoint.get_tensor(scale_key) if scale_key in checkpoint.keys() else torch.tensor(1.0)
            if value.numel() != 1 or not torch.isfinite(value).all() or value.item() <= 0:
                raise ValueError(f"Invalid activation scale: {scale_key}")
            scale = constant(p + ".scale", value.item())
            rewritten.append(helper.make_node("TRT_FP4DynamicQuantize", [x, scale],
                [p + ".q", p + ".block_scale"], name=p + ".quantize", domain="trt",
                axis=-1, block_size=16, scale_type=T.FLOAT8E4M3FN))
            # ORT's custom-op schema cannot infer these output types.
            model.graph.value_info.extend([
                helper.make_tensor_value_info(p + ".q", T.FLOAT4E2M1, None),
                helper.make_tensor_value_info(p + ".block_scale", T.FLOAT8E4M3FN, None),
            ])
            block_scale = node(rewritten, "DequantizeLinear", [p + ".block_scale", scale], p + ".scale_dq")
            dq = node(rewritten, "DequantizeLinear", [p + ".q", block_scale], p + ".dq", axis=-1, block_size=16)
            dq = node(rewritten, "Cast", [dq], p + ".cast", to=dtype)
            original_node.input[0] = dq
        rewritten.append(original_node)
    kept = []
    for initializer in model.graph.initializer:
        if initializer.name not in mapping:
            writer.copy(initializer, input_path)
            kept.append(initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept + new_initializers)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    if quant == "nvfp4" and not any(o.domain == "trt" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("trt", 1))
    return {"quant": quant, "weights": len(mapping), "linear_calls": sum(uses.values())}


def convert(input_path: Path, output_path: Path, checkpoint_path: Path, quant: str):
    input_path, output_path = input_path.resolve(), output_path.resolve()
    data_path = output_path.with_name(output_path.name + ".data")
    if input_path == output_path or output_path.exists() or data_path.exists():
        raise ValueError("Choose a new output path; overwriting models/data is not supported")
    model = onnx.load(input_path, load_external_data=False)
    # Avoid overwriting any input external data, even with an unusual filename.
    sources = {(input_path.parent / e.value).resolve() for t in model.graph.initializer
               for e in t.external_data if e.key == "location"}
    if data_path in sources or output_path in sources:
        raise ValueError("Output overlaps input external data")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".seedvr-quant-", dir=output_path.parent) as temporary:
        stage = Path(temporary)
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            with (stage / data_path.name).open("wb") as stream:
                stats = rewrite(model, checkpoint, quant, Writer(stream, data_path.name), input_path)
        onnx.save_model(model, stage / output_path.name)
        onnx.checker.check_model(str(stage / output_path.name))
        shutil.move(stage / data_path.name, data_path)
        shutil.move(stage / output_path.name, output_path)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_MODEL_DIR / "seedvr2_3b_fp16_dit.onnx",
                        help="Unquantized DiT; default: models/SeedVR2-3B-onnx/seedvr2_3b_fp16_dit.onnx")
    parser.add_argument("--output", type=Path, help="Default: seedvr2_3b_<quant>_dit.onnx beside the input; writes <filename>.data alongside it")
    parser.add_argument("--quant", choices=FILES, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Local safetensors; otherwise use the global HF cache")
    parser.add_argument("--revision", default=REVISION, help="Comfy-Org/SeedVR2 revision for download")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input.parent / f"seedvr2_3b_{args.quant}_dit.onnx"
    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = Path(hf_hub_download(REPO, "diffusion_models/" + FILES[args.quant], revision=args.revision))
    stats = convert(args.input, args.output, checkpoint, "fp8" if args.quant == "fp8_wo" else args.quant)
    stats["quant"] = args.quant
    print(json.dumps({**stats, "output": str(args.output), "checkpoint": str(checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
