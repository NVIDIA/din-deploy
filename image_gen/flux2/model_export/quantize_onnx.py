# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Insert FLUX.2 NVFP4 quantizers into an existing unquantized ONNX model.

This is intentionally ONNX-only: it does not instantiate the PyTorch model and
does not modify the export scripts. It uses the quantized checkpoint metadata
to find the FLUX linear layers, maps them onto MatMul module scopes, and inserts
ModelOpt-style activation and weight quantization subgraphs. NVFP4 export
translates the checkpoint's packed weights and blocked scales directly while
streaming external data.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch
from comfy_kitchen.float_utils import from_blocked, swap_nibbles
from modelopt_export_compat import topologically_sort_graph
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open
from huggingface_hub import snapshot_download

EXTERNAL_DATA_NAME = "model.onnx_data"
BLOCK_SIZE = 16

DEFAULT_BASE_MODEL_NAME = "black-forest-labs/FLUX.2-klein-4b"
DEFAULT_FP8_MODEL_NAME = "black-forest-labs/FLUX.2-klein-4b-fp8"
FP8_FILE = "flux-2-klein-4b-fp8.safetensors"
DEFAULT_NVFP4_MODEL_NAME = "black-forest-labs/FLUX.2-klein-4b-nvfp4"
NVFP4_FILE = "flux-2-klein-4b-nvfp4.safetensors"

def resolve_model_snapshot(model_name: str) -> Path:
    local_path = Path(model_name).expanduser()
    if local_path.exists():
        return local_path.resolve()
    return Path(snapshot_download(repo_id=model_name)).resolve()

def map_key(key: str, _shape: tuple[int, ...]) -> tuple[str, ...]:
    suffix = key.rsplit(".", 1)[-1]

    direct_prefixes = {
        "img_in": "x_embedder",
        "txt_in": "context_embedder",
        "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "final_layer.adaLN_modulation.1": "norm_out.linear",
        "final_layer.linear": "proj_out",
    }
    for old, new in direct_prefixes.items():
        if key.startswith(f"{old}."):
            return (f"{new}.{suffix}",)

    parts = key.split(".")
    if len(parts) < 4:
        raise KeyError(f"Unsupported checkpoint key: {key}")

    if parts[0] == "double_blocks":
        block = parts[1]
        stream = parts[2]
        rest = ".".join(parts[3:-1])
        base = f"transformer_blocks.{block}"
        mapping = {
            "img_attn.qkv": (
                f"{base}.attn.to_q",
                f"{base}.attn.to_k",
                f"{base}.attn.to_v",
            ),
            "txt_attn.qkv": (
                f"{base}.attn.add_q_proj",
                f"{base}.attn.add_k_proj",
                f"{base}.attn.add_v_proj",
            ),
            "img_attn.proj": (f"{base}.attn.to_out.0",),
            "txt_attn.proj": (f"{base}.attn.to_add_out",),
            "img_attn.norm.query_norm": (f"{base}.attn.norm_q",),
            "img_attn.norm.key_norm": (f"{base}.attn.norm_k",),
            "txt_attn.norm.query_norm": (f"{base}.attn.norm_added_q",),
            "txt_attn.norm.key_norm": (f"{base}.attn.norm_added_k",),
            "img_mlp.0": (f"{base}.ff.linear_in",),
            "img_mlp.2": (f"{base}.ff.linear_out",),
            "txt_mlp.0": (f"{base}.ff_context.linear_in",),
            "txt_mlp.2": (f"{base}.ff_context.linear_out",),
        }
        mapped = mapping.get(f"{stream}.{rest}")
        if mapped is None:
            raise KeyError(f"Unsupported double-block checkpoint key: {key}")
        return tuple(f"{name}.{'weight' if suffix == 'scale' else suffix}" for name in mapped)

    if parts[0] == "single_blocks":
        block = parts[1]
        rest = ".".join(parts[2:-1])
        base = f"single_transformer_blocks.{block}.attn"
        mapping = {
            "linear1": (f"{base}.to_qkv_mlp_proj",),
            "linear2": (f"{base}.to_out",),
            "norm.query_norm": (f"{base}.norm_q",),
            "norm.key_norm": (f"{base}.norm_k",),
        }
        mapped = mapping.get(rest)
        if mapped is None:
            raise KeyError(f"Unsupported single-block checkpoint key: {key}")
        return tuple(f"{name}.{'weight' if suffix == 'scale' else suffix}" for name in mapped)

    raise KeyError(f"Unsupported checkpoint key: {key}")


@dataclass
class Nvfp4Payload:
    module_name: str
    checkpoint_layer_name: str
    checkpoint_chunk_index: int
    checkpoint_chunk_count: int
    weight_packed_shape_hint: tuple[int, int]
    weight_scale_2: np.ndarray
    input_scale: np.ndarray

    @property
    def weight_shape(self) -> tuple[int, int]:
        packed_shape = self.weight_packed_shape_hint
        return (packed_shape[0], packed_shape[1] * 2)

    @property
    def matmul_weight_shape(self) -> tuple[int, int]:
        out_features, in_features = self.weight_shape
        return (in_features, out_features)


@dataclass
class Fp8Payload:
    module_name: str
    weight_shape_hint: tuple[int, int]
    weight_scale: np.ndarray
    input_scale: np.ndarray

    @property
    def weight_shape(self) -> tuple[int, int]:
        return self.weight_shape_hint

    @property
    def matmul_weight_shape(self) -> tuple[int, int]:
        out_features, in_features = self.weight_shape
        return (in_features, out_features)

def torch_tensor_to_numpy(tensor) -> np.ndarray:
    if str(tensor.dtype).startswith("torch.float8"):
        return tensor.contiguous().view(dtype=__import__("torch").uint8).numpy()
    return tensor.detach().cpu().contiguous().numpy()


def scalar_float(tensor) -> np.ndarray:
    arr = torch_tensor_to_numpy(tensor).astype(np.float32, copy=False).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Expected scalar scale, got shape {tuple(arr.shape)}")
    return arr


def load_nvfp4_payloads(checkpoint_path: Path) -> dict[str, Nvfp4Payload]:
    payloads: dict[str, Nvfp4Payload] = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        quant_metadata = json.loads(f.metadata()["_quantization_metadata"])
        quant_layers = quant_metadata["layers"]

        for layer_name, layer_meta in quant_layers.items():
            if layer_meta.get("format") != "nvfp4":
                raise ValueError(f"Only nvfp4 is supported for now; {layer_name} is {layer_meta}")

            weight_slice = f.get_slice(f"{layer_name}.weight")
            weight_shape = tuple(int(dim) for dim in weight_slice.get_shape())
            weight_scale_2 = f.get_tensor(f"{layer_name}.weight_scale_2")
            input_scale = f.get_tensor(f"{layer_name}.input_scale")
            mapped_weight_keys = map_key(f"{layer_name}.weight", weight_shape)
            module_names = [key.removesuffix(".weight") for key in mapped_weight_keys]
            if weight_shape[0] % len(module_names) != 0:
                raise ValueError(f"Cannot split {layer_name}.weight shape {weight_shape} across {len(module_names)} modules")
            packed_shape = (weight_shape[0] // len(module_names), *weight_shape[1:])

            for chunk_index, module_name in enumerate(module_names):
                payloads[module_name] = Nvfp4Payload(
                    module_name=module_name,
                    checkpoint_layer_name=layer_name,
                    checkpoint_chunk_index=chunk_index,
                    checkpoint_chunk_count=len(module_names),
                    weight_packed_shape_hint=packed_shape,
                    weight_scale_2=scalar_float(weight_scale_2),
                    input_scale=scalar_float(input_scale),
                )

    return payloads

def load_fp8_payloads(checkpoint_path: Path) -> dict[str, Fp8Payload]:
    payloads: dict[str, Fp8Payload] = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        quant_metadata = json.loads(f.metadata()["_quantization_metadata"])
        quant_layers = quant_metadata["layers"]

        for layer_name, layer_meta in quant_layers.items():
            if layer_meta.get("format") not in {"fp8", "float8_e4m3fn"}:
                raise ValueError(f"Only fp8 is supported in this path; {layer_name} is {layer_meta}")

            weight_slice = f.get_slice(f"{layer_name}.weight")
            weight_shape = tuple(int(dim) for dim in weight_slice.get_shape())
            weight_scale = f.get_tensor(f"{layer_name}.weight_scale")
            input_scale = f.get_tensor(f"{layer_name}.input_scale")
            mapped_weight_keys = map_key(f"{layer_name}.weight", weight_shape)
            module_names = [key.removesuffix(".weight") for key in mapped_weight_keys]
            if weight_shape[0] % len(module_names) != 0:
                raise ValueError(f"Cannot split {layer_name}.weight shape {weight_shape} across {len(module_names)} modules")
            chunk_shape = (weight_shape[0] // len(module_names), *weight_shape[1:])

            for module_name in module_names:
                payloads[module_name] = Fp8Payload(
                    module_name=module_name,
                    weight_shape_hint=chunk_shape,
                    weight_scale=scalar_float(weight_scale),
                    input_scale=scalar_float(input_scale),
                )

    return payloads
def expected_flux2_klein_modules() -> list[str]:
    modules: list[str] = []
    for block in range(5):
        base = f"transformer_blocks.{block}"
        modules.extend(
            [
                f"{base}.attn.to_q",
                f"{base}.attn.to_k",
                f"{base}.attn.to_v",
                f"{base}.attn.add_q_proj",
                f"{base}.attn.add_k_proj",
                f"{base}.attn.add_v_proj",
                f"{base}.attn.to_out.0",
                f"{base}.attn.to_add_out",
                f"{base}.ff.linear_in",
                f"{base}.ff.linear_out",
                f"{base}.ff_context.linear_in",
                f"{base}.ff_context.linear_out",
            ]
        )
    for block in range(20):
        base = f"single_transformer_blocks.{block}.attn"
        modules.extend([f"{base}.to_qkv_mlp_proj", f"{base}.to_out"])
    return modules


def ensure_trt_opset(model: onnx.ModelProto) -> None:
    for opset in model.opset_import:
        if opset.domain == "trt":
            opset.version = max(opset.version, 1)
            break
    else:
        model.opset_import.append(helper.make_opsetid("trt", 1))


def ensure_min_onnx_opset(model: onnx.ModelProto, version: int) -> None:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            if opset.version < version:
                opset.version = version
            return
    model.opset_import.append(helper.make_opsetid("", version))


def matmul_weight_nodes(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    initializers = {initializer.name for initializer in model.graph.initializer}
    return [
        node
        for node in model.graph.node
        if node.op_type == "MatMul" and len(node.input) >= 2 and node.input[1] in initializers
    ]


def match_modules_to_matmuls(
    model: onnx.ModelProto,
    payloads: dict[str, Nvfp4Payload | Fp8Payload],
    modules: list[str],
) -> dict[str, onnx.NodeProto]:
    inits = {initializer.name: initializer for initializer in model.graph.initializer}
    candidates = matmul_weight_nodes(model)

    scoped_matches: dict[str, onnx.NodeProto] = {}
    for node in candidates:
        metadata = {entry.key: entry.value for entry in node.metadata_props}
        raw_scopes = metadata.get("pkg.torch.onnx.name_scopes")
        if raw_scopes is None:
            continue
        try:
            scopes = ast.literal_eval(raw_scopes)
        except (SyntaxError, ValueError) as error:
            raise RuntimeError(
                f"Invalid name-scope metadata on {node.name or node.output[0]}: "
                f"{raw_scopes!r}"
            ) from error
        matched_modules = [
            scope.removeprefix("transformer.")
            for scope in scopes
            if scope.startswith("transformer.")
            and scope.removeprefix("transformer.") in payloads
        ]
        if not matched_modules:
            continue
        module_name = matched_modules[-1]
        if module_name in scoped_matches:
            raise RuntimeError(f"Multiple MatMuls carry scope {module_name}")
        expected_shape = payloads[module_name].matmul_weight_shape
        actual_shape = tuple(int(dim) for dim in inits[node.input[1]].dims)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Scoped MatMul shape mismatch for {module_name}: "
                f"ONNX={actual_shape}, checkpoint={expected_shape}"
            )
        scoped_matches[module_name] = node

    missing_scopes = [module for module in modules if module not in scoped_matches]
    if missing_scopes:
        raise RuntimeError(
            "ONNX module-scope metadata matched only "
            f"{len(scoped_matches)}/{len(modules)} quantized MatMuls; "
            f"missing {missing_scopes[:10]}"
        )
    return scoped_matches


def add_value_info(model: onnx.ModelProto, name: str, dtype: int, shape: list[int | str] | None = None) -> None:
    known = {vi.name for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}
    if name in known:
        return
    model.graph.value_info.append(helper.make_tensor_value_info(name, dtype, shape))


def make_scalar_initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value.astype(np.float32, copy=False).reshape(()), name)


def make_raw_initializer(name: str, dtype: int, dims: tuple[int, ...], raw: bytes) -> onnx.TensorProto:
    return helper.make_tensor(name=name, data_type=dtype, dims=list(dims), vals=raw, raw=True)


def external_data_fields(initializer: onnx.TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in initializer.external_data}


def set_external_data_reference(
    initializer: onnx.TensorProto,
    *,
    offset: int,
    length: int,
) -> None:
    initializer.ClearField("raw_data")
    del initializer.external_data[:]
    initializer.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", EXTERNAL_DATA_NAME),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = initializer.external_data.add()
        entry.key = key
        entry.value = value


class ExternalDataWriter:
    """Write ONNX tensor payloads sequentially without materializing the model."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None

    def __enter__(self) -> ExternalDataWriter:
        self._file = self.path.open("wb")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def offset(self) -> int:
        if self._file is None:
            raise RuntimeError("ExternalDataWriter is not open")
        return self._file.tell()

    def _write_buffer(self, buffer) -> tuple[int, int]:
        if self._file is None:
            raise RuntimeError("ExternalDataWriter is not open")
        offset = self._file.tell()
        view = memoryview(buffer)
        length = self._file.write(view)
        if length != view.nbytes:
            raise OSError(
                f"Short external-data write: expected {view.nbytes}, wrote {length}"
            )
        return offset, length

    def write_tensor(
        self,
        name: str,
        dtype: int,
        dims: tuple[int, ...],
        tensor: torch.Tensor,
    ) -> onnx.TensorProto:
        byte_view = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        offset, length = self._write_buffer(memoryview(byte_view))
        initializer = onnx.TensorProto()
        initializer.name = name
        initializer.data_type = dtype
        initializer.dims.extend(dims)
        set_external_data_reference(initializer, offset=offset, length=length)
        return initializer

    def copy_initializer(self, initializer: onnx.TensorProto, input_model_path: Path) -> None:
        metadata = external_data_fields(initializer)
        if not metadata:
            return
        if "location" not in metadata or "length" not in metadata:
            raise RuntimeError(
                f"External initializer {initializer.name} has incomplete metadata: {metadata}"
            )
        source_path = input_model_path.parent / metadata["location"]
        source_offset = int(metadata.get("offset", 0))
        remaining = int(metadata["length"])
        destination_offset = self.offset
        with source_path.open("rb") as source:
            source.seek(source_offset)
            while remaining:
                chunk = source.read(min(remaining, 16 * 1024 * 1024))
                if not chunk:
                    raise EOFError(
                        f"Unexpected end of {source_path} while copying {initializer.name}"
                    )
                self._write_buffer(chunk)
                remaining -= len(chunk)
        set_external_data_reference(
            initializer,
            offset=destination_offset,
            length=int(metadata["length"]),
        )


def make_scalar_dtype_initializer(name: str, value: np.ndarray, dtype: int) -> onnx.TensorProto:
    scalar = float(value.astype(np.float32, copy=False).reshape(-1)[0])
    if dtype == TensorProto.FLOAT:
        return numpy_helper.from_array(np.array(scalar, dtype=np.float32), name)
    if dtype == TensorProto.FLOAT16:
        return numpy_helper.from_array(np.array(scalar, dtype=np.float16), name)
    if dtype == TensorProto.BFLOAT16:
        raw = (np.array([scalar], dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
        return helper.make_tensor(name=name, data_type=TensorProto.BFLOAT16, dims=[], vals=raw, raw=True)
    raise ValueError(f"Unsupported scalar initializer dtype: {dtype}")


def initializer_to_float32(initializer: onnx.TensorProto) -> np.ndarray:
    return np.asarray(numpy_helper.to_array(initializer), dtype=np.float32)


def cast_float32_to_fp8e4m3_raw(array: np.ndarray) -> bytes:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32)).clamp(min=-448.0, max=448.0)
    return tensor.to(torch.float8_e4m3fn).view(torch.uint8).contiguous().numpy().astype(np.uint8, copy=False).tobytes()


def fold_fp8_dynamic_quantize(weight: np.ndarray, scale: np.ndarray) -> bytes:
    scalar = float(scale.astype(np.float32, copy=False).reshape(-1)[0])
    if scalar == 0.0:
        scalar = 1.0
    scaled = np.asarray(weight, dtype=np.float32) / np.float32(scalar)
    return cast_float32_to_fp8e4m3_raw(scaled)

def insert_nvfp4_for_matmul(
    model: onnx.ModelProto,
    node: onnx.NodeProto,
    payload: Nvfp4Payload,
    compute_dtype: int,
    checkpoint_weight: torch.Tensor,
    checkpoint_scale: torch.Tensor,
    external_writer: ExternalDataWriter,
) -> list[onnx.NodeProto]:
    prefix = f"transformer.{payload.module_name}"
    original_x = node.input[0]

    input_scale_name = f"{prefix}.input_quantizer.scale"
    x_f4 = f"{prefix}.input_quantizer.TRT_FP4DynamicQuantize_output_0"
    x_sx_f8 = f"{prefix}.input_quantizer.TRT_FP4DynamicQuantize_output_1"
    x_sx_f32 = f"{prefix}.input_quantizer.scale_dq"
    x_dq = f"{prefix}.input_quantizer.dq"
    x_cast = f"{prefix}.input_quantizer.cast_to_compute"

    weight_dq = f"{prefix}.weight_quantizer.dq"
    weight_cast = f"{prefix}.weight_quantizer.cast_to_compute"

    weight_f4_name = f"{prefix}.weight_quantizer.weight_f4"
    weight_f8_scale_name = f"{prefix}.weight_quantizer.weight_f8_scale"
    weight_f32_scale_name = f"{prefix}.weight_quantizer.weight_f32_scale"
    weight_scale_dq = f"{prefix}.weight_quantizer.weight_scale_dq"
    weight_transpose = f"{prefix}.weight_quantizer.transpose"

    out_features, in_features = payload.weight_shape
    matmul_weight_shape = [in_features, out_features]
    linear_weight_shape = [out_features, in_features]

    initializers = [
        make_scalar_initializer(input_scale_name, payload.input_scale),
        make_scalar_initializer(weight_f32_scale_name, payload.weight_scale_2),
    ]
    nodes = [
        helper.make_node(
            "TRT_FP4DynamicQuantize",
            inputs=[original_x, input_scale_name],
            outputs=[x_f4, x_sx_f8],
            name=f"{prefix}.input_quantizer.TRT_FP4DynamicQuantize",
            domain="trt",
            axis=-1,
            block_size=BLOCK_SIZE,
            scale_type=TensorProto.FLOAT8E4M3FN,
        ),
        helper.make_node(
            "DequantizeLinear",
            inputs=[x_sx_f8, input_scale_name],
            outputs=[x_sx_f32],
            name=f"{prefix}.input_quantizer.scale_DequantizeLinear",
        ),
        helper.make_node(
            "DequantizeLinear",
            inputs=[x_f4, x_sx_f32],
            outputs=[x_dq],
            name=f"{prefix}.input_quantizer.DequantizeLinear",
            axis=-1,
            block_size=BLOCK_SIZE,
        ),
        helper.make_node(
            "Cast",
            inputs=[x_dq],
            outputs=[x_cast],
            name=f"{prefix}.input_quantizer.CastToCompute",
            to=compute_dtype,
        ),
    ]

    if tuple(checkpoint_weight.shape) != tuple(payload.weight_packed_shape_hint):
        raise ValueError(
            f"Packed checkpoint shape mismatch for {payload.module_name}: "
            f"{tuple(checkpoint_weight.shape)} != {payload.weight_packed_shape_hint}"
        )
    weight_scale_shape = (out_features, in_features // BLOCK_SIZE)
    if tuple(checkpoint_scale.shape) != weight_scale_shape:
        raise ValueError(
            f"Checkpoint scale shape mismatch for {payload.module_name}: "
            f"{tuple(checkpoint_scale.shape)} != {weight_scale_shape}"
        )

    initializers.extend(
        [
            external_writer.write_tensor(
                weight_f4_name,
                TensorProto.FLOAT4E2M1,
                tuple(linear_weight_shape),
                swap_nibbles(checkpoint_weight),
            ),
            external_writer.write_tensor(
                weight_f8_scale_name,
                TensorProto.FLOAT8E4M3FN,
                weight_scale_shape,
                checkpoint_scale,
            ),
        ]
    )
    nodes.extend(
        [
            helper.make_node(
                "DequantizeLinear",
                inputs=[weight_f8_scale_name, weight_f32_scale_name],
                outputs=[weight_scale_dq],
                name=f"{prefix}.weight_quantizer.scale_DequantizeLinear",
            ),
            helper.make_node(
                "DequantizeLinear",
                inputs=[weight_f4_name, weight_scale_dq],
                outputs=[weight_dq],
                name=f"{prefix}.weight_quantizer.DequantizeLinear",
                axis=-1,
                block_size=BLOCK_SIZE,
            ),
        ]
    )
    add_value_info(model, weight_scale_dq, TensorProto.FLOAT, list(weight_scale_shape))

    nodes.append(
        helper.make_node(
            "Cast",
            inputs=[weight_dq],
            outputs=[weight_cast],
            name=f"{prefix}.weight_quantizer.CastToCompute",
            to=compute_dtype,
        )
    )
    nodes.append(
        helper.make_node(
            "Transpose",
            inputs=[weight_cast],
            outputs=[weight_transpose],
            name=f"{prefix}.weight_quantizer.Transpose",
            perm=[1, 0],
        )
    )

    model.graph.initializer.extend(initializers)
    add_value_info(model, x_f4, TensorProto.FLOAT4E2M1, None)
    add_value_info(model, x_sx_f8, TensorProto.FLOAT8E4M3FN, None)
    add_value_info(model, x_sx_f32, TensorProto.FLOAT, None)
    add_value_info(model, x_dq, TensorProto.FLOAT, None)
    add_value_info(model, x_cast, compute_dtype, None)
    add_value_info(model, weight_dq, TensorProto.FLOAT, linear_weight_shape)
    add_value_info(model, weight_cast, compute_dtype, linear_weight_shape)
    add_value_info(model, weight_transpose, compute_dtype, matmul_weight_shape)

    node.input[0] = x_cast
    node.input[1] = weight_transpose
    return nodes


def insert_fp8_for_matmul(
    model: onnx.ModelProto,
    node: onnx.NodeProto,
    payload: Fp8Payload,
    compute_dtype: int,
    fold_weight_quantizer: bool = False,
    original_weight_initializer: onnx.TensorProto | None = None,
) -> list[onnx.NodeProto]:
    prefix = f"transformer.{payload.module_name}"
    original_x = node.input[0]
    original_weight = node.input[1]

    input_scale_name = f"{prefix}.input_quantizer.scale"
    x_q = f"{prefix}.input_quantizer.TRT_FP8QuantizeLinear_output_0"
    x_dq = f"{prefix}.input_quantizer.TRT_FP8DequantizeLinear_output_0"
    x_cast = f"{prefix}.input_quantizer.cast_to_compute"

    weight_scale_name = f"{prefix}.weight_quantizer.scale"
    weight_q = f"{prefix}.weight_quantizer.TRT_FP8QuantizeLinear_output_0"
    weight_dq = f"{prefix}.weight_quantizer.TRT_FP8DequantizeLinear_output_0"
    weight_cast = f"{prefix}.weight_quantizer.cast_to_compute"
    weight_fp8_name = f"{prefix}.weight_quantizer.weight_fp8"

    out_features, in_features = payload.weight_shape
    matmul_weight_shape = [in_features, out_features]
    initializers = [
        make_scalar_dtype_initializer(input_scale_name, payload.input_scale, compute_dtype),
        make_scalar_dtype_initializer(weight_scale_name, payload.weight_scale, compute_dtype),
    ]
    nodes = [
        helper.make_node(
            "TRT_FP8QuantizeLinear",
            inputs=[original_x, input_scale_name],
            outputs=[x_q],
            name=f"{prefix}.input_quantizer.TRT_FP8QuantizeLinear",
            domain="trt",
        ),
        helper.make_node(
            "TRT_FP8DequantizeLinear",
            inputs=[x_q, input_scale_name],
            outputs=[x_dq],
            name=f"{prefix}.input_quantizer.TRT_FP8DequantizeLinear",
            domain="trt",
        ),
        helper.make_node(
            "Cast",
            inputs=[x_dq],
            outputs=[x_cast],
            name=f"{prefix}.input_quantizer.CastToCompute",
            to=compute_dtype,
        ),
    ]

    if fold_weight_quantizer:
        if original_weight_initializer is None:
            raise ValueError(f"Cannot fold FP8 weight quantizer for {payload.module_name}: missing original initializer")
        weight = initializer_to_float32(original_weight_initializer)
        if tuple(int(dim) for dim in weight.shape) != tuple(matmul_weight_shape):
            raise ValueError(
                f"Cannot fold {payload.module_name}: ONNX weight shape {weight.shape} does not match expected {matmul_weight_shape}"
            )
        initializers.append(
            make_raw_initializer(
                weight_fp8_name,
                TensorProto.FLOAT8E4M3FN,
                tuple(matmul_weight_shape),
                fold_fp8_dynamic_quantize(weight, payload.weight_scale),
            )
        )
        nodes.append(
            helper.make_node(
                "TRT_FP8DequantizeLinear",
                inputs=[weight_fp8_name, weight_scale_name],
                outputs=[weight_dq],
                name=f"{prefix}.weight_quantizer.TRT_FP8DequantizeLinear",
                domain="trt",
            )
        )
    else:
        nodes.extend(
            [
                helper.make_node(
                    "TRT_FP8QuantizeLinear",
                    inputs=[original_weight, weight_scale_name],
                    outputs=[weight_q],
                    name=f"{prefix}.weight_quantizer.TRT_FP8QuantizeLinear",
                    domain="trt",
                ),
                helper.make_node(
                    "TRT_FP8DequantizeLinear",
                    inputs=[weight_q, weight_scale_name],
                    outputs=[weight_dq],
                    name=f"{prefix}.weight_quantizer.TRT_FP8DequantizeLinear",
                    domain="trt",
                ),
            ]
        )
        add_value_info(model, weight_q, TensorProto.FLOAT8E4M3FN, matmul_weight_shape)

    nodes.append(
        helper.make_node(
            "Cast",
            inputs=[weight_dq],
            outputs=[weight_cast],
            name=f"{prefix}.weight_quantizer.CastToCompute",
            to=compute_dtype,
        )
    )

    model.graph.initializer.extend(initializers)
    add_value_info(model, x_q, TensorProto.FLOAT8E4M3FN, None)
    add_value_info(model, x_dq, TensorProto.FLOAT16, None)
    add_value_info(model, x_cast, compute_dtype, None)
    add_value_info(model, weight_dq, TensorProto.FLOAT16, matmul_weight_shape)
    add_value_info(model, weight_cast, compute_dtype, matmul_weight_shape)

    node.input[0] = x_cast
    node.input[1] = weight_cast
    return nodes

def remove_initializers(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [initializer for initializer in model.graph.initializer if initializer.name not in names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def quantize_nvfp4(
    model: onnx.ModelProto,
    payloads: dict[str, Nvfp4Payload],
    compute_dtype: int,
    checkpoint_path: Path,
    external_writer: ExternalDataWriter,
) -> tuple[int, int]:
    modules = expected_flux2_klein_modules()
    missing = [module for module in modules if module not in payloads]
    if missing:
        raise RuntimeError(f"Checkpoint is missing {len(missing)} expected NVFP4 modules: {missing[:10]}")

    matches = match_modules_to_matmuls(model, payloads, modules)
    new_nodes_by_matmul: dict[str, list[onnx.NodeProto]] = {}
    replaced_weights: set[str] = set()

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        cached_layer_name = None
        packed_chunks = ()
        scale_chunks = ()
        for module_name in modules:
            payload = payloads[module_name]
            if payload.checkpoint_layer_name != cached_layer_name:
                cached_layer_name = payload.checkpoint_layer_name
                packed = checkpoint.get_tensor(f"{cached_layer_name}.weight")
                blocked_scale = checkpoint.get_tensor(
                    f"{cached_layer_name}.weight_scale"
                )
                logical_scale = from_blocked(
                    blocked_scale,
                    num_rows=packed.shape[0],
                    num_cols=packed.shape[-1] * 2 // BLOCK_SIZE,
                ).contiguous()
                packed_chunks = packed.chunk(payload.checkpoint_chunk_count, dim=0)
                scale_chunks = logical_scale.chunk(
                    payload.checkpoint_chunk_count,
                    dim=0,
                )

            matmul = matches[module_name]
            original_weight = matmul.input[1]
            replaced_weights.add(original_weight)
            new_nodes_by_matmul[matmul.name or matmul.output[0]] = (
                insert_nvfp4_for_matmul(
                    model,
                    matmul,
                    payload,
                    compute_dtype,
                    checkpoint_weight=packed_chunks[payload.checkpoint_chunk_index],
                    checkpoint_scale=scale_chunks[payload.checkpoint_chunk_index],
                    external_writer=external_writer,
                )
            )

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        key = node.name or node.output[0]
        if key in new_nodes_by_matmul:
            rewritten_nodes.extend(new_nodes_by_matmul[key])
        rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    remove_initializers(model, replaced_weights)

    ensure_trt_opset(model)
    ensure_min_onnx_opset(model, 23)
    topologically_sort_graph(model)
    return len(matches), len(replaced_weights)


def quantize_fp8(
    model: onnx.ModelProto,
    payloads: dict[str, Fp8Payload],
    compute_dtype: int,
    fold_weight_quantizers: bool = False,
) -> tuple[int, int]:
    modules = expected_flux2_klein_modules()
    missing = [module for module in modules if module not in payloads]
    if missing:
        raise RuntimeError(f"Checkpoint is missing {len(missing)} expected FP8 modules: {missing[:10]}")

    matches = match_modules_to_matmuls(model, payloads, modules)
    initializer_map = {initializer.name: initializer for initializer in model.graph.initializer}
    new_nodes_by_matmul: dict[str, list[onnx.NodeProto]] = {}
    replaced_weights: set[str] = set()
    for module_name in modules:
        matmul = matches[module_name]
        original_weight = matmul.input[1]
        replaced_weights.add(original_weight)
        new_nodes_by_matmul[matmul.name or matmul.output[0]] = insert_fp8_for_matmul(
            model,
            matmul,
            payloads[module_name],
            compute_dtype,
            fold_weight_quantizer=fold_weight_quantizers,
            original_weight_initializer=initializer_map.get(original_weight),
        )

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        key = node.name or node.output[0]
        if key in new_nodes_by_matmul:
            rewritten_nodes.extend(new_nodes_by_matmul[key])
        rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    if fold_weight_quantizers:
        remove_initializers(model, replaced_weights)

    ensure_trt_opset(model)
    ensure_min_onnx_opset(model, 25)
    return len(matches), len(replaced_weights) if fold_weight_quantizers else 0

def resolve_model_path(path: Path) -> Path:
    return path / "model.onnx" if path.is_dir() else path


def resolve_output_model_path(path: Path) -> Path:
    if path.is_dir() or path.suffix.lower() != ".onnx":
        return path / "model.onnx"
    return path


def prepare_output(input_path: Path, output_path: Path) -> Path:
    output_model_path = resolve_output_model_path(output_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.resolve() != output_model_path.resolve():
        shutil.copy2(input_path, output_model_path)
        input_data = input_path.parent / EXTERNAL_DATA_NAME
        if input_data.exists():
            shutil.copy2(input_data, output_model_path.parent / EXTERNAL_DATA_NAME)
    return output_model_path


def save_external_model(model: onnx.ModelProto, onnx_path: Path) -> None:
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=EXTERNAL_DATA_NAME,
        size_threshold=1024,
        convert_attribute=False,
    )


def save_streaming_nvfp4_model(
    model: onnx.ModelProto,
    input_model_path: Path,
    output_model_path: Path,
    checkpoint_path: Path,
    payloads: dict[str, Nvfp4Payload],
    compute_dtype: int,
) -> tuple[int, int, int]:
    """Build a static NVFP4 model without loading the baseline weight data."""
    if input_model_path.resolve() == output_model_path.resolve():
        raise ValueError(
            "The streaming NVFP4 path requires --output to differ from --input"
        )

    modules = expected_flux2_klein_modules()
    matches = match_modules_to_matmuls(model, payloads, modules)
    replaced_weights = {matches[module].input[1] for module in modules}

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    copied_initializers = 0

    with tempfile.TemporaryDirectory(
        prefix=".quantize_onnx.",
        dir=output_model_path.parent,
    ) as staging_dir_name:
        staging_dir = Path(staging_dir_name)
        model_temp_path = staging_dir / output_model_path.name
        data_temp_path = staging_dir / EXTERNAL_DATA_NAME
        with ExternalDataWriter(data_temp_path) as writer:
            for initializer in model.graph.initializer:
                if initializer.name in replaced_weights:
                    continue
                if initializer.external_data:
                    writer.copy_initializer(initializer, input_model_path)
                    copied_initializers += 1

            inserted, removed = quantize_nvfp4(
                model,
                payloads,
                compute_dtype=compute_dtype,
                checkpoint_path=checkpoint_path,
                external_writer=writer,
            )

        onnx.save_model(model, str(model_temp_path), convert_attribute=False)
        onnx.checker.check_model(str(model_temp_path))

        output_data_path = output_model_path.parent / EXTERNAL_DATA_NAME
        data_temp_path.replace(output_data_path)
        model_temp_path.replace(output_model_path)
        return inserted, removed, copied_initializers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Unquantized ONNX model file or directory containing model.onnx")
    parser.add_argument("--model_name", default=None, help="Base Hugging Face model repo ID used for transformer config")
    parser.add_argument("--output", required=True, type=Path, help="Output ONNX model file or directory")
    parser.add_argument("--quant", choices=["nvfp4", "fp8"], default="nvfp4")
    parser.add_argument("--compute-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--fold-weight-quantizers",
        action="store_true",
        help="FP8 only: fold weight quantizers into static initializers",
    )
    args = parser.parse_args()
    dtype_map = {
        "bf16": TensorProto.BFLOAT16,
        "fp16": TensorProto.FLOAT16,
        "fp32": TensorProto.FLOAT,
    }
    input_model_path = resolve_model_path(args.input)
    if not input_model_path.exists():
        raise FileNotFoundError(input_model_path)
    model_name = args.model_name
    if args.model_name is None:
        if args.quant == "nvfp4":
            model_name = DEFAULT_NVFP4_MODEL_NAME
        elif args.quant == "fp8":
            model_name = DEFAULT_FP8_MODEL_NAME
        else:
            raise ValueError(f"Unknown quantization mode: {args.quant}")

    checkpoint = resolve_model_snapshot(model_name)
    if args.quant == "nvfp4":
        checkpoint /= NVFP4_FILE
        payloads = load_nvfp4_payloads(checkpoint)
        model = onnx.load(str(input_model_path), load_external_data=False)
        output_model_path = resolve_output_model_path(args.output)
        inserted, removed, copied = save_streaming_nvfp4_model(
            model,
            input_model_path,
            output_model_path,
            checkpoint,
            payloads,
            compute_dtype=dtype_map[args.compute_dtype],
        )
        print(f"Inserted NVFP4 quantizer pairs: {inserted}")
        print(f"Removed original weight initializers: {removed}")
        print(f"Stream-copied retained external initializers: {copied}")
        print(f"Saved: {output_model_path}")
        return

    checkpoint /= FP8_FILE
    payloads = load_fp8_payloads(checkpoint)
    model = onnx.load(str(input_model_path), load_external_data=True)
    output_model_path = prepare_output(input_model_path, args.output)
    if output_model_path != input_model_path:
        model = onnx.load(str(output_model_path), load_external_data=True)
    inserted, removed = quantize_fp8(
        model,
        payloads,
        compute_dtype=dtype_map[args.compute_dtype],
        fold_weight_quantizers=args.fold_weight_quantizers,
    )
    save_external_model(model, output_model_path)
    print(f"Inserted FP8 quantizer pairs: {inserted}")
    print(f"Removed original weight initializers: {removed}")
    print(f"Saved: {output_model_path}")


if __name__ == "__main__":
    main()
