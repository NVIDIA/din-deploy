# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ORT tensor adapters and comparison helpers for the two SAM verifiers."""

from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path

import numpy as np


def add_runtime_arguments(parser):
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("trt-rtx", "cpu"), default="trt-rtx")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--onnx-components",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Run named components with ONNX Runtime (default: all). Unselected components use PyTorch; use 'none' for an all-PyTorch candidate.",
    )
    parser.add_argument("--ep-lib", type=Path)
    parser.add_argument("--ep-dll-dir", type=Path)
    parser.add_argument("--trt-bin", type=Path)
    parser.add_argument("--ep-cache", type=Path)
    parser.add_argument("--validate-only", action="store_true", help="Check ONNX structure without inference")
    parser.add_argument("--graphs", nargs="+", help="Select graphs for --validate-only")
    parser.add_argument("--fixtures", action="store_true",
                        help="Opt in to per-graph numerical checks of saved export NPZ fixtures; requires --validate-only (off by default)")
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--rtol", type=float, default=0.002)
    parser.add_argument("--min-iou", type=float, default=0.95)
    parser.add_argument(
        "--din-sam", type=Path, help="Also run din_sam_cli and compare its masks with the component pipeline"
    )


def validate_arguments(parser, args):
    args.onnx_dir = args.onnx_dir.resolve()
    args.output = args.output.resolve()
    if args.output.is_relative_to(args.onnx_dir):
        parser.error("--output must be outside the model bundle")
    if args.ep_cache and args.ep_cache.resolve().is_relative_to(args.onnx_dir):
        parser.error("--ep-cache must be outside the model bundle")
    if (
            not 0 <= args.min_iou <= 1
            or args.atol < 0
            or args.rtol < 0
            or not math.isfinite(args.atol)
            or not math.isfinite(args.rtol)
    ):
        parser.error("IoU must be in [0, 1] and numerical tolerances must be nonnegative")
    if args.graphs and not args.validate_only:
        parser.error("--graphs is only supported with --validate-only; inference needs the complete pipeline")
    if args.fixtures and not args.validate_only:
        parser.error("--fixtures is only supported with --validate-only")
    if args.validate_only and args.onnx_components is not None:
        parser.error("--onnx-components selects pipeline execution and cannot be used with --validate-only")
    if args.onnx_components is None:
        args.onnx_components = [] if args.validate_only else ["all"]
    elif "none" in args.onnx_components:
        if args.onnx_components != ["none"]:
            parser.error("Use --onnx-components none by itself")
        args.onnx_components = []
    if (args.onnx_components or args.fixtures) and args.provider == "trt-rtx" and args.device != "cuda":
        parser.error("--provider trt-rtx requires --device cuda")


def select_components(requested, available, graphs):
    """Select only components used by this invocation; unselected graphs need not exist."""
    selected = set(requested)
    if "all" in selected:
        if len(selected) != 1:
            raise ValueError("Use --onnx-components all by itself")
        selected = set(available)
    if selected - set(available):
        raise ValueError(
            f"Unknown or unused ONNX components: {sorted(selected - set(available))}; available: {sorted(available)}"
        )
    if selected - set(graphs):
        raise ValueError(f"Requested ONNX components are missing from the bundle: {sorted(selected - set(graphs))}")
    return selected


class ModelComponent:
    """One exporter wrapper with an interchangeable export (PyTorch) or ONNX forward.

    Both backends return the same named tensors. Fixed-capacity ONNX attention
    gets padding/validity here; the Python pipeline and PyTorch wrapper keep
    natural, unpadded histories. Only selected components create ORT sessions.
    """

    def __init__(self, bundle, name, module, output_names, input_names=None):
        self.bundle, self.name, self.module = bundle, name, module.eval()
        self.input_names = tuple(input_names or inspect.signature(module.forward).parameters)
        self.output_names = tuple(output_names)
        self.backend = "export"

    def export(self, values):
        import torch

        outputs = self.module(*(values[name] for name in self.input_names))
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        result = dict(zip(self.output_names, outputs, strict=True))
        if not all(bool(torch.isfinite(value).all()) for value in result.values()):
            raise ValueError(f"Non-finite PyTorch output from {self.name}")
        return result

    def onnx_inputs(self, values):
        import torch

        if self.name != "memory_attention":
            return values
        inputs = {item.name: item for item in self.bundle.session(self.name).get_inputs()}
        result = dict(values)
        groups = [
            (
                "memory_valid",
                ("memory", "memory_pos")
                if self.bundle.family == "sam2"
                else ("memory_image", "memory_image_position", "memory", "memory_position"),
                4096 if self.bundle.family == "sam2" else 5184,
            ),
            ("pointer_valid", ("object_pointers", "pointer_position"), 16),
        ]
        for validity, names, tokens in groups:
            if validity not in inputs:
                continue
            count = values[names[0]].shape[0]
            capacity = inputs[names[0]].shape[0]
            if not isinstance(capacity, int) or count % tokens or not 0 < count <= capacity:
                raise ValueError(f"Invalid {self.name}/{names[0]} occupancy: {count}, capacity: {capacity}")
            tensor = values[names[0]]
            result[validity] = torch.arange(capacity // tokens, device=tensor.device) < count // tokens
            for name in names:
                value = values[name]
                if value.shape[0] != count:
                    raise ValueError(f"Inconsistent history lengths for {name}")
                result[name] = torch.cat([value, value.new_zeros((capacity - count, *value.shape[1:]))])
        return result

    def __call__(self, values):
        self.bundle.active_graph = self.name
        if self.backend == "export":
            result = self.export(values)
        elif self.backend == "onnx":
            result = self.bundle.run(self.name, self.onnx_inputs(values))
            if set(result) != set(self.output_names):
                raise ValueError(f"ONNX output names differ from the export component: {self.name}")
            self.bundle.compare(self.name, result, self.export(values))
        else:
            raise ValueError(f"Unknown component backend: {self.backend}")
        calls = self.bundle.component_calls.setdefault(self.name, {"export": 0, "onnx": 0})
        calls[self.backend] += 1
        return result


def set_component_backends(bundle, components, selected):
    if set(selected) - set(components):
        raise ValueError(f"Unknown components: {sorted(set(selected) - set(components))}")
    for name, component in components.items():
        component.backend = "onnx" if name in selected else "export"
    bundle.component_backends = {name: component.backend for name, component in components.items()}
    bundle.component_calls = {}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tensor_stats(actual, expected, atol, rtol):
    actual, expected = np.asarray(actual), np.asarray(expected)
    same_shape = actual.shape == expected.shape
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    result = {"same_shape": same_shape, "finite": finite, "passed": False}
    if same_shape and finite:
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        result.update(
            passed=bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
            max_absolute_error=float(difference.max()) if difference.size else 0.0,
            relative_l2=float(
                np.linalg.norm(difference.ravel()) / max(np.linalg.norm(expected.astype(np.float64).ravel()), 1e-30)
            ),
        )
    return result


def mask_iou(reference, candidate):
    reference, candidate = np.asarray(reference, dtype=bool), np.asarray(candidate, dtype=bool)
    if reference.shape != candidate.shape:
        raise ValueError(f"Mask dimensions differ: {reference.shape} versus {candidate.shape}")
    union = np.count_nonzero(reference | candidate)
    return float(np.count_nonzero(reference & candidate) / union) if union else 1.0


def compare_instances(reference, candidate, min_iou):
    """Match masks one-to-one; IDs from independent executions need not agree."""
    from scipy.optimize import linear_sum_assignment

    matrix = np.zeros((len(reference), len(candidate)), dtype=np.float64)
    for i, first in enumerate(reference):
        for j, second in enumerate(candidate):
            matrix[i, j] = mask_iou(first, second)
    rows, columns = linear_sum_assignment(-matrix)
    pairs = [
        {"reference_index": int(i), "candidate_index": int(j), "iou": float(matrix[i, j])}
        for i, j in zip(rows, columns, strict=True)
    ]
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "matches": pairs,
        "passed": len(reference) == len(candidate) and all(row["iou"] >= min_iou for row in pairs),
    }


def check_instance_ids(row, reference_ids, candidate_ids, mapping, min_iou):
    """Keep one ID mapping across a sequence, so per-frame matching cannot hide ID switches."""
    row["identity_consistent"] = True
    for pair in row["matches"]:
        first = int(reference_ids[pair["reference_index"]])
        second = int(candidate_ids[pair["candidate_index"]])
        pair.update(reference_id=first, candidate_id=second)
        if pair["iou"] < min_iou:
            continue
        previous = mapping.get(second)
        collision = first in mapping.values() and previous != first
        if previous is not None and previous != first or collision:
            row["identity_consistent"] = False
        else:
            mapping[second] = first
    row["passed"] &= row["identity_consistent"]


class OnnxBundle:
    """Read-only bundle; lazy sessions use the sample's dynamically registered RTX EP."""

    def __init__(self, args, family):
        self.args, self.family = args, family
        self.directory = args.onnx_dir.resolve()
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8-sig"))
        self.graphs = self.metadata["graphs"]
        self.sessions = {}
        self.calls = {}
        self.comparisons = {}
        self.component_calls = {}
        self.component_backends = {}
        self.export_reference_calls = {}
        self.frame_comparisons = []
        self.active_graph = None
        self.dll_handles = []
        self.ep_name = None
        if not self.graphs and (args.validate_only or args.onnx_components):
            raise ValueError("metadata.json lists no graphs")
        # The families have different metadata schemas. Reject accidental cross-family use.
        recorded = self.metadata.get("model_family", "")
        if (
                recorded
                and not recorded.startswith(family)
                or family == "sam2"
                and "vision_encoder" in self.graphs
                or family == "sam3"
                and "image_encoder" in self.graphs
        ):
            raise ValueError(f"Bundle does not match {family}")
        if (
                args.provider == "cpu"
                and self.metadata.get("dtype") != "float32"
                and args.onnx_components
                and not args.validate_only
        ):
            raise ValueError("CPU inference requires a float32 CPU-compatible bundle")

    def path(self, name):
        if name not in self.graphs:
            raise ValueError(f"Required graph {name} is missing from metadata.json")
        filename = self.graphs[name].get("file", name + ".onnx") if isinstance(self.graphs, dict) else name + ".onnx"
        path = (self.directory / filename).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError(f"Graph path escapes model bundle: {name}")
        return path

    def validate(self, names=None):
        names = (self.args.graphs or list(self.graphs)) if names is None else sorted(names)
        if not names:
            return {}
        import onnx

        if set(names) - set(self.graphs):
            raise ValueError("--graphs must select names listed in metadata.json")
        report = {}
        for name in names:
            try:
                path = self.path(name)
                onnx.checker.check_model(str(path), full_check=True)
                row = {"passed": True, "onnx_checked": True}
                if self.args.fixtures:
                    if not all(path.with_suffix(suffix).is_file() for suffix in (".inputs.npz", ".outputs.npz")):
                        hint = "export with --fixtures or regenerate with --fixtures-only" if self.family == "sam3" else "the SAM2 exporter does not save fixtures"
                        raise ValueError(f"Missing export fixtures; {hint}")
                    with np.load(path.with_suffix(".inputs.npz")) as inputs:
                        actual = self.run_numpy(name, dict(inputs))
                    with np.load(path.with_suffix(".outputs.npz")) as expected:
                        row["outputs"] = {
                            key: tensor_stats(value, expected[key], self.args.atol, self.args.rtol)
                            for key, value in actual.items()
                        }
                    row["passed"] = all(value["passed"] for value in row["outputs"].values())
                report[name] = row
            except Exception as error:
                report[name] = {"passed": False, "error": str(error)}
            print(json.dumps({"graph": name, **report[name]}), flush=True)
        return report

    def session(self, name):
        import onnxruntime as ort

        self.active_graph = name
        if name not in self.sessions:
            print(f"Loading ORT {self.family}/{name} ({self.args.provider})", flush=True)
            options = ort.SessionOptions()
            options.log_severity_level = 3
            providers = ["CPUExecutionProvider"]
            if self.args.provider == "trt-rtx":
                if self.ep_name is None:
                    if hasattr(os, "add_dll_directory"):
                        for directory in (self.args.ep_dll_dir, self.args.trt_bin):
                            if directory:
                                self.dll_handles.append(os.add_dll_directory(str(directory.resolve())))
                    if self.args.ep_lib:
                        self.ep_name, library = "nv_tensorrt_rtx", str(self.args.ep_lib.resolve())
                    else:
                        try:
                            import onnxruntime_ep_nv_tensorrt_rtx as ep
                        except ImportError as error:
                            raise RuntimeError("Install onnxruntime-ep-nv-tensorrt-rtx or supply --ep-lib") from error
                        self.ep_name, library = ep.get_ep_name(), ep.get_library_path()
                    ort.register_execution_provider_library(self.ep_name, library)
                devices = [device for device in ort.get_ep_devices() if device.ep_name == self.ep_name]
                if not devices:
                    raise RuntimeError("TensorRT RTX EP has no available device")
                cache = (self.args.ep_cache or self.args.output / "ep_cache") / self.family / name
                cache.mkdir(parents=True, exist_ok=True)
                options.add_provider_for_devices([devices[0]], {"nv_runtime_cache_path": str(cache)})
                providers = None
            self.sessions[name] = ort.InferenceSession(str(self.path(name)), sess_options=options, providers=providers)
            self.sessions[name].disable_fallback()
        return self.sessions[name]

    def run_numpy(self, name, values):
        session = self.session(name)
        if any(info.type == "tensor(bfloat16)" for info in [*session.get_inputs(), *session.get_outputs()]):
            import torch

            actual = self.run(name, {key: torch.as_tensor(value) for key, value in values.items()}, output_device="cpu")
            return {
                key: value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()
                for key, value in actual.items()
            }
        types = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(bool)": np.bool_,
            "tensor(uint8)": np.uint8,
        }
        feeds = {
            info.name: np.ascontiguousarray(values[info.name], dtype=types[info.type]) for info in session.get_inputs()
        }
        actual = session.run(None, feeds)
        self.calls[name] = self.calls.get(name, 0) + 1
        result = dict(zip((value.name for value in session.get_outputs()), actual, strict=True))
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError(f"Non-finite output from {name}")
        return result

    def run(self, name, values, *, output_device=None):
        """Bind CPU Torch storage, including BF16 (which NumPy cannot represent).

        RTX plugin devices carry NVIDIA vendor IDs. Generic CUDA IOBinding memory
        lacks that ID; CPU staging uses the plugin's supported transfer path.
        This is a correctness verifier, not a zero-copy performance benchmark.
        """
        import torch

        session = self.session(name)
        dtype = {
            "tensor(float)": torch.float32,
            "tensor(float16)": torch.float16,
            "tensor(bfloat16)": torch.bfloat16,
            "tensor(int64)": torch.int64,
            "tensor(int32)": torch.int32,
            "tensor(bool)": torch.bool,
            "tensor(uint8)": torch.uint8,
        }
        element_type = {
            torch.float32: 1,
            torch.uint8: 2,
            torch.int32: 6,
            torch.int64: 7,
            torch.bool: 9,
            torch.float16: 10,
            torch.bfloat16: 16,
        }
        device = torch.device("cpu")
        binding = session.io_binding()
        retained = []
        for info in session.get_inputs():
            value = values[info.name].to(device=device, dtype=dtype[info.type]).contiguous()
            retained.append(value)
            binding.bind_input(
                info.name,
                device.type,
                device.index or 0,
                element_type[value.dtype],
                tuple(value.shape),
                value.data_ptr(),
            )
        # CPU output allocation supports dynamic shapes and BF16; copy via DLPack
        # is not universally available in ORT, so use a typed view of the buffer.
        for info in session.get_outputs():
            binding.bind_output(info.name, "cpu")
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        outputs = {}
        for info, value in zip(session.get_outputs(), binding.get_outputs(), strict=True):
            if info.type == "tensor(bfloat16)":
                import ctypes

                shape = tuple(value.shape())
                buffer = (ctypes.c_uint16 * int(np.prod(shape))).from_address(value.data_ptr())
                tensor = torch.from_numpy(np.ctypeslib.as_array(buffer).copy()).view(torch.bfloat16).reshape(shape)
            else:
                tensor = torch.from_numpy(value.numpy().copy())
            outputs[info.name] = tensor.to(output_device or self.args.device)
            if not bool(torch.isfinite(outputs[info.name]).all()):
                raise ValueError(f"Non-finite output: {name}/{info.name}")
        self.calls[name] = self.calls.get(name, 0) + 1
        return outputs

    def compare(self, name, actual, expected):
        """Compare real pipeline inputs against the matching export wrapper, not saved dummy inputs."""
        for key, value in expected.items():
            row = tensor_stats(
                actual[key].detach().float().cpu().numpy(),
                value.detach().float().cpu().numpy(),
                self.args.atol,
                self.args.rtol,
            )
            summary = self.comparisons.setdefault(name, {}).setdefault(
                key, {"passed": True, "calls": 0, "max_absolute_error": 0.0}
            )
            summary["calls"] += 1
            summary["passed"] &= row["passed"]
            summary["max_absolute_error"] = max(summary["max_absolute_error"], row.get("max_absolute_error", 0.0))
            if not row["passed"]:
                summary["last_failure"] = row


def run_native(args, family, source, *, prompt=None, text=None, tracking=False):
    import subprocess

    output = args.output / "native"
    # Reject stale artifacts rather than accidentally accepting masks from an earlier run.
    if output.exists():
        raise FileExistsError(f"Use a fresh --output directory for --din-sam: {output}")
    if source.is_file():
        from PIL import Image

        staged = args.output / "native_input.png"
        Image.open(source).convert("RGB").save(staged)
        source = staged
    command = [
        str(args.din_sam.resolve()),
        str(source.resolve()),
        str(output),
        "--model",
        family,
        "--model-dir",
        str(args.onnx_dir),
        "--provider",
        args.provider,
        "--ep-cache",
        str(args.ep_cache or args.output / "native_ep_cache"),
    ]
    command += ["--prompt", str(prompt.resolve())] if prompt else ["--text", text]
    if family == "sam3":
        command += ["--threshold", str(args.threshold)]
    if tracking and args.limit_frames:
        command += ["--max-frames", str(args.limit_frames)]
    subprocess.run(command, check=True)
    return output
