# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# !/usr/bin/env python3
"""Validate decoder_prefill.onnx against repeated decoder.onnx steps.

The test runs a short prompt through both paths and compares every logit plus
the final self-KV cache. It also executes the prefill graph at the configured
profile maximum. With ``--provider trt-rtx`` it creates separate TensorRT-RTX
sessions using the intended static-decode and dynamic-prefill profiles.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

EP_NAME = "nv_tensorrt_rtx"


def register_trt_rtx(args) -> list:
    if args.ep_lib:
        for directory in (args.ep_dll_dir, args.trt_bin):
            if directory and os.path.isdir(directory):
                os.add_dll_directory(os.path.abspath(directory))
        ort.register_execution_provider_library(EP_NAME, str(args.ep_lib))
    else:
        try:
            import onnxruntime_ep_nv_tensorrt_rtx as ep
        except ImportError as exc:
            raise SystemExit("trt-rtx needs onnxruntime-ep-nv-tensorrt-rtx or --ep-lib") from exc
        ort.register_execution_provider_library(ep.get_ep_name(), ep.get_library_path())

    devices = [device for device in ort.get_ep_devices() if device.ep_name == EP_NAME]
    if not devices:
        raise SystemExit(f"EP '{EP_NAME}' registered but no devices were discovered")
    return devices


def make_session(path: Path, provider: str, devices, profile, cache_path: Path | None):
    options = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    if provider == "trt-rtx":
        minimum, optimum, maximum = profile
        ep_options = {
            "nv_profile_min_shapes": minimum,
            "nv_profile_opt_shapes": optimum,
            "nv_profile_max_shapes": maximum,
        }
        if cache_path:
            ep_options["nv_runtime_cache_path"] = str(cache_path)
        options.add_provider_for_devices(devices, ep_options)
        providers = None
    return ort.InferenceSession(str(path), sess_options=options, providers=providers)


def numpy_dtype(session) -> type[np.generic]:
    cache_type = session.get_inputs()[3].type
    if cache_type == "tensor(float16)":
        return np.float16
    if cache_type == "tensor(float)":
        return np.float32
    raise RuntimeError(f"unsupported decoder cache type: {cache_type}")


def zero_cache(session, dtype) -> dict[str, np.ndarray]:
    return {
        value.name: np.zeros(tuple(int(dimension) for dimension in value.shape), dtype)
        for value in session.get_inputs()[3:]
    }


def run_prefill(session, tokens, cache):
    length = tokens.shape[1]
    feeds = {
        "input_ids": tokens,
        "write_indices": np.arange(length, dtype=np.int64),
        "nonpad_kv_seqlen": np.array([length], dtype=np.int64),
        **cache,
    }
    return dict(zip((value.name for value in session.get_outputs()), session.run(None, feeds), strict=True))


def run_decode(session, tokens, cache):
    logits = []
    values = None
    for position in range(tokens.shape[1]):
        feeds = {
            "input_ids": tokens[:, position:position + 1],
            "write_indices": np.array([position], dtype=np.int64),
            "nonpad_kv_seqlen": np.array([position + 1], dtype=np.int64),
            **cache,
        }
        values = dict(zip((value.name for value in session.get_outputs()), session.run(None, feeds), strict=True))
        logits.append(values["logits"])
        for name in list(cache):
            if name.startswith("past_key_self_") or name.startswith("past_value_self_"):
                cache[name] = values[name.replace("past_", "present_")]
    values["logits"] = np.concatenate(logits, axis=1)
    return values, cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=["cpu", "trt-rtx"], default="cpu")
    parser.add_argument("--parity-length", type=int, default=4)
    parser.add_argument("--prefill-max-length", type=int, default=224)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--ep-lib", type=Path, default=None)
    parser.add_argument("--ep-dll-dir", type=Path, default=None)
    parser.add_argument("--trt-bin", type=Path, default=None)
    parser.add_argument("--ep-cache-dir", type=Path, default=None)
    args = parser.parse_args()

    if not 1 <= args.parity_length <= args.prefill_max_length:
        raise SystemExit("parity length must be within the prefill profile")

    devices = register_trt_rtx(args) if args.provider == "trt-rtx" else None
    one = "input_ids:1x1,write_indices:1"
    maximum = f"input_ids:1x{args.prefill_max_length},write_indices:{args.prefill_max_length}"
    decode_cache = args.ep_cache_dir / "decoder" if args.ep_cache_dir else None
    prefill_cache = args.ep_cache_dir / "decoder_prefill" if args.ep_cache_dir else None
    decode = make_session(args.onnx_dir / "decoder.onnx", args.provider, devices, (one, one, one), decode_cache)
    prefill = make_session(
        args.onnx_dir / "decoder_prefill.onnx", args.provider, devices, (one, maximum, maximum), prefill_cache
    )

    dtype = numpy_dtype(decode)
    if numpy_dtype(prefill) != dtype:
        raise RuntimeError("decoder and prefill cache types differ")
    vocab_size = int(decode.get_outputs()[0].shape[2])
    tokens = (np.arange(args.parity_length, dtype=np.int32) % vocab_size).reshape(1, -1)
    prefill_result = run_prefill(prefill, tokens, zero_cache(prefill, dtype))
    decode_result, decode_cache_values = run_decode(decode, tokens, zero_cache(decode, dtype))

    logits_diff = float(np.max(np.abs(prefill_result["logits"].astype(np.float32)
                                      - decode_result["logits"].astype(np.float32))))
    cache_diff = 0.0
    for name, value in decode_cache_values.items():
        if name.startswith(("past_key_self_", "past_value_self_")):
            present_name = name.replace("past_", "present_")
            cache_diff = max(cache_diff, float(np.max(np.abs(
                prefill_result[present_name].astype(np.float32) - value.astype(np.float32)
            ))))

    boundary_tokens = np.zeros((1, args.prefill_max_length), dtype=np.int32)
    boundary = run_prefill(prefill, boundary_tokens, zero_cache(prefill, dtype))["logits"]
    default_tolerance = 5e-2 if dtype == np.float16 else (1e-3 if args.provider == "trt-rtx" else 1e-4)
    tolerance = args.atol if args.atol is not None else default_tolerance
    print(f"provider={args.provider} dtype={dtype.__name__}")
    print(f"prefill boundary logits={boundary.shape}")
    print(f"parity logits max_abs_diff={logits_diff:.6g}")
    print(f"parity cache max_abs_diff={cache_diff:.6g}")
    if logits_diff > tolerance or cache_diff > tolerance:
        raise SystemExit(f"prefill parity failed (atol={tolerance})")


if __name__ == "__main__":
    main()
