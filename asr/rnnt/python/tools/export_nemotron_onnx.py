# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import nn

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from rnnt.nemotron_asr.nemo_backend import (
    DEFAULT_ATT_CONTEXT_SIZE,
    DEFAULT_MODEL_ID,
    load_nemo_model,
    prompt_dictionary,
    prompt_id,
    write_tokenizer_artifacts,
)

from nemo_preprocessor_export import export_nemo_preprocessor

ARTIFACTS_DIR = Path("artifacts") / "nemotron"
ONNX_DIR = ARTIFACTS_DIR / "onnx"
ONNX_GRAPH_NAMES = ("preprocessor", "encoder_step", "decoder_init", "predict_step")
PREPROCESSOR_PROFILE_FRAMES = 65_536
SUPPORTED_CHUNK_SIZES = (1, 2, 4, 7, 14)
LEFT_CONTEXT_FRAMES = 56
LEFT_CHUNKS = 2
NEXT_SAMPLING_FRAMES = 8
NEXT_PRE_ENCODE_CACHE_FRAMES = 9
T = TypeVar("T")


@dataclass(frozen=True)
class StreamingConfig:
    chunk_size: int
    right_context: int
    first_chunk_feature_frames: int
    next_chunk_feature_frames: int
    first_shift_feature_frames: int
    next_shift_feature_frames: int
    first_pre_encode_cache_frames: int
    next_pre_encode_cache_frames: int
    first_drop_extra_pre_encoded: int
    next_drop_extra_pre_encoded: int
    valid_out_len: int

    @property
    def max_input_feature_frames(self) -> int:
        return max(
            self.first_pre_encode_cache_frames + self.first_chunk_feature_frames,
            self.next_pre_encode_cache_frames + self.next_chunk_feature_frames,
        )


def att_context_size_for_chunk_size(chunk_size: int) -> tuple[int, int]:
    if chunk_size not in SUPPORTED_CHUNK_SIZES:
        raise ValueError(f"chunk_size must be one of {SUPPORTED_CHUNK_SIZES}.")
    return LEFT_CONTEXT_FRAMES, chunk_size - 1


def streaming_config_for_chunk_size(chunk_size: int, subsampling_factor: int = 8) -> StreamingConfig:
    if chunk_size not in SUPPORTED_CHUNK_SIZES:
        raise ValueError(f"chunk_size must be one of {SUPPORTED_CHUNK_SIZES}.")
    right_context = chunk_size - 1
    next_chunk_frames = NEXT_SAMPLING_FRAMES + subsampling_factor * right_context
    first_chunk_frames = next_chunk_frames - right_context - 1
    valid_out_len = chunk_size
    return StreamingConfig(
        chunk_size=chunk_size,
        right_context=right_context,
        first_chunk_feature_frames=first_chunk_frames,
        next_chunk_feature_frames=next_chunk_frames,
        first_shift_feature_frames=first_chunk_frames,
        next_shift_feature_frames=next_chunk_frames,
        first_pre_encode_cache_frames=0,
        next_pre_encode_cache_frames=NEXT_PRE_ENCODE_CACHE_FRAMES,
        first_drop_extra_pre_encoded=2,
        next_drop_extra_pre_encoded=2,
        valid_out_len=valid_out_len,
    )


class _EncoderStepExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_features,
        feature_lengths,
        prompt,
        att_cache,
        conv_cache,
        right_context,
        cache_last_channel_len,
    ):
        compute_dtype = next(self.model.encoder.parameters()).dtype
        encoded, output_lengths, next_att_cache, next_conv_cache, next_cache_last_channel_len = (
            self.model.encoder.forward_for_export(
                audio_signal=input_features.to(dtype=compute_dtype).transpose(1, 2),
                length=feature_lengths.to(torch.long),
                cache_last_channel=att_cache.to(dtype=compute_dtype).transpose(0, 1),
                cache_last_time=conv_cache.to(dtype=compute_dtype).transpose(0, 1),
                cache_last_channel_len=cache_last_channel_len.to(torch.long),
            )
        )
        encoder_states = encoded.transpose(1, 2)
        prompt = prompt[:, None, :].expand(-1, encoder_states.shape[1], -1).to(dtype=encoder_states.dtype)
        encoder_states = self.model.prompt_kernel(torch.cat([encoder_states, prompt], dim=-1)).float()
        # Keep right_context in the graph ABI even though NeMo bakes the value into
        # the configured export cache path.
        encoder_states = encoder_states + right_context.to(dtype=encoder_states.dtype).sum() * 0
        return (
            encoder_states,
            output_lengths.to(torch.int32),
            next_att_cache.transpose(0, 1).float(),
            next_conv_cache.transpose(0, 1).float(),
            next_cache_last_channel_len.to(torch.int32),
        )


class _PredictStepExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.decoder = model.decoder
        self.joint = model.joint

    def forward(self, input_id, hidden, cell, encoder_frame):
        compute_dtype = next(self.decoder.parameters()).dtype
        hidden = hidden.to(dtype=compute_dtype)
        cell = cell.to(dtype=compute_dtype)
        encoder_frame = encoder_frame.to(dtype=compute_dtype)
        decoder_output = hidden[-1:].transpose(0, 1)
        enc = self.joint.project_encoder(encoder_frame)
        pred = self.joint.project_prednet(decoder_output)
        logits = self.joint.joint_after_projection(enc, pred).squeeze(2)
        token_id = logits.argmax(dim=-1).to(torch.int64)
        token_id = token_id + input_id.sum().to(torch.int64) * 0
        _, (next_hidden, next_cell) = self.decoder.predict(y=token_id, state=(hidden, cell), add_sos=False)
        return token_id, next_hidden.float(), next_cell.float()


class _DecoderInitExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.decoder = model.decoder
        self.encoder_dim = int(model.cfg.joint.jointnet.encoder_hidden)
        self.blank_id = int(model.cfg.decoder.vocab_size)

    def forward(self):
        device = next(self.decoder.parameters()).device
        state = self.decoder.initialize_state(torch.zeros(1, 1, self.encoder_dim, device=device))
        _, (hidden, cell) = self.decoder.predict(
            torch.tensor([[self.blank_id]], dtype=torch.long, device=device),
            state,
            add_sos=False,
            batch_size=1,
        )
        return hidden.float(), cell.float()


def export_options(
    *,
    opset: int,
    dynamo: bool,
    optimize: bool,
    external_data: bool,
    dynamic_axes: dict | None = None,
    dynamic_shapes: object | None = None,
) -> dict:
    options = {
        "opset_version": opset,
        "dynamo": dynamo,
        "optimize": optimize,
        "external_data": external_data,
    }
    if dynamo:
        if dynamic_shapes is not None:
            options["dynamic_shapes"] = dynamic_shapes
    elif dynamic_axes is not None:
        options["dynamic_axes"] = dynamic_axes
    return options


def quiet_step[T](name: str, enabled: bool, fn: Callable[[], T]) -> T:
    if not enabled:
        print(name, file=sys.stderr)
        return fn()

    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = fn()
    except Exception:
        captured = stdout.getvalue() + stderr.getvalue()
        if captured:
            print(captured, file=sys.stderr, end="")
        raise
    print(name, file=sys.stderr)
    return result


def main() -> None:
    args = parse_args()
    args.att_context_size = att_context_size_for_chunk_size(args.chunk_size)
    args.streaming_config = streaming_config_for_chunk_size(args.chunk_size)
    args.encoder_step_input_frames = args.streaming_config.max_input_feature_frames
    validate_args(args)
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    args.torch_dtype = dtype
    model = quiet_step(
        "Loaded NeMo model.",
        args.quiet,
        lambda: load_nemo_model(
            args.model,
            device=device,
            target_lang=args.target_lang,
            att_context_size=args.att_context_size,
            dtype=dtype,
            configure_for_export=True,
        ),
    )
    if hasattr(model.encoder, "setup_streaming_params"):
        model.encoder.setup_streaming_params(
            chunk_size=args.chunk_size,
            left_chunks=LEFT_CHUNKS,
            shift_size=args.chunk_size,
        )
    model.eval()
    model, _ = restore_quantized_modelopt_checkpoint(model, args)

    graphs = parse_graph_names(args.graphs)
    if "preprocessor" in graphs:
        quiet_step(
            "Exported preprocessor.onnx.",
            args.quiet,
            lambda: export_preprocessor(model, args.output, args.opset, args.external_data, args.dynamo, args.optimize),
        )
    if "encoder_step" in graphs:
        quiet_step(
            "Exported encoder_step.onnx.",
            args.quiet,
            lambda: export_encoder_step(
                model,
                args.output,
                device,
                args.opset,
                args.external_data,
                args.dynamo,
                args.optimize,
                args,
            ),
        )
    if "decoder_init" in graphs:
        quiet_step(
            "Exported decoder_init.onnx.",
            args.quiet,
            lambda: export_decoder_init(
                model, args.output, device, args.opset, args.external_data, args.dynamo, args.optimize
            ),
        )
    if "predict_step" in graphs:
        quiet_step(
            "Exported predict_step.onnx.",
            args.quiet,
            lambda: export_predict_step(
                model, args.output, device, args.opset, args.external_data, args.dynamo, args.optimize
            ),
        )
    write_metadata(model, args.output, args)
    write_decoder_initial_state(model, args.output, device)
    write_tokenizer_artifacts(model, args.output)


def export_preprocessor(
    model: nn.Module,
    output: Path,
    opset: int,
    external_data: bool,
    dynamo: bool,
    optimize: bool,
) -> None:
    export_nemo_preprocessor(
        model.preprocessor,
        output / "preprocessor.onnx",
        output_kind="lengths",
        output_names=["input_features", "feature_lengths"],
        opset=opset,
        external_data=external_data,
        dynamo=dynamo,
        optimize=optimize,
        length_dtype=torch.int32,
    )


def export_encoder_step(
    model: nn.Module,
    output: Path,
    device: torch.device,
    opset: int,
    external_data: bool,
    dynamo: bool,
    optimize: bool,
    args: argparse.Namespace,
) -> None:
    attention_cache_frames = int(model.encoder.streaming_cfg.last_channel_cache_size)
    features = torch.zeros(
        1, args.encoder_step_input_frames, int(model.cfg.preprocessor.features), device=device, dtype=torch.float32
    )
    lengths = torch.tensor([args.encoder_step_input_frames], device=device, dtype=torch.int32)
    prompt = torch.zeros(1, 128, device=device, dtype=torch.float32)
    prompt[:, prompt_id(model, args.target_lang)] = 1.0
    att_cache = torch.zeros(
        int(model.cfg.encoder.n_layers),
        1,
        attention_cache_frames,
        int(model.cfg.encoder.d_model),
        device=device,
        dtype=torch.float32,
    )
    conv_cache = torch.zeros(
        int(model.cfg.encoder.n_layers),
        1,
        int(model.cfg.encoder.d_model),
        int(model.cfg.encoder.conv_kernel_size) - 1,
        device=device,
        dtype=torch.float32,
    )
    right_context = torch.tensor([args.streaming_config.right_context], device=device, dtype=torch.int32)
    cache_last_channel_len = torch.zeros(1, device=device, dtype=torch.int32)
    torch.onnx.export(
        _EncoderStepExportWrapper(model).eval(),
        (
            features,
            lengths,
            prompt,
            att_cache,
            conv_cache,
            right_context,
            cache_last_channel_len,
        ),
        str(output / "encoder_step.onnx"),
        input_names=[
            "input_features",
            "feature_lengths",
            "prompt",
            "att_cache",
            "conv_cache",
            "right_context",
            "cache_last_channel_len",
        ],
        output_names=[
            "encoder_states",
            "output_lengths",
            "next_att_cache",
            "next_conv_cache",
            "next_cache_last_channel_len",
        ],
        **export_options(opset=opset, dynamo=dynamo, optimize=optimize, external_data=external_data),
    )


def export_predict_step(
    model: nn.Module,
    output: Path,
    device: torch.device,
    opset: int,
    external_data: bool,
    dynamo: bool,
    optimize: bool,
) -> None:
    model.decoder.float()
    model.joint.float()
    input_id = torch.tensor([[int(model.cfg.decoder.vocab_size)]], device=device, dtype=torch.long)
    state_shape = (
        int(model.cfg.decoder.prednet.pred_rnn_layers),
        1,
        int(model.cfg.decoder.prednet.pred_hidden),
    )
    hidden = torch.zeros(state_shape, device=device, dtype=torch.float32)
    cell = torch.zeros_like(hidden)
    encoder_frame = torch.zeros(1, 1, int(model.cfg.joint.jointnet.encoder_hidden), device=device, dtype=torch.float32)
    torch.onnx.export(
        _PredictStepExportWrapper(model).eval(),
        (input_id, hidden, cell, encoder_frame),
        str(output / "predict_step.onnx"),
        input_names=["input_id", "hidden", "cell", "encoder_frame"],
        output_names=["token_id", "next_hidden", "next_cell"],
        **export_options(opset=opset, dynamo=dynamo, optimize=optimize, external_data=external_data),
    )


def export_decoder_init(
    model: nn.Module,
    output: Path,
    device: torch.device,
    opset: int,
    external_data: bool,
    dynamo: bool,
    optimize: bool,
) -> None:
    model.decoder.float()
    torch.onnx.export(
        _DecoderInitExportWrapper(model).eval(),
        (),
        str(output / "decoder_init.onnx"),
        input_names=[],
        output_names=["hidden", "cell"],
        **export_options(opset=opset, dynamo=dynamo, optimize=optimize, external_data=external_data),
    )


def restore_quantized_modelopt_checkpoint(model, args: argparse.Namespace) -> tuple[object, dict[str, Any]]:
    if args.quantized_modelopt_path is None:
        return model, {}
    from rnnt.validation.quantization import restore_modelopt_quantized_model

    restored_model, info = restore_modelopt_quantized_model(
        model,
        args.quantized_modelopt_path,
        summary_path=args.output / "quant_summary.txt",
    )
    return restored_model, {"modelopt_checkpoint": info}


def write_metadata(
    model: nn.Module,
    output: Path,
    args: argparse.Namespace,
) -> None:
    streaming = args.streaming_config
    metadata = {
        "blank_token_id": int(model.cfg.decoder.vocab_size),
        "max_symbols_per_step": args.max_symbols_per_step,
        "decoder_hidden_size": int(model.cfg.decoder.prednet.pred_hidden),
        "joint_dim": int(model.cfg.joint.jointnet.encoder_hidden),
        "num_decoder_layers": int(model.cfg.decoder.prednet.pred_rnn_layers),
        "sampling_rate": int(model.cfg.preprocessor.sample_rate),
        "hop_length": round(float(model.cfg.preprocessor.window_stride) * int(model.cfg.preprocessor.sample_rate)),
        "num_mel_bins": int(model.cfg.preprocessor.features),
        "subsampling_factor": int(model.cfg.encoder.subsampling_factor),
        "preprocessor_profile_frames": PREPROCESSOR_PROFILE_FRAMES,
        "max_mel_frames": args.encoder_step_input_frames,
        "chunk_size": args.chunk_size,
        "att_context_size": list(args.att_context_size),
        "encoder_step_input_frames": args.encoder_step_input_frames,
        "chunk_feature_frames_first": streaming.first_chunk_feature_frames,
        "chunk_feature_frames_next": streaming.next_chunk_feature_frames,
        "shift_feature_frames_first": streaming.first_shift_feature_frames,
        "shift_feature_frames_next": streaming.next_shift_feature_frames,
        "pre_encode_cache_frames_first": streaming.first_pre_encode_cache_frames,
        "pre_encode_cache_frames_next": streaming.next_pre_encode_cache_frames,
        "drop_extra_pre_encoded_first": streaming.first_drop_extra_pre_encoded,
        "drop_extra_pre_encoded_next": streaming.next_drop_extra_pre_encoded,
        "valid_out_len": streaming.valid_out_len,
        "right_context": streaming.right_context,
        "attention_cache_shape": [
            int(model.cfg.encoder.n_layers),
            1,
            int(model.encoder.streaming_cfg.last_channel_cache_size),
            int(model.cfg.encoder.d_model),
        ],
        "convolution_cache_shape": [
            int(model.cfg.encoder.n_layers),
            1,
            int(model.cfg.encoder.d_model),
            int(model.cfg.encoder.conv_kernel_size) - 1,
        ],
        "prompt_dictionary": prompt_dictionary(model),
        "prompt_id": prompt_id(model, args.target_lang),
        "strip_lang_tags": args.strip_lang_tags,
        "onnx_dtype": args.dtype,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_decoder_initial_state(model: nn.Module, output: Path, device: torch.device) -> None:
    model.decoder.float().eval()
    with torch.inference_mode():
        state = model.decoder.initialize_state(
            torch.zeros(1, 1, int(model.cfg.joint.jointnet.encoder_hidden), device=device)
        )
        _, (hidden, cell) = model.decoder.predict(
            torch.tensor([[int(model.cfg.decoder.vocab_size)]], dtype=torch.long, device=device),
            state,
            add_sos=False,
            batch_size=1,
        )
    hidden.detach().cpu().float().numpy().tofile(output / "decoder_init_hidden.f32")
    cell.detach().cpu().float().numpy().tofile(output / "decoder_init_cell.f32")


def parse_chunk_size(value: str) -> int:
    try:
        chunk_size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chunk size must be an integer.") from exc
    if chunk_size not in {1, 2, 4, 7, 14}:
        raise argparse.ArgumentTypeError("chunk size must be one of 1, 2, 4, 7, or 14.")
    return chunk_size


def parse_graph_names(value: str) -> list[str]:
    if value == "all":
        return list(ONNX_GRAPH_NAMES)
    graphs = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(graphs) - set(ONNX_GRAPH_NAMES))
    if unknown:
        raise ValueError(f"Unsupported Nemotron ONNX graph(s): {', '.join(unknown)}")
    return graphs


def validate_args(args: argparse.Namespace) -> None:
    if args.max_symbols_per_step <= 0:
        raise ValueError("--max-symbols-per-step must be positive.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Nemotron ASR ONNX artifacts for the C++ pipeline.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, default=ONNX_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--opset", type=int, default=23)
    parser.add_argument("--external-data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress NeMo and torch.onnx exporter chatter unless a step fails.",
    )
    parser.add_argument("--target-lang", "--target_lang", default="en-US")
    parser.add_argument(
        "--chunk-size",
        "--chunk_size",
        type=parse_chunk_size,
        default=DEFAULT_ATT_CONTEXT_SIZE[1] + 1,
        help="Export-time streaming chunk size in 80 ms encoder frames: 1, 2, 4, 7, or 14.",
    )
    parser.add_argument("--strip-lang-tags", "--strip_lang_tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-symbols-per-step", type=int, default=10)
    parser.add_argument(
        "--quantized-modelopt-path",
        type=Path,
        default=None,
        help="Restore a saved ModelOpt encoder checkpoint before exporting.",
    )
    parser.add_argument(
        "--graphs",
        default="preprocessor,encoder_step,predict_step",
        help="Comma-separated list of graphs to export: preprocessor,encoder_step,decoder_init,predict_step.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
