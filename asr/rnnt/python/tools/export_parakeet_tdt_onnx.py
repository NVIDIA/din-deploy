# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from rnnt.parakeet_tdt.nemo_backend import (
    DEFAULT_MODEL_ID,
    ONNX_DIR,
    apply_attention,
    configure_encoder_sdpa,
    load_nemo_model,
    write_tokenizer_artifacts,
)

from nemo_preprocessor_export import export_nemo_preprocessor

ONNX_GRAPH_NAMES = ("preprocessor", "encoder", "predict_step")


class _EncoderExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_features, attention_mask):
        compute_dtype = next(self.model.encoder.parameters()).dtype
        lengths = attention_mask.sum(dim=-1).to(torch.long)
        encoded, encoded_lengths = self.model.encoder(
            audio_signal=input_features.to(dtype=compute_dtype).transpose(1, 2),
            length=lengths,
        )
        states = encoded.transpose(1, 2)
        states = project_encoder_for_joint(self.model, states)
        frame_idx = torch.arange(states.shape[1], device=states.device)
        encoder_mask = frame_idx[None] < encoded_lengths[:, None]
        return states.float(), encoder_mask


class _PredictStepExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.decoder = model.decoder
        self.joint = model.joint
        self.blank_id = blank_token_id(model)
        self.token_vocab_size = self.blank_id + 1

    def forward(self, input_id, hidden, cell, encoder_frame):
        compute_dtype = next(self.decoder.parameters()).dtype
        decoder_output, (next_hidden, next_cell) = self.decoder.predict(
            y=input_id,
            state=(hidden.to(dtype=compute_dtype), cell.to(dtype=compute_dtype)),
            add_sos=False,
            batch_size=input_id.shape[0],
        )
        logits = joint_from_projected_encoder(
            self.joint,
            encoder_frame.to(dtype=compute_dtype),
            decoder_output.to(dtype=compute_dtype),
        )
        token_id = logits[..., : self.token_vocab_size].argmax(dim=-1).to(torch.int64)
        duration_index = logits[..., self.token_vocab_size :].argmax(dim=-1).to(torch.int64)
        control = torch.stack((token_id, duration_index), dim=-1)
        return control, next_hidden.float(), next_cell.float()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    model = load_nemo_model(args.model, device=device, dtype=dtype)
    if args.local_window:
        apply_attention(model, "rel_pos_local_attn", [256, 256])
    configure_encoder_sdpa(model.encoder)
    extend_encoder_positional_encoding(model, args.encoder_profile_frames)
    model.eval()

    model, quantization_info = restore_quantized_modelopt_checkpoint(model, args)
    graphs = parse_onnx_graph_names(args.graphs)
    if "preprocessor" in graphs:
        export_preprocessor(model, args.output, args.opset, args.external_data, args.dynamo, args.optimize)
    if "encoder" in graphs:
        export_encoder(model, args.output, device, args.opset, args.external_data, args.dynamo, args.optimize)
    if "predict_step" in graphs:
        export_predict_step(model, args.output, device, args.opset, args.external_data, args.dynamo, args.optimize)
    if args.onnx_int4:
        quantization_info["onnx_int4"] = quantize_exported_onnx_graphs(
            args.output,
            args.onnx_int4_targets,
            args.external_data,
            args.onnx_int4_calibration_method,
            args.onnx_int4_calibration_eps,
        )
    write_metadata(model, args.output, args.dtype, args.encoder_profile_frames, quantization_info)
    write_tokenizer_artifacts(model, args.output)


def export_preprocessor(
    model,
    output: Path,
    opset: int,
    external_data: bool,
    dynamo: bool,
    optimize: bool,
) -> None:
    export_nemo_preprocessor(
        model.preprocessor,
        output / "preprocessor.onnx",
        output_kind="mask",
        output_names=["input_features", "attention_mask"],
        opset=opset,
        external_data=external_data,
        dynamo=dynamo,
        optimize=optimize,
        length_dtype=torch.int64,
    )


def export_encoder(
    model,
    output: Path,
    device: torch.device,
    opset: int,
    external_data: bool,
    dynamo: bool,
    optimize: bool,
) -> None:
    features = torch.zeros(1, 400, num_mel_bins(model), device=device, dtype=torch.float32)
    mask = torch.ones(1, 400, device=device, dtype=torch.bool)
    torch.onnx.export(
        _EncoderExportWrapper(model).eval(),
        (features, mask),
        str(output / "encoder.onnx"),
        input_names=["input_features", "attention_mask"],
        output_names=["encoder_states", "encoder_mask"],
        dynamic_axes={
            "input_features": {1: "frames"},
            "attention_mask": {1: "frames"},
            "encoder_states": {1: "encoded_frames"},
            "encoder_mask": {1: "encoded_frames"},
        },
        opset_version=opset,
        dynamo=dynamo,
        optimize=optimize,
        external_data=external_data,
    )


def export_predict_step(
    model,
    output: Path,
    device: torch.device,
    opset: int,
    external_data: bool,
    dynamo: bool,
    optimize: bool,
) -> None:
    model.decoder.float()
    model.joint.float()
    input_id = torch.tensor([[blank_token_id(model)]], device=device, dtype=torch.long)
    shape = (decoder_layers(model), 1, decoder_hidden_size(model))
    hidden = torch.zeros(shape, device=device, dtype=torch.float32)
    cell = torch.zeros_like(hidden)
    encoder_frame = torch.zeros(1, 1, joint_hidden_size(model), device=device, dtype=torch.float32)
    torch.onnx.export(
        _PredictStepExportWrapper(model),
        (input_id, hidden, cell, encoder_frame),
        str(output / "predict_step.onnx"),
        input_names=["input_id", "hidden", "cell", "encoder_frame"],
        output_names=["control", "next_hidden", "next_cell"],
        opset_version=opset,
        dynamo=dynamo,
        optimize=optimize,
        external_data=external_data,
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


def quantize_exported_onnx_graphs(
    output: Path,
    targets: str,
    external_data: bool,
    calibration_method: str,
    calibration_eps: str,
) -> dict[str, Any]:
    try:
        import onnx
        from modelopt.onnx.quantization.int4 import quantize as quantize_int4  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "ONNX INT4 export requires `pip install nvidia-modelopt` or `pip install -e .[quantization]`."
        ) from error

    graph_names = parse_onnx_graph_names(targets)
    providers = split_csv(calibration_eps)
    quantized_paths = []
    for graph_name in graph_names:
        onnx_path = output / f"{graph_name}.onnx"
        quantized_model = quantize_int4(
            str(onnx_path),
            calibration_method=calibration_method,
            calibration_data_reader=None,
            calibration_eps=providers,
        )
        onnx.save_model(
            quantized_model,
            onnx_path,
            save_as_external_data=external_data,
            location=onnx_path.name + "_data",
            size_threshold=0 if external_data else 1024,
        )
        quantized_paths.append(str(onnx_path))
    return {
        "mode": "modelopt-onnx-int4",
        "targets": graph_names,
        "calibration_method": calibration_method,
        "calibration_data": "modelopt-random",
        "calibration_eps": providers,
        "paths": quantized_paths,
    }


def write_metadata(
    model,
    output: Path,
    dtype: str,
    encoder_profile_frames: int,
    quantization_info: dict[str, Any] | None = None,
) -> None:
    metadata = {
        "blank_token_id": blank_token_id(model),
        "pad_token_id": pad_token_id(model),
        "vocab_size": blank_token_id(model) + 1,
        "durations": list(durations(model)),
        "max_symbols_per_step": int(getattr(model.cfg, "max_symbols_per_step", 10)),
        "decoder_hidden_size": decoder_hidden_size(model),
        "joint_dim": joint_hidden_size(model),
        "num_decoder_layers": decoder_layers(model),
        "sampling_rate": sampling_rate(model),
        "hop_length": hop_length(model),
        "subsampling_factor": subsampling_factor(model),
        "encoder_profile_frames": encoder_profile_frames,
        "onnx_dtype": dtype,
        "boundary_dtype": "float32",
    }
    if quantization_info:
        metadata["quantization"] = quantization_info
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def blank_token_id(model: nn.Module) -> int:
    return int(model.cfg.decoder.vocab_size)


def pad_token_id(model: nn.Module) -> int:
    tokenizer = getattr(model, "tokenizer", None)
    for attr in ("pad_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return int(getattr(model.cfg, "pad_token_id", 0))


def durations(model: nn.Module) -> tuple[int, ...]:
    for value in (
        getattr(getattr(model.cfg, "model_defaults", None), "tdt_durations", None),
        getattr(getattr(model.cfg, "joint", None), "durations", None),
        getattr(model.cfg, "durations", None),
    ):
        if value is not None:
            return tuple(int(item) for item in value)
    return (0, 1, 2, 3, 4)


def decoder_hidden_size(model: nn.Module) -> int:
    prednet = getattr(model.cfg.decoder, "prednet", model.cfg.decoder)
    return int(prednet.pred_hidden)


def decoder_layers(model: nn.Module) -> int:
    prednet = getattr(model.cfg.decoder, "prednet", model.cfg.decoder)
    return int(prednet.pred_rnn_layers)


def joint_hidden_size(model: nn.Module) -> int:
    return int(getattr(model.cfg.joint.jointnet, "joint_hidden", decoder_hidden_size(model)))


def sampling_rate(model: nn.Module) -> int:
    return int(model.cfg.preprocessor.sample_rate)


def hop_length(model: nn.Module) -> int:
    return round(float(model.cfg.preprocessor.window_stride) * sampling_rate(model))


def subsampling_factor(model: nn.Module) -> int:
    return int(getattr(model.cfg.encoder, "subsampling_factor", 8))


def num_mel_bins(model: nn.Module) -> int:
    return int(model.cfg.preprocessor.features)


def max_encoded_frames(model: nn.Module, input_frames: int) -> int:
    factor = subsampling_factor(model)
    return (input_frames - 1) // factor + 1


def extend_encoder_positional_encoding(model: nn.Module, encoder_profile_frames: int) -> None:
    pos_enc = getattr(model.encoder, "pos_enc", None)
    if pos_enc is None or not hasattr(pos_enc, "extend_pe"):
        return
    param = next(model.encoder.parameters())
    pos_enc.extend_pe(max_encoded_frames(model, encoder_profile_frames), param.device, param.dtype)


def validate_args(args: argparse.Namespace) -> None:
    parse_onnx_graph_names(args.onnx_int4_targets)
    parse_onnx_graph_names(args.graphs)
    if args.encoder_profile_frames < 16:
        raise ValueError("--encoder-profile-frames must be at least 16.")
    if args.onnx_int4 and args.quantized_modelopt_path is not None:
        raise ValueError("Choose either --onnx-int4 or --quantized-modelopt-path for INT4 export, not both.")
    if (args.onnx_int4 or args.quantized_modelopt_path is not None) and args.opset < 21:
        raise ValueError("INT4 ONNX export requires --opset 21 or newer.")


def parse_onnx_graph_names(value: str) -> list[str]:
    names = split_csv(value)
    if not names:
        raise ValueError("At least one ONNX INT4 target is required.")
    if "all" in names:
        if len(names) > 1:
            raise ValueError("--onnx-int4-targets cannot combine 'all' with explicit graph names.")
        return list(ONNX_GRAPH_NAMES)
    unknown = sorted(set(names) - set(ONNX_GRAPH_NAMES))
    if unknown:
        raise ValueError(f"Unsupported ONNX INT4 target(s): {', '.join(unknown)}.")
    return names


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Parakeet TDT ONNX artifacts for the C++ baseline.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, default=ONNX_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--opset", type=int, default=23)
    parser.add_argument("--external-data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-window", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graphs", default="preprocessor,encoder,predict_step")
    parser.add_argument(
        "--encoder-profile-frames",
        type=int,
        default=65536,
        help="Maximum mel frames used by the C++ TensorRT encoder profile.",
    )
    parser.add_argument(
        "--quantized-modelopt-path",
        type=Path,
        default=None,
        help="Restore a saved ModelOpt checkpoint before exporting with torch.onnx.export.",
    )
    parser.add_argument(
        "--onnx-int4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Post-process exported ONNX graphs with ModelOpt ONNX INT4 PTQ. Requires --opset 21 or newer.",
    )
    parser.add_argument(
        "--onnx-int4-targets",
        default="encoder",
        help="Comma-separated ONNX graphs to quantize: preprocessor, encoder, predict_step, or all.",
    )
    parser.add_argument("--onnx-int4-calibration-method", default="awq_lite")
    parser.add_argument(
        "--onnx-int4-calibration-eps",
        default="dml,cpu",
        help="Comma-separated calibration execution providers passed to ModelOpt ONNX INT4 PTQ.",
    )
    return parser.parse_args()


def project_encoder_for_joint(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    joint = model.joint
    if hasattr(joint, "project_encoder"):
        states = joint.project_encoder(states)
        if states.dim() == 4 and states.shape[2] == 1:
            states = states.squeeze(2)
        return states
    projector = getattr(model, "encoder_projector", None)
    if projector is not None:
        return projector(states)
    return states


def joint_from_projected_encoder(
    joint: nn.Module, encoder_frame: torch.Tensor, decoder_output: torch.Tensor
) -> torch.Tensor:
    if hasattr(joint, "project_prednet") and hasattr(joint, "joint_after_projection"):
        logits = joint.joint_after_projection(encoder_frame, joint.project_prednet(decoder_output))
    else:
        logits = joint(encoder_frame, decoder_output)
    if logits.dim() == 4 and logits.shape[2] == 1:
        logits = logits.squeeze(2)
    return logits


if __name__ == "__main__":
    with torch.inference_mode():
        main()
