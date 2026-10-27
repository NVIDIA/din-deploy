# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "model_export"))
from nemo_export import configure_encoder_sdpa, export_nemo_preprocessor, load_nemo_model

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
ONNX_DIR = Path(os.environ.get("DIN_DEPLOY_ARTIFACTS_PATH", "artifacts")) / "parakeet" / "onnx"

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
        model.change_attention_model("rel_pos_local_attn", [256, 256])
    configure_encoder_sdpa(model.encoder)
    extend_encoder_positional_encoding(model, args.encoder_profile_frames)
    model.eval()

    graphs = parse_onnx_graph_names(args.graphs)
    if "preprocessor" in graphs:
        export_preprocessor(model, args.output, args.opset, args.external_data, args.dynamo, args.optimize)
    if "encoder" in graphs:
        export_encoder(model, args.output, device, args.opset, args.external_data, args.dynamo, args.optimize)
    if "predict_step" in graphs:
        export_predict_step(model, args.output, device, args.opset, args.external_data, args.dynamo, args.optimize)
    write_metadata(model, args.output, args.dtype, args.encoder_profile_frames)
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


def write_metadata(
    model,
    output: Path,
    dtype: str,
    encoder_profile_frames: int,
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
    if args.encoder_profile_frames <= 0:
        raise ValueError("--encoder-profile-frames must be positive.")


def parse_onnx_graph_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one ONNX graph is required.")
    if "all" in names:
        if len(names) > 1:
            raise ValueError("--graphs cannot combine 'all' with explicit graph names.")
        return list(ONNX_GRAPH_NAMES)
    unknown = sorted(set(names) - set(ONNX_GRAPH_NAMES))
    if unknown:
        raise ValueError(f"Unsupported ONNX graph(s): {', '.join(unknown)}.")
    return names


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


def write_tokenizer_artifacts(model: nn.Module, output: Path) -> None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("NeMo Parakeet model does not expose a tokenizer.")

    sentencepiece = getattr(tokenizer, "tokenizer", tokenizer)
    if hasattr(sentencepiece, "get_piece_size") and hasattr(sentencepiece, "id_to_piece"):
        vocab = {sentencepiece.id_to_piece(index): index for index in range(sentencepiece.get_piece_size())}
        (output / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}}, indent=2), encoding="utf-8")
        return
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "save"):
        tokenizer.tokenizer.save(str(output / "tokenizer.json"))
        return
    if hasattr(tokenizer, "save_to"):
        tokenizer.save_to(str(output / "tokenizer.json"))
        return
    if hasattr(tokenizer, "vocab_path"):
        vocab_path = Path(tokenizer.vocab_path)
        if vocab_path.is_file():
            (output / "tokenizer.json").write_bytes(vocab_path.read_bytes())
            return
    raise RuntimeError("Unable to write tokenizer.json from NeMo Parakeet tokenizer.")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
