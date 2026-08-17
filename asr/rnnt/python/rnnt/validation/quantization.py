# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Iterable, Iterator
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rnnt.audio import Audio

from .data import DEFAULT_SPLIT, ValidationItem, iter_audio_dir, iter_fleurs, iter_manifest, parse_languages


@dataclass(frozen=True)
class QuantModel:
    model: torch.nn.Module
    sample_rate: int
    score_step: Any


@dataclass(frozen=True)
class JointLossOutput:
    token_logits: torch.Tensor
    token_targets: torch.Tensor
    duration_logits: torch.Tensor | None = None
    duration_targets: torch.Tensor | None = None


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    log(f"loading {args.model} candidate model on {args.device}")
    quant_model = load_quant_model(args)
    log(f"loading calibration data from {calibration_source(args)}")
    calib_items = list(load_calibration_items(args, quant_model.sample_rate))
    if not calib_items:
        raise ValueError("No calibration audio found.")
    if len(calib_items) < args.calib_samples:
        raise ValueError(
            f"Requested {args.calib_samples} calibration samples, but only found {len(calib_items)} "
            f"from {calibration_source(args)}."
        )
    log(f"loaded {len(calib_items)} calibration samples")

    info = auto_quantize_encoder(
        quant_model.model,
        calib_items,
        quant_model.score_step,
        precision=args.precision,
        effective_bits=args.effective_bits,
        calib_steps=args.calib_steps,
        score_steps=args.score_steps,
        verbose=args.verbose,
        summary_path=args.output / "quant_summary.txt",
        state_path=args.output / "auto_quant_state.json",
        modelopt_path=args.output / "auto_quantize_model.pt",
    )
    (args.output / "quantization.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


def auto_quantize_encoder(
    model,
    calib_items: list[ValidationItem],
    score_step,
    *,
    precision: str,
    effective_bits: float,
    calib_steps: int,
    score_steps: int,
    verbose: bool,
    summary_path: Path,
    state_path: Path,
    modelopt_path: Path,
) -> dict[str, Any]:
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq

    def forward_step(quant_model, data: ValidationItem) -> JointLossOutput:
        return score_step(quant_model, data)

    def loss_func(output: JointLossOutput, data: ValidationItem) -> torch.Tensor:
        loss = torch.nn.functional.cross_entropy(output.token_logits.float(), output.token_targets)
        if output.duration_logits is not None and output.duration_targets is not None:
            loss = loss + torch.nn.functional.cross_entropy(output.duration_logits.float(), output.duration_targets)
        log(f"loss: {loss.item()}")
        return loss

    model.eval()
    log(f"starting ModelOpt auto_quantize ({precision}, effective_bits={effective_bits})")
    model, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": effective_bits},
        quantization_formats=[quantization_format(mtq, precision)],
        data_loader=calib_items,
        forward_step=forward_step,
        loss_func=loss_func,
        disabled_layers=["feature_extractor*", "preprocessor*", "decoder*", "joint*"],
        num_calib_steps=min(calib_steps, len(calib_items)),
        num_score_steps=min(score_steps, len(calib_items)),
        verbose=verbose,
    )
    log(f"saving searched model to {modelopt_path}")
    mto.save(model, modelopt_path)
    log(f"writing quantization summary to {summary_path}")
    print_and_store_quant_summary(mtq, model, summary_path, verbose=verbose)
    state_path.write_text(json.dumps(jsonable(search_state), indent=2), encoding="utf-8")
    return {
        "mode": "auto-quantize-encoder",
        "precision": precision,
        "effective_bits": effective_bits,
        "calib_steps": min(calib_steps, len(calib_items)),
        "score_steps": min(score_steps, len(calib_items)),
        "sample_count": len(calib_items),
        "modelopt_model_path": str(modelopt_path),
        "auto_quant_state_path": str(state_path),
        "quant_summary_path": str(summary_path),
    }


def quantization_format(mtq, precision: str) -> dict[str, Any]:
    if precision == "int4":
        return mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG
    if precision == "fp8":
        return mtq.FP8_DEFAULT_CFG
    raise ValueError(f"Unsupported precision: {precision}")


def restore_modelopt_quantized_model(
    model,
    modelopt_path: Path,
    *,
    summary_path: Path | None = None,
) -> tuple[object, dict[str, Any]]:
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq

    model = mto.restore(model, modelopt_path)
    model.eval()
    print_and_store_quant_summary(mtq, model, summary_path, verbose=False)
    return model, {
        "mode": "modelopt-restore",
        "modelopt_model_path": str(modelopt_path),
        "quant_summary_path": str(summary_path) if summary_path is not None else None,
    }


def load_quant_model(args: argparse.Namespace) -> QuantModel:
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if args.model == "parakeet":
        from rnnt.parakeet_tdt.nemo_backend import DEFAULT_MODEL_ID, MODEL_DIR, load_nemo_model

        model_id = args.model_id or DEFAULT_MODEL_ID
        model_dir = args.model_dir or MODEL_DIR
        log(f"loading Parakeet candidate from {model_id}")
        del model_dir
        model = load_nemo_model(
            model_id,
            device=args.device,
            dtype=dtype,
        )
        model.eval()
        return QuantModel(model, int(model.cfg.preprocessor.sample_rate), parakeet_joint_ce_step)

    from rnnt.nemotron_asr.nemo_backend import DEFAULT_ATT_CONTEXT_SIZE, DEFAULT_MODEL_ID, load_nemo_model, prompt_id

    model_id = args.model_id or DEFAULT_MODEL_ID
    log(f"loading Nemotron candidate from {model_id}")
    model = load_nemo_model(
        model_id,
        device=args.device,
        target_lang=args.target_lang,
        att_context_size=DEFAULT_ATT_CONTEXT_SIZE,
        dtype=dtype,
    )
    model.eval()
    model._din_prompt_id = prompt_id(model, args.target_lang)
    return QuantModel(model, int(model.cfg.preprocessor.sample_rate), nemotron_joint_ce_step)


def parakeet_encoder_step(model, item: ValidationItem) -> torch.Tensor:
    samples, lengths = audio_batch([item.audio], model.device)
    features, feature_lengths = model.preprocessor(input_signal=samples, length=lengths)
    compute_dtype = next(model.encoder.parameters()).dtype
    encoded, encoded_lengths = model.encoder(audio_signal=features.to(dtype=compute_dtype), length=feature_lengths)
    states = encoded.transpose(1, 2)
    encoder_mask = torch.arange(states.shape[1], device=states.device)[None] < encoded_lengths[:, None]
    if encoder_mask is not None:
        states = states[:, : int(encoder_mask[0].sum().item())]
    return states


def parakeet_joint_ce_step(model, item: ValidationItem, max_steps: int = 256) -> JointLossOutput:
    states = parakeet_encoder_step(model, item)
    valid_length = min(states.shape[1], max_steps)
    if valid_length <= 0:
        raise ValueError(f"Calibration sample {item.id} produced no encoder frames.")
    state = initial_decoder_state(model, states.device)
    token_logits = []
    token_targets = []
    duration_logits = []
    duration_targets = []
    vocab_size = blank_token_id(model) + 1

    for time_idx in range(valid_length):
        decoder_input = torch.tensor([[state.last_token]], device=states.device, dtype=torch.long)
        with torch.no_grad():
            decoder_state, (next_hidden, next_cell) = model.decoder.predict(
                y=decoder_input,
                state=(state.hidden, state.cell),
                add_sos=False,
                batch_size=1,
            )
            decoder_state = decoder_state.detach()
        logits = parakeet_joint_logits(model, states[:, time_idx : time_idx + 1], decoder_state).squeeze(0).squeeze(0)
        token_logit = logits[:vocab_size]
        duration_logit = logits[vocab_size:]
        token_target = token_logit.detach().argmax(dim=-1)
        duration_target = duration_logit.detach().argmax(dim=-1)

        token_logits.append(token_logit)
        token_targets.append(token_target)
        duration_logits.append(duration_logit)
        duration_targets.append(duration_target)

        token = int(token_target.item())
        if token != blank_token_id(model):
            state.last_token = token
            state.hidden = next_hidden.detach()
            state.cell = next_cell.detach()

    return JointLossOutput(
        token_logits=torch.stack(token_logits),
        token_targets=torch.stack(token_targets),
        duration_logits=torch.stack(duration_logits),
        duration_targets=torch.stack(duration_targets),
    )


def parakeet_joint_logits(model, encoder_states: torch.Tensor, decoder_output: torch.Tensor) -> torch.Tensor:
    joint = model.joint
    if (
        hasattr(joint, "project_encoder")
        and hasattr(joint, "project_prednet")
        and hasattr(joint, "joint_after_projection")
    ):
        enc = joint.project_encoder(encoder_states)
        pred = joint.project_prednet(decoder_output)
        logits = joint.joint_after_projection(enc, pred)
    else:
        logits = joint(encoder_states, decoder_output)
    if logits.dim() == 4 and logits.shape[2] == 1:
        logits = logits.squeeze(2)
    return logits


def initial_decoder_state(model, device):
    from types import SimpleNamespace

    hidden, cell = model.decoder.initialize_state(torch.zeros(1, 1, joint_encoder_hidden(model), device=device))
    return SimpleNamespace(last_token=blank_token_id(model), hidden=hidden, cell=cell)


def blank_token_id(model) -> int:
    return int(model.cfg.decoder.vocab_size)


def joint_encoder_hidden(model) -> int:
    prednet = getattr(model.cfg.decoder, "prednet", model.cfg.decoder)
    return int(getattr(model.cfg.joint.jointnet, "encoder_hidden", prednet.pred_hidden))


def nemotron_encoder_step(model, item: ValidationItem) -> torch.Tensor:
    samples, lengths = audio_batch([item.audio], model.device)
    features, feature_lengths = model.preprocessor(input_signal=samples, length=lengths)
    compute_dtype = next(model.encoder.parameters()).dtype
    encoded, encoded_lengths = model.encoder(audio_signal=features.to(dtype=compute_dtype), length=feature_lengths)
    states = encoded.transpose(1, 2)
    prompt = torch.zeros(states.shape[0], states.shape[1], 128, device=states.device, dtype=states.dtype)
    prompt[:, :, int(model._din_prompt_id)] = 1.0
    states = model.prompt_kernel(torch.cat([states, prompt], dim=-1))
    return states[:, : int(encoded_lengths[0].item())]


def nemotron_joint_ce_step(model, item: ValidationItem, max_steps: int = 256) -> JointLossOutput:
    states = nemotron_encoder_step(model, item)
    valid_length = min(states.shape[1], max_steps)
    if valid_length <= 0:
        raise ValueError(f"Calibration sample {item.id} produced no encoder frames.")
    token_logits = []
    token_targets = []

    with torch.no_grad():
        decoder_out, state = model.decoder.predict(y=None, state=None, batch_size=1)
        decoder_out = decoder_out[:, -1:, :].detach()

    for time_idx in range(valid_length):
        enc = model.joint.project_encoder(states[:, time_idx : time_idx + 1])
        pred = model.joint.project_prednet(decoder_out)
        logits = model.joint.joint_after_projection(enc, pred).squeeze(0).squeeze(0).squeeze(0)
        token_target = logits.detach().argmax(dim=-1)
        token_logits.append(logits)
        token_targets.append(token_target)

        token = int(token_target.item())
        if token != int(model.cfg.decoder.vocab_size):
            token_tensor = torch.tensor([[token]], device=states.device, dtype=torch.long)
            with torch.no_grad():
                decoder_out, state = model.decoder.predict(y=token_tensor, state=state, add_sos=False)
                decoder_out = decoder_out[:, -1:, :].detach()

    return JointLossOutput(
        token_logits=torch.stack(token_logits),
        token_targets=torch.stack(token_targets),
    )


def audio_batch(audios: list[Audio], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(audio.samples) for audio in audios)
    batch = torch.zeros(len(audios), max_len, device=device, dtype=torch.float32)
    lengths = torch.empty(len(audios), device=device, dtype=torch.long)
    for index, audio in enumerate(audios):
        samples = torch.from_numpy(audio.samples).to(device=device, dtype=torch.float32)
        batch[index, : samples.numel()] = samples
        lengths[index] = samples.numel()
    return batch, lengths


def load_calibration_items(args: argparse.Namespace, sample_rate: int) -> Iterator[ValidationItem]:
    if args.manifest is not None:
        log(f"reading up to {args.calib_samples} samples from manifest {args.manifest}")
        items: Iterable[ValidationItem] = iter_manifest(args.manifest, target_sampling_rate=sample_rate)
    elif args.audio_dir is not None:
        log(f"reading up to {args.calib_samples} samples from audio directory {args.audio_dir}")
        items = iter_audio_dir(args.audio_dir, target_sampling_rate=sample_rate)
    else:
        yield from iter_fleurs_round_robin(args, sample_rate)
        return

    for index, item in enumerate(items):
        if index >= args.calib_samples:
            break
        log(f"loaded calibration sample {index + 1}/{args.calib_samples}: {item.id}")
        yield item


def iter_fleurs_round_robin(args: argparse.Namespace, sample_rate: int) -> Iterator[ValidationItem]:
    languages = parse_languages(args.languages)
    if not languages:
        return
    per_language = max(1, (args.calib_samples + len(languages) - 1) // len(languages))
    log(
        f"streaming up to {args.calib_samples} samples from FLEURS split={args.split} "
        f"round-robin languages={','.join(languages)}"
    )
    iterators = [
        iter(
            iter_fleurs(
                [language],
                split=args.split,
                max_samples_per_language=per_language,
                target_sampling_rate=sample_rate,
            )
        )
        for language in languages
    ]

    yielded = 0
    while iterators and yielded < args.calib_samples:
        remaining = []
        for iterator in iterators:
            if yielded >= args.calib_samples:
                break
            try:
                item = next(iterator)
            except StopIteration:
                continue
            yielded += 1
            log(f"loaded calibration sample {yielded}/{args.calib_samples}: {item.id}")
            yield item
            remaining.append(iterator)
        iterators = remaining


def calibration_source(args: argparse.Namespace) -> str:
    if args.manifest is not None:
        return f"manifest {args.manifest}"
    if args.audio_dir is not None:
        return f"audio directory {args.audio_dir}"
    return f"FLEURS split {args.split} languages={args.languages}"


def print_and_store_quant_summary(mtq, model, summary_path: Path | None, *, verbose: bool) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        mtq.print_quant_summary(model)
    summary = buffer.getvalue()
    if verbose and summary:
        print(summary, end="")
    if summary_path is not None:
        summary_path.write_text(summary, encoding="utf-8")
    return summary


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoQuantize the Parakeet or Nemotron encoder with ModelOpt.")
    parser.add_argument("--model", choices=("parakeet", "nemotron"), required=True)
    parser.add_argument("--precision", choices=("int4", "fp8"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--effective-bits", type=float, default=None)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-steps", type=int, default=16)
    parser.add_argument("--score-steps", type=int, default=16)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--target-lang", default="auto")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path)
    source.add_argument("--audio-dir", type=Path)
    parser.add_argument("--languages", default="en_us,it_it,es_419,de_de")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--samples-per-language", type=int, default=128)

    args = parser.parse_args()
    if args.effective_bits is None:
        args.effective_bits = 8.0 if args.precision == "fp8" else 4.8
    if args.calib_samples <= 0:
        raise ValueError("--calib-samples must be positive.")
    if args.calib_steps <= 0 or args.score_steps <= 0:
        raise ValueError("--calib-steps and --score-steps must be positive.")
    return args


def log(message: str) -> None:
    print(f"[quantization] {message}", flush=True)


if __name__ == "__main__":
    main()
