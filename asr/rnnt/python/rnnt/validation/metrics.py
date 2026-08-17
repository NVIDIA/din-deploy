# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from rnnt.audio import Audio


def normalize_text(text: str) -> str:
    chars = []
    for char in text.casefold():
        if unicodedata.category(char).startswith("P"):
            chars.append(" ")
        else:
            chars.append(char)
    return " ".join("".join(chars).split())


@dataclass
class WerResult:
    wer: float
    errors: int
    reference_words: int


@dataclass
class WerAccumulator:
    errors: int = 0
    reference_words: int = 0
    count: int = 0

    def add(self, reference: str, hypothesis: str) -> WerResult:
        result = compute_wer(reference, hypothesis)
        self.errors += result.errors
        self.reference_words += result.reference_words
        self.count += 1
        return result

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.reference_words

    def as_dict(self) -> dict[str, float | int]:
        return {
            "wer": self.wer,
            "errors": self.errors,
            "reference_words": self.reference_words,
            "count": self.count,
        }


@dataclass
class ValidationSummary:
    total: WerAccumulator = field(default_factory=WerAccumulator)
    by_language: dict[str, WerAccumulator] = field(default_factory=dict)
    audio_seconds: float = 0.0
    transcribe_seconds: float = 0.0
    unscored_count: int = 0

    def add(
        self,
        *,
        language: str,
        duration: float,
        transcribe_seconds: float,
        reference: str,
        hypothesis: str,
    ) -> WerResult:
        result = self.total.add(reference, hypothesis)
        self.by_language.setdefault(language or "unknown", WerAccumulator()).add(reference, hypothesis)
        self.audio_seconds += duration
        self.transcribe_seconds += transcribe_seconds
        return result

    def add_unscored(
        self,
        *,
        duration: float,
        transcribe_seconds: float,
    ) -> None:
        self.unscored_count += 1
        self.audio_seconds += duration
        self.transcribe_seconds += transcribe_seconds

    def as_dict(self) -> dict:
        rtf = self.transcribe_seconds / self.audio_seconds if self.audio_seconds else 0.0
        return {
            **self.total.as_dict(),
            "unscored_count": self.unscored_count,
            "audio_seconds": self.audio_seconds,
            "transcribe_seconds": self.transcribe_seconds,
            "rtf": rtf,
            "by_language": {
                language: accumulator.as_dict() for language, accumulator in sorted(self.by_language.items())
            },
        }


def compute_wer(reference: str, hypothesis: str) -> WerResult:
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    errors = edit_distance(reference_words, hypothesis_words)
    total = len(reference_words)
    wer = (0.0 if errors == 0 else 1.0) if total == 0 else errors / total
    return WerResult(wer=wer, errors=errors, reference_words=total)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def student_losses(
    teacher,
    student,
    audio: Audio,
    *,
    max_joint_steps: int = 256,
) -> dict[str, float]:
    import torch
    from torch.nn import functional as F

    torch.set_grad_enabled(False)
    teacher_states, teacher_mask = encode_audio(teacher, audio)
    student_states, student_mask = encode_audio(student, audio)

    common = min(teacher_states.shape[1], student_states.shape[1])
    teacher_states = teacher_states[:, :common]
    student_states = student_states[:, :common]
    if teacher_mask is not None and student_mask is not None:
        valid_mask = teacher_mask[:, :common] & student_mask[:, :common]
    else:
        valid_mask = torch.ones(teacher_states.shape[:2], device=teacher_states.device, dtype=torch.bool)

    encoder_mse = F.mse_loss(
        student_states[valid_mask].float(),
        teacher_states[valid_mask].to(student_states.device).float(),
    )
    joint_kl, steps = teacher_path_joint_kl(
        teacher,
        student,
        teacher_states,
        student_states,
        int(valid_mask[0].sum().item()),
        max_steps=max_joint_steps,
    )
    return {
        "encoder_mse": float(encoder_mse.item()),
        "joint_kl": float(joint_kl),
        "joint_steps": steps,
    }


def encode_audio(model, audio: Audio):
    import torch

    samples, lengths = audio_batch([audio], model.device)
    features, feature_lengths = model.preprocessor(input_signal=samples, length=lengths)
    compute_dtype = next(model.encoder.parameters()).dtype
    encoded, encoded_lengths = model.encoder(audio_signal=features.to(dtype=compute_dtype), length=feature_lengths)
    states = encoded.transpose(1, 2)
    mask = torch.arange(states.shape[1], device=states.device)[None] < encoded_lengths[:, None]
    return states, mask


def teacher_path_joint_kl(
    teacher,
    student,
    teacher_states,
    student_states,
    valid_length: int,
    *,
    max_steps: int,
) -> tuple[float, int]:
    import torch
    from torch.nn import functional as F

    teacher_state = initial_decoder_state(teacher, teacher_states.device)
    student_state = initial_decoder_state(student, student_states.device)
    time_idx = 0
    steps = 0
    kl_sum = 0.0

    while time_idx < valid_length and steps < max_steps:
        decoder_input = torch.tensor([[teacher_state.last_token]], device=teacher_states.device, dtype=torch.long)
        teacher_decoder, (teacher_hidden, teacher_cell) = teacher.decoder.predict(
            y=decoder_input,
            state=(teacher_state.hidden, teacher_state.cell),
            add_sos=False,
            batch_size=1,
        )
        teacher_logits = parakeet_joint_logits(teacher, teacher_states[:, time_idx : time_idx + 1], teacher_decoder)

        student_input = decoder_input.to(student_states.device)
        student_decoder, (student_hidden, student_cell) = student.decoder.predict(
            y=student_input,
            state=(student_state.hidden, student_state.cell),
            add_sos=False,
            batch_size=1,
        )
        student_logits = parakeet_joint_logits(student, student_states[:, time_idx : time_idx + 1], student_decoder)

        kl_sum += float(
            F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits.to(student_logits.device), dim=-1),
                reduction="batchmean",
            ).item()
        )

        vocab_size = blank_token_id(teacher) + 1
        token = int(teacher_logits[..., :vocab_size].argmax(dim=-1).item())
        duration_index = int(teacher_logits[..., vocab_size:].argmax(dim=-1).item())
        skip = int(durations(teacher)[duration_index])
        steps += 1

        if token != blank_token_id(teacher):
            teacher_state.last_token = token
            teacher_state.hidden = teacher_hidden
            teacher_state.cell = teacher_cell
            student_state.last_token = token
            student_state.hidden = student_hidden.float()
            student_state.cell = student_cell.float()

        time_idx += skip if skip else 1

    return (kl_sum / steps if steps else 0.0), steps


def audio_batch(audios: list[Audio], device):
    import torch

    max_len = max(len(audio.samples) for audio in audios)
    batch = torch.zeros(len(audios), max_len, device=device, dtype=torch.float32)
    lengths = torch.empty(len(audios), device=device, dtype=torch.long)
    for index, audio in enumerate(audios):
        samples = torch.from_numpy(audio.samples).to(device=device, dtype=torch.float32)
        batch[index, : samples.numel()] = samples
        lengths[index] = samples.numel()
    return batch, lengths


def initial_decoder_state(model, device):
    from types import SimpleNamespace

    import torch

    blank_token = blank_token_id(model)
    encoder_dim = joint_encoder_hidden(model)
    hidden, cell = model.decoder.initialize_state(torch.zeros(1, 1, encoder_dim, device=device))
    return SimpleNamespace(last_token=blank_token, hidden=hidden, cell=cell)


def blank_token_id(model) -> int:
    return int(model.cfg.decoder.vocab_size)


def durations(model) -> tuple[int, ...]:
    for value in (
        getattr(getattr(model.cfg, "model_defaults", None), "tdt_durations", None),
        getattr(getattr(model.cfg, "joint", None), "durations", None),
        getattr(model.cfg, "durations", None),
    ):
        if value is not None:
            return tuple(int(item) for item in value)
    return (0, 1, 2, 3, 4)


def joint_encoder_hidden(model) -> int:
    prednet = getattr(model.cfg.decoder, "prednet", model.cfg.decoder)
    return int(getattr(model.cfg.joint.jointnet, "encoder_hidden", prednet.pred_hidden))


def parakeet_joint_logits(model, encoder_states, decoder_output):
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
    return logits.float()
