# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from difflib import get_close_matches
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from rnnt.audio import Audio, load_audio, resample

DEFAULT_FLEURS_LANGUAGES = ("en_us", "de_de", "fr_fr", "es_419", "it_it")
LIBRISPEECH_LONG_DATASET = "distil-whisper/librispeech_long"
LIBRISPEECH_LONG_CONFIG = "clean"
DEFAULT_SPLIT = "validation"
DEFAULT_MAX_SAMPLES_PER_LANGUAGE = 50
AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}


@dataclass(frozen=True)
class ValidationItem:
    id: str
    audio: Audio
    text: str
    language: str = ""

    @property
    def duration(self) -> float:
        return self.audio.duration


def parse_languages(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_FLEURS_LANGUAGES)
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_max_samples(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"none", "all"}:
        return None
    count = int(normalized)
    return None if count <= 0 else count


def iter_manifest(path: Path, target_sampling_rate: int = 16_000) -> Iterator[ValidationItem]:
    root = path.parent
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            audio_path = Path(record["audio"])
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            yield ValidationItem(
                id=str(record.get("id", f"{path.stem}-{line_number}")),
                audio=load_audio(audio_path, target_sampling_rate),
                text=str(record.get("text", "")),
                language=str(record.get("language", "")),
            )


def iter_audio_dir(path: Path, target_sampling_rate: int = 16_000) -> Iterator[ValidationItem]:
    for audio_path in sorted(path.rglob("*")):
        if not audio_path.is_file() or audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        relative = audio_path.relative_to(path)
        language = relative.parts[0] if len(relative.parts) > 1 else ""
        text = read_audio_dir_groundtruth(path, audio_path)
        yield ValidationItem(
            id=relative.as_posix(),
            audio=load_audio(audio_path, target_sampling_rate),
            text=text,
            language=language,
        )


def read_audio_dir_groundtruth(root: Path, audio_path: Path) -> str:
    relative = audio_path.relative_to(root)
    candidates = [audio_path.with_suffix(".txt")]
    groundtruth_dir = None
    if len(relative.parts) > 1:
        language_root = root / relative.parts[0]
        groundtruth_dir = language_root / "groundtruth"
        candidates.append(groundtruth_dir / f"{audio_path.stem}.txt")

    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    if groundtruth_dir is not None and groundtruth_dir.is_dir():
        stems = {candidate.stem: candidate for candidate in groundtruth_dir.glob("*.txt")}
        matches = get_close_matches(audio_path.stem, stems.keys(), n=1, cutoff=0.85)
        if matches:
            return stems[matches[0]].read_text(encoding="utf-8").strip()
    return ""


def iter_fleurs(
    languages: Iterable[str],
    *,
    split: str = DEFAULT_SPLIT,
    max_samples_per_language: int | None = DEFAULT_MAX_SAMPLES_PER_LANGUAGE,
    target_sampling_rate: int = 16_000,
) -> Iterator[ValidationItem]:
    try:
        from datasets import Audio as HfAudio
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("FLEURS validation requires `pip install -e .[validation]`.") from error

    for language in languages:
        dataset = load_dataset("google/fleurs", language, split=split, streaming=True)
        dataset = dataset.cast_column("audio", HfAudio(decode=False))
        for index, record in enumerate(dataset):
            if max_samples_per_language is not None and index >= max_samples_per_language:
                break
            yield item_from_hf_record(
                record,
                language=language,
                fallback_id=f"{language}-{split}-{index}",
                target_sampling_rate=target_sampling_rate,
            )


def iter_librispeech_long(
    *,
    split: str = DEFAULT_SPLIT,
    max_samples: int | None = None,
    target_sampling_rate: int = 16_000,
) -> Iterator[ValidationItem]:
    try:
        from datasets import Audio as HfAudio
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("LibriSpeech long validation requires `pip install -e .[validation]`.") from error

    dataset = load_dataset(
        LIBRISPEECH_LONG_DATASET,
        LIBRISPEECH_LONG_CONFIG,
        split=split,
        streaming=True,
    )
    dataset = dataset.cast_column("audio", HfAudio(decode=False))
    for index, record in enumerate(dataset):
        if max_samples is not None and index >= max_samples:
            break
        yield item_from_hf_record(
            record,
            language="en",
            fallback_id=f"librispeech-long-{split}-{index}",
            target_sampling_rate=target_sampling_rate,
        )


def item_from_hf_record(
    record: dict[str, Any],
    *,
    language: str,
    fallback_id: str,
    target_sampling_rate: int = 16_000,
) -> ValidationItem:
    audio_record = record["audio"]
    audio = audio_from_hf_record(audio_record, target_sampling_rate)

    text = record.get("transcription", record.get("raw_transcription", ""))
    return ValidationItem(
        id=str(record.get("id", fallback_id)),
        audio=audio,
        text=str(text),
        language=language,
    )


def audio_from_hf_record(audio_record: dict[str, Any], target_sampling_rate: int) -> Audio:
    if "array" in audio_record and audio_record["array"] is not None:
        samples = np.asarray(audio_record["array"], dtype=np.float32)
        sampling_rate = int(audio_record["sampling_rate"])
        if sampling_rate != target_sampling_rate:
            samples = resample(samples, sampling_rate, target_sampling_rate)
            sampling_rate = target_sampling_rate
        return Audio(samples=samples, sampling_rate=sampling_rate)

    if audio_record.get("path") and Path(audio_record["path"]).is_file():
        return load_audio(audio_record["path"], target_sampling_rate)

    if audio_record.get("bytes"):
        samples, sampling_rate = sf.read(BytesIO(audio_record["bytes"]), dtype="float32", always_2d=False)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        if sampling_rate != target_sampling_rate:
            samples = resample(samples, sampling_rate, target_sampling_rate)
            sampling_rate = target_sampling_rate
        return Audio(np.asarray(samples, dtype=np.float32), sampling_rate)

    raise ValueError("HF audio record must contain array, path, or bytes.")


def maybe_group_long_form(
    items: Iterable[ValidationItem],
    target_seconds: float | None,
    *,
    silence_seconds: float = 0.25,
) -> Iterator[ValidationItem]:
    if target_seconds is None or target_seconds <= 0:
        yield from items
        return

    pending: list[ValidationItem] = []
    pending_seconds = 0.0
    pending_language: str | None = None

    for item in items:
        if pending and pending_language != item.language:
            yield combine_items(pending, silence_seconds)
            pending = []
            pending_seconds = 0.0

        pending.append(item)
        pending_language = item.language
        pending_seconds += item.duration

        if pending_seconds >= target_seconds:
            yield combine_items(pending, silence_seconds)
            pending = []
            pending_seconds = 0.0
            pending_language = None

    if pending:
        yield combine_items(pending, silence_seconds)


def combine_items(items: list[ValidationItem], silence_seconds: float = 0.25) -> ValidationItem:
    if not items:
        raise ValueError("Cannot combine an empty item list.")

    sampling_rate = items[0].audio.sampling_rate
    silence = np.zeros(max(0, round(silence_seconds * sampling_rate)), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for index, item in enumerate(items):
        if item.audio.sampling_rate != sampling_rate:
            raise ValueError("Long-form grouping requires matching sampling rates.")
        if index:
            pieces.append(silence)
        pieces.append(item.audio.samples)

    return ValidationItem(
        id="+".join(item.id for item in items),
        audio=Audio(np.concatenate(pieces).astype(np.float32, copy=False), sampling_rate),
        text=" ".join(item.text for item in items),
        language=items[0].language,
    )
