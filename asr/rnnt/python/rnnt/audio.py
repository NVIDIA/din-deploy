# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray
    sampling_rate: int

    @property
    def duration(self) -> float:
        return float(len(self.samples) / self.sampling_rate)


def audio_duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return float(info.frames / info.samplerate)


def load_audio(path: str | Path, target_sampling_rate: int) -> Audio:
    samples, sampling_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sampling_rate != target_sampling_rate:
        samples = resample(samples, sampling_rate, target_sampling_rate)
        sampling_rate = target_sampling_rate
    return Audio(samples=np.asarray(samples, dtype=np.float32), sampling_rate=sampling_rate)


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sampling rates must be positive.")
    if len(samples) == 0 or source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    duration = len(samples) / source_rate
    source_times = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    target_len = max(1, round(duration * target_rate))
    target_times = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(target_times, source_times, samples).astype(np.float32)


def iter_audio_windows(
    audio: Audio,
    window_seconds: float,
    stride_seconds: float | None = None,
):
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")

    stride_seconds = stride_seconds or window_seconds
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive.")

    window = max(1, round(window_seconds * audio.sampling_rate))
    stride = max(1, round(stride_seconds * audio.sampling_rate))

    start = 0
    while start < len(audio.samples):
        end = min(start + window, len(audio.samples))
        offset = start / audio.sampling_rate
        yield offset, Audio(audio.samples[start:end], audio.sampling_rate)
        if end == len(audio.samples):
            break
        start += stride
