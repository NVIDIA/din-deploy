# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Whisper/Qwen log-mel ONNX frontend; FP32 computation, configurable output frames."""

import math

import torch
from torch import nn
from torch.nn import functional as F

MEL_N_FFT = 400
MEL_HOP = 160


class LogMel(nn.Module):
    """Whisper log-mel spectrogram as an ONNX-exportable graph, so the front-end
    runs on GPU/TRT instead of the (slow) CPU implementation in asr/whisper/mel.cpp.

    samples [1, 480000] (fp32 PCM, 30 s @ 16 kHz) -> audio_features [1, n_mels, 3000].
    STFT is done as a matmul against precomputed DFT bases (Conv/MatMul-friendly for
    TensorRT-RTX), not torch.stft. Matches HF WhisperProcessor to ~1e-3 (matmul DFT
    vs FFT rounding). Compute is fp32; the output is cast to `out_dtype` so it can
    chain straight into the fp16/fp32 encoder's `audio_features` input.
    """

    def __init__(self, mel_fb, out_dtype, frames=3000):
        super().__init__()
        self.out_dtype = out_dtype
        self.frames = frames
        self.register_buffer("window", torch.hann_window(MEL_N_FFT))
        n_freq = MEL_N_FFT // 2 + 1
        k = torch.arange(n_freq, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(MEL_N_FFT, dtype=torch.float32).unsqueeze(0)
        angle = 2.0 * math.pi * k * n / MEL_N_FFT
        self.register_buffer("dft_real", torch.cos(angle))  # [n_freq, n_fft]
        self.register_buffer("dft_imag", -torch.sin(angle))
        self.register_buffer("mel_fb", torch.as_tensor(mel_fb, dtype=torch.float32))  # [n_mels, n_freq]

    def forward(self, samples):
        x = samples.float()
        # center reflect padding (torch.stft center=True) then frame [1, 3000, n_fft]
        x = F.pad(x, (MEL_N_FFT // 2, MEL_N_FFT // 2), mode="reflect")
        count = self.frames if self.frames is not None else samples.shape[1] // MEL_HOP
        frames = x.unfold(1, MEL_N_FFT, MEL_HOP)[:, :count, :] * self.window
        real = frames @ self.dft_real.T
        imag = frames @ self.dft_imag.T
        power = real * real + imag * imag  # |stft|^2, [1, 3000, n_freq]
        mel = power @ self.mel_fb.T  # [1, 3000, n_mels]
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=(1, 2), keepdim=True) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.transpose(1, 2).to(self.out_dtype)  # [1, n_mels, 3000]
