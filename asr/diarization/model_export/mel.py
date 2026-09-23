# SPDX-License-Identifier: Apache-2.0
import torch
from common.model_export.log_mel import dft_basis
from torch import nn
from torch.nn import functional as F


class LogMel(nn.Module):
    # Raw PCM includes 256 left/right STFT samples and one preemphasis sample.
    samples = 3039 * 160 + 512 + 1

    def __init__(self, featurizer):
        super().__init__()
        assert featurizer.n_fft == 512 and featurizer.hop_length == 160
        assert featurizer.log_zero_guard_type == "add" and featurizer.mag_power == 2
        self.preemph = featurizer.preemph
        self.guard = featurizer.log_zero_guard_value
        real, imag = dft_basis(512)
        window = F.pad(featurizer.window.float(), (56, 56))
        self.register_buffer("real", real * window)
        self.register_buffer("imag", imag * window)
        self.register_buffer("filters", featurizer.fb.squeeze(0).float())

    def forward(self, samples, sample_bounds, feature_length):
        x = samples[:, 1:] - self.preemph * samples[:, :-1]
        positions = torch.arange(1, self.samples, device=x.device)
        x = x * ((positions >= sample_bounds[0]) & (positions < sample_bounds[1]))
        frames = x.unfold(1, 512, 160)
        real, imag = frames @ self.real.T, frames @ self.imag.T
        features = ((real.square() + imag.square()) @ self.filters.T + self.guard).log()
        return features * (torch.arange(3040, device=x.device) < feature_length)[None, :, None]
