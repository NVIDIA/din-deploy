# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PixelDiT T2I sampling glue: prompt encoding, CFG, and flow DPM-Solver++ (2M).

Reproduces upstream t2i/inference.py with sample_steps/cfg_scale/flow_shift and
DPMS(model_type="flow", schedule="FLOW").sample(order=2, skip_type="time_uniform_flow",
method="multistep"). The text encoder and DiT are callables so the same code drives
PyTorch modules and ONNX Runtime sessions:

    encode(input_ids [1,S] int64, attention_mask [1,S] int64, select_index [300] int64) -> [1,300,D]
    dit(x [2,3,H,W] fp32, timestep [2] fp32 (t * 1000), y [2,300,D] fp32) -> velocity [2,3,H,W]

Batch index 0 is the negative (unconditional) prompt, 1 is the positive prompt.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

# t2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml text_encoder.chi_prompt
CHI_PROMPT = "\n".join(
    [
        'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    ]
)
DEFAULT_NEGATIVE_PROMPT = "low quality, worst quality, over-saturated, blurry, deformed, watermark"
TXT_MAX_LENGTH = 300
NUM_TRAIN_TIMESTEPS = 1000  # NoiseScheduleFlow.total_N; DiT timestep = t * 1000, t_0 = 1 / 1000

EncodeFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
DiTFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class SamplerConfig:
    steps: int = 50
    cfg_scale: float = 2.75
    flow_shift: float = 4.0
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT


@dataclass
class TextLayout:
    """Fixed text-encoder sequence layout; text_seq_len is 506 for the shipped chi_prompt."""

    chi_prompt: str
    chi_prompt_tokens: int
    txt_max_length: int = TXT_MAX_LENGTH
    text_seq_len: int = field(init=False)

    def __post_init__(self):
        # inference.py: num_chi_prompt_tokens + model_max_length - 2  ("magic number 2: [bos], [_]")
        self.text_seq_len = self.chi_prompt_tokens + self.txt_max_length - 2

    @classmethod
    def from_tokenizer(cls, tokenizer, chi_prompt: str = CHI_PROMPT, txt_max_length: int = TXT_MAX_LENGTH):
        return cls(chi_prompt, len(tokenizer.encode(chi_prompt)), txt_max_length)

    def positive_select_index(self) -> list[int]:
        # inference.py: [0] + list(range(-model_max_length + 1, 0))
        return [0] + list(range(self.text_seq_len - self.txt_max_length + 1, self.text_seq_len))

    def negative_select_index(self) -> list[int]:
        # Causal encoder + right padding: the first 300 positions of a 506-token encode equal a 300-token encode.
        return list(range(self.txt_max_length))


def tokenize(tokenizer, text: str, max_length: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    tok = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    return tok.input_ids.to(device), tok.attention_mask.to(device)


def encode_prompts(tokenizer, encode: EncodeFn, layout: TextLayout, prompt: str, negative_prompt: str, device):
    """Return (y_negative, y_positive), each [1, 300, D], using the single 506-token layout."""
    select_neg = torch.tensor(layout.negative_select_index(), dtype=torch.long, device=device)
    select_pos = torch.tensor(layout.positive_select_index(), dtype=torch.long, device=device)
    y_neg = encode(*tokenize(tokenizer, negative_prompt, layout.text_seq_len, device), select_neg)
    y_pos = encode(*tokenize(tokenizer, layout.chi_prompt + prompt.strip(), layout.text_seq_len, device), select_pos)
    return y_neg, y_pos


def initial_noise(height: int, width: int, seed: int, device="cuda") -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(1, 3, height, width, device=device, generator=generator)


def flow_timesteps(steps: int, flow_shift: float, device) -> torch.Tensor:
    """DPM_Solver.get_time_steps(skip_type="time_uniform_flow"); t goes from ~1 down to 0."""
    t_T, t_0 = 1, 1.0 / NUM_TRAIN_TIMESTEPS
    betas = torch.linspace(t_T, t_0, steps + 1).to(device)
    sigmas = 1.0 - betas
    return (flow_shift * sigmas / (1 + (flow_shift - 1) * sigmas)).flip(dims=[0])


# Flow noise schedule (NoiseScheduleFlow): alpha_t = 1 - t, sigma_t = t.
def _log_alpha(t):
    return torch.log(1 - t)


def _lambda(t):
    return _log_alpha(t) - torch.log(t)


class FlowDPMSolver:
    """DPM-Solver++ multistep (order 2, lower_order_final) with classifier-free guidance."""

    def __init__(self, dit: DiTFn, y_neg: torch.Tensor, y_pos: torch.Tensor, cfg_scale: float):
        self.dit = dit
        self.y = torch.cat([y_neg, y_pos])
        self.cfg_scale = cfg_scale

    def data_prediction(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x0 estimate at time t (model_wrapper classifier-free + DPM_Solver.data_prediction_fn)."""
        t_in = t.expand(1)
        t_in = torch.cat([t_in] * 2)
        x_in = torch.cat([x] * 2)
        out = self.dit(x_in, t_in * NUM_TRAIN_TIMESTEPS, self.y)
        sigma = t_in.view(-1, 1, 1, 1).to(x)
        noise_uncond, noise = ((1 - sigma) * out + x_in).chunk(2)
        noise = noise_uncond + self.cfg_scale * (noise - noise_uncond)
        return (x - t * noise) / (1 - t)

    @staticmethod
    def first_order_update(x, s, t, model_s):
        if t == 0:
            return model_s  # sigma_t = 0, alpha_t = 1, expm1(-inf) = -1: the update is exactly x0
        h = _lambda(t) - _lambda(s)
        alpha_t = torch.exp(_log_alpha(t))
        phi_1 = torch.expm1(-h)
        return t / s * x - alpha_t * phi_1 * model_s

    @staticmethod
    def second_order_update(x, model_prev_list, t_prev_list, t):
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = _lambda(t_prev_1), _lambda(t_prev_0), _lambda(t)
        alpha_t = torch.exp(_log_alpha(t))
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        phi_1 = torch.expm1(-h)
        return (t / t_prev_0) * x - (alpha_t * phi_1) * model_prev_0 - 0.5 * (alpha_t * phi_1) * D1_0

    def sample(self, x: torch.Tensor, steps: int, flow_shift: float, progress: Callable[[int], None] | None = None):
        order = 2
        assert steps >= order
        timesteps = flow_timesteps(steps, flow_shift, x.device)
        t_prev_list = [timesteps[0]]
        model_prev_list = [self.data_prediction(x, timesteps[0])]
        for step in range(1, steps + 1):
            t = timesteps[step]
            step_order = 1 if step < order else min(order, steps + 1 - step)
            if step_order == 1:
                x = self.first_order_update(x, t_prev_list[-1], t, model_prev_list[-1])
            else:
                x = self.second_order_update(x, model_prev_list, t_prev_list, t)
            if len(t_prev_list) < order:
                t_prev_list.append(t)
                model_prev_list.append(None)
            else:
                t_prev_list = t_prev_list[1:] + [t]
                model_prev_list = model_prev_list[1:] + [None]
            if step < steps:  # no model evaluation after the final step
                model_prev_list[-1] = self.data_prediction(x, t)
            if progress:
                progress(step)
        return x


def generate(
    tokenizer,
    encode: EncodeFn,
    dit: DiTFn,
    layout: TextLayout,
    prompt: str,
    height: int,
    width: int,
    seed: int,
    sampler: SamplerConfig,
    device="cuda",
    progress: Callable[[int], None] | None = None,
) -> torch.Tensor:
    """Full text-to-image run; returns the final sample [1,3,H,W] in [-1, 1] (fp32)."""
    with torch.inference_mode():
        y_neg, y_pos = encode_prompts(tokenizer, encode, layout, prompt, sampler.negative_prompt, device)
        z = initial_noise(height, width, seed, device)
        solver = FlowDPMSolver(dit, y_neg, y_pos, sampler.cfg_scale)
        return solver.sample(z, sampler.steps, sampler.flow_shift, progress)


def to_uint8_image(sample: torch.Tensor) -> np.ndarray:
    """torchvision save_image(normalize=True, value_range=(-1, 1)) -> HxWx3 uint8."""
    img = sample[0].float().clamp(-1, 1).add(1).div(2)
    img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8)
    return img.numpy()
