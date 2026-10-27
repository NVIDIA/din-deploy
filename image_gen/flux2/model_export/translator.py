# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment architecture for gafr1/translator; shared by export and verification."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_TRANSLATOR_REPO = "gafr1/translator"
DEFAULT_TRANSLATOR_FILE = "mlp_d3072_b4_e2.pt"


class ResMLP(nn.Module):
    def __init__(self, width, expansion):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * expansion)
        self.fc2 = nn.Linear(width * expansion, width)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class MLPTranslator(nn.Module):
    def __init__(self, in_dim, width, blocks, expansion):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([ResMLP(width, expansion) for _ in range(blocks)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 7680)

    def forward(self, x):
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))


class TranslatorEncoder(nn.Module):
    def __init__(self, qwen, translator, layers):
        super().__init__()
        self.qwen = qwen
        self.translator = translator
        self.layers = layers

    def forward(self, input_ids, attention_mask):
        output = self.qwen(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True, use_cache=False)
        features = torch.cat([output.hidden_states[i] for i in self.layers], dim=-1)
        return self.translator(features).float()


def add_translator_arguments(parser):
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-0.6B", help="HF repository or local Qwen3-0.6B snapshot")
    parser.add_argument("--translator-checkpoint", type=Path,
                        help="Local checkpoint; takes precedence over HF download")
    parser.add_argument("--translator-repo", default=DEFAULT_TRANSLATOR_REPO)
    parser.add_argument("--translator-file", default=DEFAULT_TRANSLATOR_FILE)
    parser.add_argument("--translator-revision", default="main",
                        help="HF revision; pin a commit for reproducible exports")


def load_translator_encoder(args, device):
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel

    checkpoint = args.translator_checkpoint
    if checkpoint is None:
        checkpoint = Path(hf_hub_download(
            args.translator_repo, args.translator_file, revision=args.translator_revision,
            local_files_only=args.local_files_only))
    # Checkpoints contain only state tensors plus plain config/scalar dictionaries.
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = data["cfg"]
    if cfg["kind"] != "mlp" or cfg["layers"] != "q4" or data["in_dim"] != 4096:
        raise ValueError("Expected a Qwen3-0.6B q4 MLP translator with input dimension 4096")
    net = MLPTranslator(data["in_dim"], cfg["d"], cfg["nblocks"], cfg["expansion"])
    net.load_state_dict(data["model"], strict=True)
    qwen = AutoModel.from_pretrained(
        args.qwen_model, torch_dtype=torch.bfloat16, local_files_only=args.local_files_only,
        attn_implementation="eager")
    if qwen.config.hidden_size != 1024 or qwen.config.num_hidden_layers != 28:
        raise ValueError("Translator requires Qwen3-0.6B (28 layers, width 1024)")
    return TranslatorEncoder(qwen, net, (7, 14, 21, 28)).eval().to(device=device, dtype=torch.bfloat16)
