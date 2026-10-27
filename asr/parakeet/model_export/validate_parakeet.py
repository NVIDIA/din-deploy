# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "model_export"))
from export_parakeet import DEFAULT_MODEL_ID
from nemo_export import load_nemo_model
from validate_asr import validate, validation_parser


def reference(args):
    model = load_nemo_model(args.model, device=args.device)
    if args.local_window:
        model.change_attention_model("rel_pos_local_attn", [256, 256])
    result = model.transcribe([str(args.audio.resolve())], batch_size=1)[0]
    return str(getattr(result, "text", result)).strip()


if __name__ == "__main__":
    parser = validation_parser(DEFAULT_MODEL_ID)
    parser.add_argument("--local-window", action="store_true", help="Match an export made with --local-window.")
    validate(parser.parse_args(), reference)
