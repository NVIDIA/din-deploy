# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "model_export"))
from export_nemotron import DEFAULT_MODEL_ID, configure_model
from nemo_export import load_nemo_model
from validate_asr import validate, validation_parser


def reference(args):
    from nemo.collections.asr.models.rnnt_bpe_models_prompt import RNNTPromptTranscribeConfig

    metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
    model = load_nemo_model(args.model, device=args.device)
    configure_model(model, args.lang_id, tuple(metadata["att_context_size"]))
    config = RNNTPromptTranscribeConfig(batch_size=1, verbose=False, target_lang=args.lang_id)
    config.use_lhotse = False
    result = model.transcribe([str(args.audio.resolve())], override_config=config)[0]
    text = str(getattr(result, "text", result)).strip()
    return re.sub(r"<[^>\s]+>", "", text).strip() if metadata.get("strip_lang_tags", True) else text


if __name__ == "__main__":
    parser = validation_parser(DEFAULT_MODEL_ID)
    parser.add_argument("--lang-id", default="auto", help="Use the same target language for NeMo and C++.")
    args = parser.parse_args()
    validate(args, reference, ["--lang-id", args.lang_id])
