# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export tokenizer vocabulary, cleaning tables, and native parity fixtures."""

import argparse
import html
import html.entities
import json
from pathlib import Path

from sam3_common import load_upstream


def export_tokenizer(output: Path, tokenizer, context_length: int):
    import ftfy

    # ftfy's character-level compatibility fixes; NFC and lowercasing run through
    # ICU in C++. Do not use NFKC, which would also change unrelated characters.
    candidates = (
        list(range(256)) + list(range(0x2000, 0x2070)) + list(range(0xFB00, 0xFB07)) + list(range(0xFF00, 0xFFF0))
    )
    replacements = {chr(i): ftfy.fix_text(chr(i)) for i in candidates if ftfy.fix_text(chr(i)) != chr(i)}
    entities = dict(html.entities.html5)
    entities.update({f"#{i};": html.unescape(f"&#{i};") for i in range(161)})
    data = {
        "vocab": tokenizer.encoder,
        "merges": [list(pair) for pair, _ in sorted(tokenizer.bpe_ranks.items(), key=lambda item: item[1])],
        "context_length": context_length,
        "start_token": tokenizer.sot_token_id,
        "end_token": tokenizer.eot_token_id,
        "pad_token": 0,
        "normalization_map": replacements,
        "html_entities": entities,
        "ftfy_version": ftfy.__version__,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    prompts = [
        "bicycle",
        "A RED bicycle!",
        "  person\t in  blue\n",
        "dog's toys",
        "café",
        "cafe\u0301",
        "Straße",
        "日本語の猫",
        "a cat &amp; a dog",
        "a cat &amp;amp; a dog",
        "“yellow” bus",
        "ＦＵＬＬ width",
        "3 bicycles, 12 dogs",
        "<start_of_text> bicycle <end_of_text>",
        "cat " * 80,
        "",
        "🦊 fox",
        "&#x1f98a; fox",
        "caf&#233;",
        "&#128; 5",
        "&#0; object",
        "a&#x10ffff;b",
        "&amp;amp;amp;amp; dog",
        "&ldquo;cat&rdquo;",
        "ΟΣ Σίσυφος",
        "İstanbul",
        "ﬃ ligature",
        "a\u00a0b\u2003cat",
    ]
    fixtures = [{"text": text, "ids": tokenizer([text], context_length=context_length)[0].tolist()} for text in prompts]
    (output / "tokenizer_tests.json").write_text(json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sam3/onnx-fp32"))
    args = parser.parse_args()
    source = load_upstream()
    from sam3.model.tokenizer_ve import SimpleTokenizer

    export_tokenizer(args.output, SimpleTokenizer(str(source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")), 32)
