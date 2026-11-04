# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the pinned checkpoint into the configured global Hugging Face cache."""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import GatedRepoError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path(__file__).parents[1] / "sam3_upstream.json")
    parser.add_argument("--source-root", type=Path, default=Path("artifacts/sam3/upstream"))
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    print(f"Global Hugging Face cache: {HF_HUB_CACHE}", flush=True)
    try:
        checkpoint = hf_hub_download(
            repo_id=lock["checkpoint_repo"],
            filename=lock["checkpoint_file"],
            revision=lock["checkpoint_revision"],
        )
    except GatedRepoError as error:
        raise SystemExit(
            f"Request access at https://huggingface.co/{lock['checkpoint_repo']} and authenticate with `hf auth login`, then retry."
        ) from error
    print(f"Checkpoint: {checkpoint}", flush=True)
    revision = lock["source_revision"]
    source = args.source_root / f"sam3-{revision}"
    if not (source / "sam3/model_builder.py").is_file():
        url = f"https://codeload.github.com/facebookresearch/sam3/zip/{revision}"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive_bytes = response.read()
        args.source_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            root = args.source_root.resolve()
            for member in archive.infolist():
                if not (root / member.filename).resolve().is_relative_to(root):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
            archive.extractall(root)
    print(f"Source: {source.resolve()}", flush=True)


if __name__ == "__main__":
    main()
