# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download a SAM3 checkpoint into the configured global Hugging Face cache."""

from __future__ import annotations

import argparse

from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import GatedRepoError

from sam3_common import add_checkpoint_arguments, checkpoint_options, checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_checkpoint_arguments(parser)
    args = parser.parse_args()
    print(f"Global Hugging Face cache: {HF_HUB_CACHE}", flush=True)
    try:
        checkpoint = checkpoint_path(**checkpoint_options(args), local_files_only=False)
    except GatedRepoError as error:
        raise SystemExit(
            f"Request access at https://huggingface.co/{args.checkpoint_repo} and authenticate with `hf auth login`, then retry."
        ) from error
    print(f"Checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
