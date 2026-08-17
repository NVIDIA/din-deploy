# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "DEFAULT_MODEL_ID",
    "load_nemo_model",
    "transcribe_paths",
]


def __getattr__(name: str):
    if name in {"DEFAULT_MODEL_ID", "load_nemo_model", "transcribe_paths"}:
        from . import nemo_backend

        return getattr(nemo_backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
