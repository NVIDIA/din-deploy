# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ["load_nemo_model", "transcribe_audio", "transcribe_paths"]


def __getattr__(name: str):
    if name in __all__:
        from . import nemo_backend

        return getattr(nemo_backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
