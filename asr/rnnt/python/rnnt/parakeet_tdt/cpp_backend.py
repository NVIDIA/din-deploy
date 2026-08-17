# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rnnt.audio import Audio


class CppBackend:
    def __init__(
        self,
        *,
        model_dir: str | Path = Path("artifacts/parakeet/onnx"),
        provider: str = "cpu",
        ep_cache: str | Path = Path("artifacts/parakeet/trt_rtx_cache"),
        ep_context_dir: str | Path = Path("artifacts/parakeet/ep_context"),
    ) -> None:
        try:
            from rnnt._parakeet_tdt_cpp import Transcriber
        except ImportError as error:
            try:
                from _parakeet_tdt_cpp import Transcriber
            except ImportError:
                raise RuntimeError(
                    "Parakeet C++ bindings are not importable. Build `_parakeet_tdt_cpp` "
                    "with CMake and add the build output directory to PYTHONPATH."
                ) from error

        self.transcriber = Transcriber(
            str(model_dir),
            provider,
            str(ep_cache),
            str(ep_context_dir),
            "auto",
        )

    def transcribe_audio(self, audio: Audio) -> dict[str, Any]:
        samples = np.ascontiguousarray(audio.samples, dtype=np.float32)
        return dict(self.transcriber.transcribe(samples, int(audio.sampling_rate)))

    def transcribe_file(self, path: str | Path) -> dict[str, Any]:
        return dict(self.transcriber.transcribe_file(str(path)))
