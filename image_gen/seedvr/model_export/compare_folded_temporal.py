#!/usr/bin/env python3
"""Compare the baseline and experimental folded SeedVR VAE encoder ONNX files."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "out" / "seedvr"
MODELS = (
    BASE / "experimental-unfolded" / "seedvr2_ema_vae_encoder.onnx",
    BASE / "experimental-folded-fp32" / "seedvr2_ema_vae_encoder.onnx",
)


def main() -> None:
    sessions = [ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for path in MODELS]
    rng = np.random.default_rng(1729)
    for height, width in ((32, 32), (128, 128), (512, 512)):
        image = rng.random((1, 3, height, width)).astype(np.float16)
        baseline, folded = [session.run(None, {"image": image})[0].astype(np.float32) for session in sessions]
        error = np.abs(baseline - folded)
        tolerance = 2e-3 + 2e-3 * np.abs(baseline)
        print(
            f"{width}x{height}: output={baseline.shape}, "
            f"max_abs={error.max():.6g}, mean_abs={error.mean():.6g}, "
            f"out_of_tolerance={np.count_nonzero(error > tolerance)}/{error.size}"
        )


if __name__ == "__main__":
    main()
