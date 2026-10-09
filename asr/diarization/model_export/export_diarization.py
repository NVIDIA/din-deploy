# SPDX-License-Identifier: Apache-2.0
"""Export Nemotron-3 diarization with fixed shapes and persistent speaker state."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import onnx
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from model import DiarizationStep, load_model  # noqa: E402

MODEL_ID = "nvidia/Nemotron-3-Diarization"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    args = parser.parse_args()
    checkpoint = args.checkpoint or hf_hub_download(MODEL_ID, "Nemotron-3-Diarization.nemo")
    model = load_model(checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    step = DiarizationStep(model).eval()
    step.model.to(dtype=dtype)
    with torch.inference_mode():
        torch.onnx.export(
            step,
            step.example_inputs(),
            args.output / "diarization.onnx",
            dynamo=True,
            opset_version=23,
            external_data=True,
            input_names=step.input_names,
            output_names=step.output_names,
        )
    onnx.checker.check_model(str(args.output / "diarization.onnx"))
    digest = hashlib.sha256()
    for path in sorted(args.output.glob("diarization.onnx*")):
        with path.open("rb") as file:
            while block := file.read(1024 * 1024):
                digest.update(block)
    metadata = {
        "model_id": MODEL_ID,
        "format_version": 3,
        "dtype": args.dtype,
        "sample_rate": 16000,
        "chunk_frames": 340,
        "right_context": 40,
        "subsampling": 8,
        "mel_bins": 128,
        "cache_frames": 264,
        "fifo_frames": 40,
        "hidden_size": 512,
        "speakers": 8,
        "frame_seconds": 0.01,
        "graph_hash": digest.hexdigest()[:16],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
