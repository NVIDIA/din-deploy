#!/usr/bin/env python3
"""Select affected model tests and fingerprint their ONNX export inputs."""

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
from pathlib import Path

SHARED_RUNTIME = [
    "CMakeLists.txt",
    "CMakePresets.json",
    "cmake/*",
    "common/*.cpp",
    "common/*.h",
    "common/*CMakeLists.txt",
    ".github/workflows/ci.yml",
    ".github/scripts/*",
    ".github/workflows/model-changes.yml",
]
EXPORTS = {
    "whisper": [
        "asr/whisper/model_export/export_whisper.py",
        "common/model_export/*.py",
        ".github/workflows/export-whisper-model.yml",
    ],
    "parakeet": [
        "asr/rnnt/python/tools/export_parakeet_tdt_onnx.py",
        "asr/rnnt/python/rnnt/parakeet_tdt/nemo_backend.py",
        "asr/rnnt/python/rnnt/parakeet_tdt/__init__.py",
    ],
    "nemotron": [
        "asr/rnnt/python/tools/export_nemotron_onnx.py",
        "asr/rnnt/python/rnnt/nemotron_asr/nemo_backend.py",
        "asr/rnnt/python/rnnt/nemotron_asr/__init__.py",
    ],
    "sam2": [
        "vision/sam2/python/export_sam2_onnx.py",
        "vision/sam2/python/modeling.py",
        ".github/workflows/export-sam2-model.yml",
    ],
}
for model in ("parakeet", "nemotron"):
    EXPORTS[model] += [
        "asr/rnnt/python/tools/nemo_preprocessor_export.py",
        "asr/rnnt/python/rnnt/__init__.py",
        "asr/rnnt/python/rnnt/audio.py",
        ".github/workflows/export-nvidia-asr-model.yml",
    ]
RUNTIME = {
    "whisper": ["asr/whisper/*", "assets/sample.wav", ".github/workflows/whisper-asr.yml"],
    "sam2": ["vision/sam2/*", "assets/sam2-*", ".github/workflows/sam2.yml"],
}
for model, package in (("parakeet", "parakeet_tdt"), ("nemotron", "nemotron_asr")):
    RUNTIME[model] = [
        f"asr/rnnt/cpp/{model}*",
        "asr/rnnt/cpp/asr*",
        "asr/rnnt/CMakeLists.txt",
        f"asr/rnnt/python/rnnt/{package}/*",
        "asr/rnnt/python/rnnt/validation/*",
        "asr/rnnt/python/tests/*",
        "assets/sample.wav",
        ".github/workflows/nvidia-asr.yml",
    ]


def affected(model, files):
    patterns = SHARED_RUNTIME + RUNTIME[model] + EXPORTS[model]
    return any(fnmatch.fnmatchcase(path, pattern) for path in files if not path.endswith(".md") for pattern in patterns)


def export_key(model, model_id, attention):
    # Index blob IDs are independent of checkout line endings and include deleted/renamed inputs.
    sources = subprocess.check_output(["git", "ls-files", "--stage", "--", *EXPORTS[model]])
    options = json.dumps([model, model_id, attention]).encode()
    return hashlib.sha256(sources + options).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["changes", "key"])
    parser.add_argument("--model", choices=EXPORTS)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--attention", default="")
    args = parser.parse_args()
    if args.mode == "key":
        if not args.model or not args.model_id:
            parser.error("key requires --model and --model-id")
        print(export_key(args.model, args.model_id, args.attention))
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    pr = event.get("pull_request")
    base = pr["base"]["sha"] if pr else event.get("before")
    files = None
    if base and subprocess.run(["git", "cat-file", "-e", f"{base}^{{commit}}"], capture_output=True).returncode == 0:
        if pr:
            base = subprocess.check_output(["git", "merge-base", base, "HEAD"], text=True).strip()
        files = (
            subprocess.check_output(["git", "diff", "--no-renames", "--name-only", "-z", base, "HEAD"])
            .decode()
            .split("\0")
        )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        for model in EXPORTS:
            changed = files is None or affected(model, files)
            output.write(f"{model}={str(changed).lower()}\n")


if __name__ == "__main__":
    main()
