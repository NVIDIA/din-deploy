# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "validate_asr", Path(__file__).resolve().parents[2] / "common/model_export/validate_asr.py"
)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


@pytest.fixture
def args(tmp_path):
    audio = tmp_path / "recording.wav"
    audio.touch()
    cli = tmp_path / "asr cli"
    cli.touch()
    return validation.validation_parser("test-model").parse_args(
        ["--audio", str(audio), "--cli", str(cli), "--model-dir", str(tmp_path)]
    )


def cli_result(monkeypatch, text):
    def run(command, **kwargs):
        assert command[0].endswith("asr cli")
        assert "--json" in command
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(command, 0, json.dumps({"transcription": text}))

    monkeypatch.setattr(validation.subprocess, "run", run)


def test_reference_comparison_and_logging(args, monkeypatch, capsys):
    cli_result(monkeypatch, "Hello, WORLD!")

    def reference(_):
        print("reference model diagnostics")
        return "hello world"

    validation.validate(args, reference)
    out = capsys.readouterr()
    assert json.loads(out.out)["wer_against_nemo"] == 0
    assert "reference model diagnostics" in out.err


def test_truth_controls_threshold(args, monkeypatch, capsys):
    cli_result(monkeypatch, "hello world")
    args.truth = args.audio.with_suffix(".txt")
    args.truth.write_text("hello world", encoding="utf-8")
    args.max_wer = 0
    validation.validate(args, lambda _: "something different")
    report = json.loads(capsys.readouterr().out)
    assert report["wer_against_truth"] == 0
    assert report["wer_against_nemo"] == 1


def test_wer_threshold_failure(args, monkeypatch):
    cli_result(monkeypatch, "wrong")
    args.max_wer = 0.1
    with pytest.raises(SystemExit, match="exceeds"):
        validation.validate(args, lambda _: "correct")


def test_inference_failure_is_not_scored(args, monkeypatch):
    def run(*_, **__):
        raise subprocess.CalledProcessError(1, ["asr"])

    monkeypatch.setattr(validation.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        validation.validate(args, lambda _: pytest.fail("Must not load reference after CLI failure"))


@pytest.mark.parametrize("limit", [-1, float("nan"), float("inf")])
def test_invalid_threshold_rejected_before_inference(args, monkeypatch, limit):
    args.max_wer = limit
    monkeypatch.setattr(validation.subprocess, "run", lambda *_a, **_k: pytest.fail("Must not run inference"))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        validation.validate(args, lambda _: "unused")
