# Nemotron 3.5 ASR

NeMo → standard ONNX → C++ inference through ONNX Runtime (CPU or TensorRT RTX).

## Export

Python 3.12+ is needed only for export and reference validation.

```bash
python -m pip install -r asr/nemotron/requirements.txt
python asr/nemotron/model_export/export_nemotron.py --output artifacts/nemotron/onnx --device cpu --external-data
```

## Build and run

```bash
cmake --preset linux-x64
cmake --build --preset linux-x64-release --target din_asr_nemotron_cli
out/build/linux-x64/bin/Release/din_asr_nemotron_cli audio.wav --model-dir artifacts/nemotron/onnx --provider cpu --lang-id en-US
```

Use `--json` for a JSON transcript, or `--timestamps token` for token timestamps.
The existing C++ target names and exported artifact formats are preserved.

## Validate

Compare one real recording against NeMo; add `--truth transcript.txt` to measure WER
against a human transcript, and `--max-wer 0.1` to enforce a threshold.

```bash
python asr/nemotron/model_export/validate_nemotron.py --audio audio.wav \
  --cli out/build/linux-x64/bin/Release/din_asr_nemotron_cli \
  --model-dir artifacts/nemotron/onnx --lang-id en-US
```

The Python inference package, native bindings, dataset evaluation, plotting, and
quantization tools have been removed. Run inference with the C++ CLI; prepare
models with the scripts above. Shared NeMo export and validation helpers live in
`common/model_export/`; shared C++ ASR utilities live in `asr/common/`.
