# RNNT ASR

ONNX transcription pipelines for Parakeet TDT and Nemotron ASR.

## Supported models

| Model | Hugging Face ID | Checkpoint | ONNX export |
| --- | --- | --- | --- |
| Parakeet TDT 0.6B v3 | `nvidia/parakeet-tdt-0.6b-v3` | FP32 | FP16 (default) |
| Nemotron 3.5 ASR Streaming 0.6B | `nvidia/nemotron-3.5-asr-streaming-0.6b` | FP32 | FP16 (default) |

## Export

```bash
PYTHONPATH=asr/rnnt/python python asr/rnnt/python/tools/export_parakeet_tdt_onnx.py --help
PYTHONPATH=asr/rnnt/python python asr/rnnt/python/tools/export_nemotron_onnx.py --help
```

## Build

```bash
cmake --preset linux-x64 -DDIN_ENABLE_PARAKEET_PYTHON_BINDINGS=ON -DDIN_ENABLE_NEMOTRON_PYTHON_BINDINGS=ON
cmake --build out/build/linux-x64 --target din_asr_parakeet_tdt_cli din_asr_nemotron_cli _parakeet_tdt_cpp _nemotron_asr_cpp
```

## Run

```bash
out/build/linux-x64/bin/din_asr_parakeet_tdt_cli audio.wav --model-dir artifacts/parakeet/onnx --provider trt-rtx
out/build/linux-x64/bin/din_asr_nemotron_cli audio.wav --model-dir artifacts/nemotron/onnx --provider trt-rtx --lang-id en-US
```

