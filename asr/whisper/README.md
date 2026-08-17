# Whisper

OpenAI Whisper transcription with ONNX Runtime and TensorRT RTX.

## Supported models

| Model | Hugging Face ID | Checkpoint | FP32 export | FP16 SDPA export | FP16 math export | Recommended export |
| --- | --- | --- | --- | --- | --- | --- |
| tiny | `openai/whisper-tiny` | FP32 | ✅ | ✅ | — | FP16 SDPA |
| base | `openai/whisper-base` | FP32 | ✅ | ⚠️ | ⚠️ | FP32 |
| small | `openai/whisper-small` | FP32 | ✅ | ⚠️ | ✅ | FP16 math |
| medium | `openai/whisper-medium` | FP32 | ✅ | ❌ | ❌ | FP32 |
| large-v3 | `openai/whisper-large-v3` | FP16 | ✅ | ✅ | — | FP16 SDPA |
| large-v3-turbo | `openai/whisper-large-v3-turbo` | FP16 | ✅ | ✅ | — | FP16 SDPA |

✅ matches the reference; ⚠️ is usable but differs from the reference; ❌ is
unusable; — was not tested.

> **Precision note:** exporting at the checkpoint's original precision is the
> reliable baseline. FP16 export is available for every model; use the
> recommended mode above and validate the exact deployment configuration on real
> audio. See the [precision and attention guidance](#precision-and-attention-guidance)
> for the test details, FP16 attention trade-offs.

Note that this was testted using TensorRT-RTX releases 1.6 and there are known accuracy issue with this fused attention operator. For older releases, use `--attention math` to export decomposed attention instead of fused SDPA.

## Export


```bash
python export_whisper.py --model openai/whisper-small --output D:/models/whisper-small-onnx-fp16
python export_whisper.py --model openai/whisper-medium --dtype fp32 --output D:/models/whisper-medium-onnx-fp32
python export_whisper.py --model openai/whisper-large-v3-turbo --output D:/models/whisper-large-v3-turbo-onnx-fp16
```

### Precision and attention guidance

Exporting at the checkpoint's original precision is the reliable baseline: the
FP32 ONNX exports matched the HuggingFace reference in real-audio validation for
every supported model. FP16 exports are also available for every FP32 checkpoint
and showed usable results for every tested variant except `medium`, which is not
currently reliable in FP16.

For FP16, try decomposed (unfused) attention has shown accuracy improvements:

```bash
python export_whisper.py --model openai/whisper-small --dtype fp16 --attention math --output D:/models/whisper-small-onnx-fp16-math
```

`--attention math` avoids fused SDPA which has shown to improve accuracy on `small` and `base` substantially.
Always run `validate_whisper.py` on representative audio for the exact model,
precision, attention mode, and TensorRT-RTX version you will ship.

## Verify

```bash
python validate_whisper.py --model openai/whisper-small --onnx-dir D:/models/whisper-small-onnx-fp16 --audio audio.mp3 --truth transcript.txt --provider trt-rtx --dtype fp16
```

`validate_whisper.py` compares the ONNX
transcription against HuggingFace's generation on example audio. To validate a locally built TensorRT-RTX provider instead
of the installed EP package, pass `--ep-lib`, plus `--ep-dll-dir` and `--trt-bin`
when its dependent DLLs are not on the loader path. Otherwise the installed python package will be used.


## Build

```powershell
cmake --build out\build\windows-x64 --target din_asr_whisper_cli
```

## Run

```powershell
out\build\windows-x64\bin\din_asr_whisper_cli.exe audio.mp3 --model-dir D:\models\whisper-small-onnx-fp16 --provider trt-rtx
out\build\windows-x64\bin\din_asr_whisper_cli.exe audio.mp3 --model-dir D:\models\whisper-medium-onnx-fp32 --provider trt-rtx --lang-id en
```
