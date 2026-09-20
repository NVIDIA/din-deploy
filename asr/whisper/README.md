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
python export_whisper.py --model openai/whisper-small --output models/whisper-small-onnx-fp16
python export_whisper.py --model openai/whisper-medium --dtype fp32 --output models/whisper-medium-onnx-fp32
python export_whisper.py --model openai/whisper-large-v3-turbo --output models/whisper-large-v3-turbo-onnx-fp16
```

### Precision and attention guidance

Exporting at the checkpoint's original precision is the reliable baseline: the
FP32 ONNX exports matched the HuggingFace reference in real-audio validation for
every supported model. FP16 exports are also available for every FP32 checkpoint
and showed usable results for every tested variant except `medium`, which is not
currently reliable in FP16.

For FP16, try decomposed (unfused) attention has shown accuracy improvements:

```bash
python export_whisper.py --model openai/whisper-small --dtype fp16 --attention math --output models/whisper-small-onnx-fp16-math
```

`--attention math` avoids fused SDPA which has shown to improve accuracy on `small` and `base` substantially.
Always run `validate_whisper.py` on representative audio for the exact model,
precision, attention mode, and TensorRT-RTX version you will ship.

## Verify

```bash
python validate_whisper.py --model openai/whisper-small --onnx-dir models/whisper-small-onnx-fp16 --audio audio.mp3 --truth transcript.txt --provider trt-rtx --dtype fp16
```

`validate_whisper.py` compares the ONNX
transcription against HuggingFace's generation on example audio. To validate a locally built TensorRT-RTX provider instead
of the installed EP package, pass `--ep-lib`, plus `--ep-dll-dir` and `--trt-bin`
when its dependent DLLs are not on the loader path. Otherwise the installed python package will be used.


## Build

Replace `<build>` and `<build/bin>` with your build and executable directories.
On Windows, append `.exe` to the CLI name.

```text
cmake --build <build> --config Release --target din_asr_whisper_cli
```

## Run

```text
<build/bin>/din_asr_whisper_cli audio.mp3 --model-dir models/whisper-small-onnx-fp16 --provider trt-rtx
<build/bin>/din_asr_whisper_cli audio.mp3 --model-dir models/whisper-medium-onnx-fp32 --provider trt-rtx --lang-id en
```

### Long recordings

```text
<build/bin>/din_asr_whisper_cli audio.mp3 --model-dir models/whisper-large-v3-turbo-onnx-fp16 --timestamps json
```

The complete recording is transcribed automatically. Audio longer than 30 seconds
uses timestamp-driven long-form windows, following upstream Whisper.
`--no-context` disables previous-text conditioning.
`--prefill-block-size` defaults to 128; 0 disables bucketing.

Use `--repeat N` to benchmark repeated transcriptions in the same pipeline.

## DGX Spark performance

Measured on an NVIDIA DGX Spark (GB10) with `openai/whisper-large-v3-turbo` FP16,
ONNX Runtime 1.27.0 and TensorRT RTX 1.6.1.120. Full recordings from the
[ASR Leaderboard Longform dataset](https://huggingface.co/datasets/hf-audio/asr-leaderboard-longform),
`earnings21` test split: SiTime (`4385072.wav`), Hershey (`4385939.wav`), and
Yeti (`4385388.wav`). Throughput is × real time; higher is faster.

| Recording | Audio | CPU EP | TRT RTX |
| --- | ---: | ---: | ---: |
| SiTime | 43 min | 3.9× | 60.0× |
| Hershey | 51 min | 3.5× | 55.5× |
| Yeti | 72 min | 3.9× | 60.0× |
| **Overall** | **166 min** | **3.8×** | **58.5×** |

TRT RTX averages warm runs 2–3; CPU uses one pass with default threading.
Loading is excluded. Overall throughput is total audio / total inference time.
