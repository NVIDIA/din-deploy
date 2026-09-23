# Qwen3 ASR and forced alignment

ONNX Runtime inference on CPU or TensorRT RTX, with independent ASR and alignment APIs.

## Supported models

| Model | Hugging Face ID |
|---|---|
| ASR 0.6B | `Qwen/Qwen3-ASR-0.6B-hf` |
| ASR 1.7B | `Qwen/Qwen3-ASR-1.7B-hf` |
| Forced Aligner 0.6B | `Qwen/Qwen3-ForcedAligner-0.6B-hf` |

All models support BF16 (original precision, default), FP16 and FP32 exports.
Use FP32 for CPU inference.

## Supported capabilities

| Model / upstream toolkit capability | C++ sample |
|---|:---:|
| Offline, single stream | ✓ |
| Online / streaming, single utterance | ✓ |
| Batched inference | — |
| Long-form ASR | ✓ |
| Automatic language detection / language hint | ✓ |
| ASR: 30 languages and 22 Chinese dialects | ✓ |
| Standalone alignment / alignment of any ASR output | ✓ |
| Long-form alignment with timed transcript segments (C++ API) | ✓ |
| Long-form alignment of unsegmented text | — |
| Word timestamps: en, de, es, fr, it, pt, ru, ko | ✓ |
| Chinese / Cantonese character timestamps | ✓ |
| Automatic Japanese word timestamps | — |
| Character timestamps: all 11 alignment languages (sample extension) | ✓ |
| Caller-supplied alignment units | ✓ |

See [upstream language support](https://github.com/QwenLM/Qwen3-ASR).
Japanese alignment requires character mode or supplied units.

## Export and validate

Run commands from the repository root; install the export dependencies first.

```bash
pip install -r asr/qwen3/requirements.txt
python asr/qwen3/model_export/export_qwen3_asr.py --output models/qwen3-asr
python asr/qwen3/model_export/export_qwen3_asr.py --task aligner --output models/qwen3-aligner
```

HF downloads checkpoints automatically. Use `--size 1.7B` for the larger ASR model,
or `--dtype fp16` / `--dtype fp32` to change precision; use a separate output directory.
Each ASR export contains one encoder, one decoder and the shared log-mel graph.

```bash
python asr/qwen3/model_export/validate_qwen3_asr.py --onnx-dir models/qwen3-asr --audio audio.wav
python asr/qwen3/model_export/validate_qwen3_asr.py --task aligner --onnx-dir models/qwen3-aligner --audio audio.wav --transcript transcript.txt --language English
```

## Build and run

Follow the [repository build setup](../../README.md). Replace `<build>` and
`<build/bin>` with your build and executable directories; append `.exe` on Windows.

```text
cmake --build <build> --config Release --target din_asr_qwen3_cli din_asr_qwen3_aligner_cli
<build/bin>/din_asr_qwen3_cli audio.wav --model-dir models/qwen3-asr
<build/bin>/din_asr_qwen3_aligner_cli audio.wav --model-dir models/qwen3-aligner --transcript transcript.txt --lang-id en
```

Both CLIs accept `--provider cpu|trt-rtx` (default: trt-rtx).
Use `--granularity characters` for character alignment or `--units` for one supplied
alignment unit per transcript line. Character timestamps have 80 ms resolution.
Japanese word segmentation can be an optional external preprocessing pass supplied through `AlignUnits` / `--units`.

## Long-form and streaming

Long recordings are split automatically. `--max-new-tokens` defaults to 1024 per
chunk; `reached_eos=false` means the transcript is incomplete. Increase the budget
or reduce `--max-chunk-seconds`. Export-time `--cache-capacity` defaults to 8192 tokens.

Add `--stream` for streaming transcription; `- --stream` reads mono 16 kHz float32
PCM from stdin. Outputs are replacement hypotheses, not incremental text.
Streaming reprocesses the current utterance, so latency grows with its length;
start a new stream before exceeding the exported context capacity.

## C++ integration

Use `Qwen3Pipeline` ([qwen3.h](qwen3.h)) for transcription and
`Qwen3ForcedAligner` ([forced_aligner.h](forced_aligner.h)) for text from any ASR.
`AlignSegments` accepts timed transcript segments and returns recording-relative
timestamps. Alignment is limited to 180 seconds, 2048 units and 8192 context tokens
per segment; use `--max-chunk-seconds 175` when transcribing for subsequent alignment.
