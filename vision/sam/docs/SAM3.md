# SAM3.1 export and verification

The shared [SAM API](../README.md) loads fixed-attention contract v1 and corrected
memory-boundary v2. The preferred BF16 export directory is `D:/models/sam3.1_bf16`.
The CPU example below exports to `artifacts/sam3/cpu_fp32`. Source and checkpoint revisions are
pinned in [sam3_upstream.json](../sam3_upstream.json).

Install dependencies with `uv pip install -e ".[sam3]"`. Obtain access to the
checkpoint on Hugging Face and authenticate before preparing its source:

```powershell
python vision/sam/python/prepare_sam3.py
```

## Export

```powershell
python vision/sam/python/export_sam3_onnx.py --static-memory --output D:/models/sam3.1_bf16
python vision/sam/python/verify_sam_onnx.py D:/models/sam3.1_bf16
```

The exporter uses opset 23 native `Attention`, sixteen object slots, seven
spatial memories and sixteen pointer histories. It also exports tokenizer assets,
temporal constants, metadata and reference graph fixtures. ONNX wrappers,
tokenizer serialization, antialias resize and sequence lowering are required
parts of the exporter.

Exporter defaults are `--dtype bfloat16`, `--opset 23`, and
`--output artifacts/sam3/onnx-bf16`. All ten graphs are exported by default;
`--graphs` takes space-separated graph names. `--static-memory`,
`--cpu-compatible`, and `--fixtures-only` are off by default. The runtime
requires `--static-memory` for GPU bundles. `--cpu-compatible` requires
`--dtype float32` and enables static memory automatically. Export uses CUDA,
including when generating an FP32 CPU bundle; there is no `--device` option.
Without `--fixture-image`, the image reference input is zero-filled. Existing
output metadata must match the selected precision, source and checkpoint.

**BF16 is the default and preferred export precision** (`--dtype bfloat16`).
Model weights and reduced-precision execution use BF16 across all graphs.
FP32 is required for CPU inference and numerical references (`--dtype float32`).
FP16 (`--dtype float16`) comes with quality tradeoffs. The exporter preserves
native attention at the selected precision without inserting attention-only
casts. Preprocessing and numerically sensitive upstream operations can retain
FP32 arithmetic.

The current native runtime requires SAM3.1 `dtype=float32`; it does not yet
accept the full BF16 exports. The existing `D:/models/sam3_onnx` bundle is used
in the native GPU example below.

Vision input remains 1008 square, decoder logits 288, binary seed prompts 1152,
ordinary consolidated seed-memory logits 1008, and tracking-memory resizing
288 to 1152.

## CPU bundle and numerical verification

```powershell
python vision/sam/python/export_sam3_onnx.py --dtype float32 --cpu-compatible --output artifacts/sam3/cpu_fp32
python vision/sam/python/verify_sam_onnx.py artifacts/sam3/cpu_fp32 --fixtures --report artifacts/sam3/verification.json
din_sam_cli input.png output --model sam3 --model-dir artifacts/sam3/cpu_fp32 --provider cpu --text bicycle --dump-masks
```

The CPU export chunks attention queries for ORT's mask handling
and preserves FP32 arithmetic. `--fixtures` requires saved input/output arrays;
without it, verification checks ONNX structure and sidecars only.

## Native inference

```powershell
cmake --build out/build/windows-arm64 --config Release --target din_sam_cli
din_sam_cli input.png output --model sam3 --model-dir D:/models/sam3_onnx --text bicycle --dump-masks
```

The build command assumes the `windows-arm64` configure preset has already
been configured. Use `out/build/windows-x64` for the `windows-x64` preset.
The CLI defaults to `--provider trt-rtx --device 0 --threshold 0.5`;
`--threshold` controls detection confidence. The mask probability cutoff is
fixed at 0.5. The CLI requires both `--model` and `--model-dir`.

Use a frame directory with `--track` for causal video inference, or `--framewise`
for independent segmentation. Cache options and timing reports are shared with
SAM2; see [CLI usage](../README.md#command-line-inference). The current
controller initializes on the first qualifying detection and retains bounded
history; association, reconditioning and full native PyTorch video parity remain
separate work.
