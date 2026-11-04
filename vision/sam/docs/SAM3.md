# SAM3.1 export and verification

The shared [SAM API](../README.md) loads fixed-attention contract v1 and corrected
memory-boundary v2. The default GPU bundle is `D:/models/Sam3`; its FP32 CPU
counterpart is `D:/models/Sam3_cpu_fp32`. Source and checkpoint revisions are
pinned in [sam3_upstream.json](../sam3_upstream.json).

Install dependencies with `uv pip install -e ".[sam3]"`. Obtain access to the
checkpoint on Hugging Face and authenticate before preparing its source:

```powershell
python vision/sam/python/prepare_sam3.py
```

## Export

```powershell
python vision/sam/python/export_sam3_onnx.py --static-memory --output artifacts/sam3/onnx
python vision/sam/python/verify_sam_onnx.py artifacts/sam3/onnx
```

The exporter uses opset 23 native `Attention`, sixteen object slots, seven
spatial memories and sixteen pointer histories. It also exports tokenizer assets,
temporal constants, metadata and reference graph fixtures. ONNX wrappers,
tokenizer serialization, antialias resize and sequence lowering are required
parts of the exporter.

Vision input remains 1008 square, decoder logits 288, binary seed prompts 1152,
ordinary consolidated seed-memory logits 1008, and tracking-memory resizing
288 to 1152. SAM3 recurrent tensors retain FP32 storage.

## CPU bundle and numerical verification

```powershell
python vision/sam/python/export_sam3_onnx.py --output artifacts/sam3/raw_fp32
python vision/sam/python/prepare_static_sam.py sam3 artifacts/sam3/raw_fp32 artifacts/sam3/cpu_fp32 --cpu-compatible
python vision/sam/python/verify_sam_onnx.py artifacts/sam3/cpu_fp32 --fixtures --report artifacts/sam3/verification.json
```

The CPU compatibility conversion chunks attention queries for ORT's mask handling
and preserves FP32 arithmetic. `--fixtures` requires saved input/output arrays;
without it, verification checks ONNX structure and sidecars only.

## Optional BF16 attention export

```powershell
python vision/sam/python/cast_sam3_attention.py artifacts/sam3/raw_fp32 artifacts/sam3/attention_bf16
python vision/sam/python/prepare_static_sam.py sam3 artifacts/sam3/attention_bf16 artifacts/sam3/gpu_static
python vision/sam/python/verify_sam_onnx.py artifacts/sam3/gpu_static
```

This casts native attention internals while preserving graph I/O types and
boolean validity masks. It does not apply broad precision conversion or certify
equivalence to the existing deployed mixed-precision bundle. The current models
in `D:/models` are unchanged by source-tracking cleanup.

## Native inference

```powershell
cmake --build out/build/windows-x64 --config Release --target din_sam3_cli
din_sam3_cli input.png output --text bicycle
```

Use a frame directory with `--track` for causal video inference. The current
controller initializes on the first qualifying detection and retains bounded
history; association, reconditioning and full native PyTorch video parity remain
separate work.
