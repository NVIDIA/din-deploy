# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Export OpenAI Whisper to ONNX with raw torch.onnx (dynamo).

This produces the graph contract used by asr/whisper's C++
``WhisperPipeline`` (see asr/whisper/whisper.cpp). The native runtime uses the
shared dynamic decoder for bucketed history prefill and single-token decoding:

  <output>/
    encoder.onnx (+ encoder.onnx.data)
    decoder.onnx (+ decoder.onnx.data)
    vocab.json   (+ the rest of the tokenizer files)
    mel.onnx (log-mel preprocessor)

Usage:
    python export_whisper.py --model openai/whisper-medium --output D:/models/whisper-medium-onnx
    python export_whisper.py --model openai/whisper-small --only mel --output ./out
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.model_export.log_mel import LogMel as WhisperMel  # noqa: E402

EXT_SUFFIX = ".onnx.data"  # external data file sits next to <name>.onnx
DEFAULT_OPSET = 23

# Whisper log-mel preprocessor (matches asr/whisper/mel.h): 30 s @ 16 kHz.
MEL_N_FFT = 400
MEL_HOP = 160
MEL_SAMPLES = 480000
MEL_FRAMES = MEL_SAMPLES // MEL_HOP  # 3000

# --------------------------------------------------------------------------- #
# FP16 PRECISION STATUS -- read before trying to "fix" fp16 for medium/large  #
# --------------------------------------------------------------------------- #
# fp16 export works end-to-end (real audio, WER 0 vs HuggingFace) for tiny,
# base and small. medium (24 layers) and large-v3 (32 layers) produce GARBAGE
# in fp16 on BOTH the ORT CPU EP and the TensorRT-RTX EP. fp32 is correct at
# every size, so medium/large are shipped as fp32 (--dtype fp32).
#
# What was tried to make medium fp16 work (real audio unless noted):
#   1. do_constant_folding=False + _sanitize_fp16_initializers .... fixed a folded
#      weight that overflowed fp16 to +/-inf, but medium still garbage.
#   2. fp32 encoder compute, fp16 IO (EncoderExport) ............. required (fp16
#      encoder hidden_states diverged ~41 on a ~26 signal); fixed small, not medium.
#   3. fp32 LayerNorm + fp32 softmax + fp32 attention in decoder .. fixed small, not medium.
#   4. fp32-accumulate matmuls: Cast(fp32) around every fp16 Linear
#      (TRT precision-control trick) ............................. NO-OP for
#      correctness: ORT-CPU already fp32-accumulates, so the casts changed the
#      decoder logits by only 0.008, and medium fp16 was still wrong.
#
# Numerical evidence (medium, real audio, logits for the first prompt tokens):
#     fp32 decoder vs HF ............ argmax MATCH   [50259, 50359, 50364, 467]
#     fp16 (any variant) vs fp32 .... max_diff 13.86, argmax WRONG
#
# Diagnosis: the failure is fp16 ROUNDING of the residual stream / intermediate
# storage, compounding over 24-32 layers -- not weights and not matmul
# accumulation. Casting back to fp16 after each op re-injects that rounding, so
# per-op fp32 tricks cannot fix it; only fp32 compute (fp32 residual) does.
#
# TODO: to get TRUE fp16 for medium/large, either (a) achieve op-by-op
# fp16 parity with PyTorch's rounding, or (b) fp16-IO + fp32-compute (fp32
# residual stream, fp16 KV-cache/IO) to keep the C++ fp16 contract while staying
# correct. Until then medium/large are fp32, so the C++ runtime must bind fp32 IO
# for those models (or a fp16-IO/fp32-compute wrapper must be added).
# --------------------------------------------------------------------------- #


def _configure_stdio() -> None:
    """Windows consoles default to cp1252; the exporter prints Unicode. Force UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Attention helpers                                                           #
# --------------------------------------------------------------------------- #


def _split_heads(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """[1, S, E] -> [1, n_heads, S, head_dim]."""
    b, s, _ = x.shape
    return x.view(b, s, n_heads, head_dim).transpose(1, 2)


def _merge_heads(x: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """[1, n_heads, S, head_dim] -> [1, S, E]."""
    b, _, s, _ = x.shape
    return x.transpose(1, 2).contiguous().view(b, s, embed_dim)


def _fp32_layer_norm(module, x):
    """LayerNorm computed in fp32, matching PyTorch/HF (nn.LayerNorm always
    upcasts internally). The exported fp16 LayerNormalization op instead computes
    in fp16, whose variance (mean of squared, de-meaned activations) overflows
    once the residual stream grows in deep models (small/medium/large) -> the
    output degrades to garbage even though tiny/base survive. Casting inputs and
    the affine params to fp32 keeps the op accurate; output returns to x's dtype.
    """
    return torch.nn.functional.layer_norm(
        x.float(), (x.shape[-1],), module.weight.float(), module.bias.float(), module.eps
    ).to(x.dtype)


def _attend(q, k, v, scaling, bias, use_sdpa):
    """Scaled-dot-product attention.

    q: [1, H, Sq, Dh]  k/v: [1, H, Sk, Dh]  bias: [1, 1, 1, Sk] or None.
    With ``use_sdpa`` we call F.scaled_dot_product_attention (the dynamo exporter
    fuses this into the ONNX Attention op). Otherwise we spell it out as
    MatMul/Softmax/MatMul.

    Fused SDPA preserves the model dtype: FP16 models emit FP16 Q/K/V/mask
    inputs, while FP32 models emit FP32 inputs. TensorRT-RTX requires the
    additive mask to have the same element type as Q/K/V.
    """
    if use_sdpa:
        attn_mask = bias.to(q.dtype) if bias is not None else None
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scaling)
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scaling
    if bias is not None:
        scores = scores + bias.float()
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vf).to(q.dtype)


# --------------------------------------------------------------------------- #
# Encoder wrapper                                                             #
# --------------------------------------------------------------------------- #


class EncoderExport(nn.Module):
    """Audio tower + pre-projected cross-attention K/V for every decoder layer."""

    def __init__(self, model):
        super().__init__()
        cfg = model.config
        # Run the encoder in fp32 (fp16 IO). The deep fp16 encoder degrades badly
        # -- on whisper-small the fp16 encoder hidden_states diverged from fp32 by
        # ~41 on a ~26-magnitude signal (fp16 LayerNorm/softmax over 12+ layers),
        # producing garbage cross-KV and, downstream, garbage transcriptions. The
        # encoder is one-shot per 30 s chunk (not the decode loop), so fp32 here is
        # cheap; IO stays fp16 to match the C++ runtime contract.
        # NOTE: k_proj/v_proj here are the decoder layers' cross-attn projections,
        # but they are only ever used for this pre-projection (the decoder consumes
        # the already-projected past_*_cross), so casting them to fp32 is safe.
        self.encoder = model.model.encoder.float()
        self.n_heads = cfg.decoder_attention_heads
        self.head_dim = cfg.d_model // cfg.decoder_attention_heads
        self.cross_k = nn.ModuleList([layer.encoder_attn.k_proj for layer in model.model.decoder.layers]).float()
        self.cross_v = nn.ModuleList([layer.encoder_attn.v_proj for layer in model.model.decoder.layers]).float()

    def forward(self, audio_features):
        out_dtype = audio_features.dtype
        hidden = self.encoder(audio_features.float()).last_hidden_state  # fp32 [1, 1500, E]
        keys, values = [], []
        for k_proj, v_proj in zip(self.cross_k, self.cross_v):
            keys.append(_split_heads(k_proj(hidden), self.n_heads, self.head_dim).to(out_dtype))
            values.append(_split_heads(v_proj(hidden), self.n_heads, self.head_dim).to(out_dtype))
        # Order must match output_names: hidden, all keys, then all values (fp16 IO).
        return (hidden.to(out_dtype), *keys, *values)


# --------------------------------------------------------------------------- #
# Decoder wrapper                                                             #
# --------------------------------------------------------------------------- #


class DecoderExport(nn.Module):
    """Variable-length decode over a fixed-capacity self-KV cache.

    The same graph supports a multi-token prompt prefill and a one-token
    autoregressive decode step. ``write_indices`` contains one absolute cache
    position per input token. One decoder session switches between prefill
    buckets and single-token generation within its dynamic shape profile.
    """

    def __init__(self, model, use_sdpa: bool):
        super().__init__()
        cfg = model.config
        dec = model.model.decoder
        self.embed_tokens = dec.embed_tokens
        self.embed_positions = dec.embed_positions  # WhisperPositionalEmbedding ([max_target_positions, E])
        self.layers = dec.layers
        self.final_norm = dec.layer_norm
        self.proj_out = model.proj_out
        self.embed_dim = cfg.d_model
        self.n_heads = cfg.decoder_attention_heads
        self.head_dim = cfg.d_model // cfg.decoder_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.max_length = cfg.max_target_positions
        self.use_sdpa = use_sdpa

    def forward(self, input_ids, write_indices, nonpad_kv_seqlen, cache):
        dtype = self.layers[0].fc1.weight.dtype
        # Token + learned positional embedding for each absolute cache position.
        pos = self.embed_positions.weight.index_select(0, write_indices)  # [S, E]
        hidden = self.embed_tokens(input_ids.long()) + pos.unsqueeze(0)
        hidden = hidden.to(dtype)

        # Causal + key-padding bias. The single-token graph only needed the
        # non-padding condition; a multi-token prefill must additionally prevent
        # each query from attending to later tokens scattered by the same call.
        # Use a large *finite* negative, not -inf: the decoder attends one query
        # against a 448-slot cache where only the first few slots are valid, so
        # most keys are masked. A tiled/flash-attention kernel (TensorRT-RTX) takes
        # a per-tile row-max; for an all-masked tile that max is -inf and
        # `score - max = -inf - (-inf) = NaN`, poisoning the output. exp(-1e4)
        # underflows to 0 in both fp16 and fp32, so this masks just as hard.
        slots = torch.arange(self.max_length, device=hidden.device)
        populated = slots.unsqueeze(0) < nonpad_kv_seqlen  # [1, max_length]
        causal = slots.unsqueeze(0) <= write_indices.unsqueeze(1)  # [S, max_length]
        valid = populated & causal
        self_bias = torch.where(valid, 0.0, -1e4).unsqueeze(0).unsqueeze(0)

        present = []
        for i, layer in enumerate(self.layers):
            past_k_self = cache[4 * i + 0]
            past_v_self = cache[4 * i + 1]
            past_k_cross = cache[4 * i + 2]
            past_v_cross = cache[4 * i + 3]

            # --- masked self-attention with in-place cache scatter ---
            residual = hidden
            x = _fp32_layer_norm(layer.self_attn_layer_norm, hidden)
            attn = layer.self_attn
            q = _split_heads(attn.q_proj(x), self.n_heads, self.head_dim)
            new_k = _split_heads(attn.k_proj(x), self.n_heads, self.head_dim)  # [1, H, 1, Dh]
            new_v = _split_heads(attn.v_proj(x), self.n_heads, self.head_dim)
            # Scatter this token's K/V into the fixed-capacity cache at write_index.
            present_k = past_k_self.index_copy(2, write_indices, new_k)
            present_v = past_v_self.index_copy(2, write_indices, new_v)
            ctx = _attend(q, present_k, present_v, self.scaling, self_bias, self.use_sdpa)
            hidden = residual + attn.out_proj(_merge_heads(ctx, self.embed_dim))
            present.append(present_k)
            present.append(present_v)

            # --- cross-attention against the pre-projected encoder K/V ---
            residual = hidden
            x = _fp32_layer_norm(layer.encoder_attn_layer_norm, hidden)
            cattn = layer.encoder_attn
            q = _split_heads(cattn.q_proj(x), self.n_heads, self.head_dim)
            ctx = _attend(q, past_k_cross, past_v_cross, self.scaling, None, self.use_sdpa)
            hidden = residual + cattn.out_proj(_merge_heads(ctx, self.embed_dim))

            # --- feed forward ---
            residual = hidden
            x = _fp32_layer_norm(layer.final_layer_norm, hidden)
            x = layer.fc2(layer.activation_fn(layer.fc1(x)))
            hidden = residual + x

        hidden = _fp32_layer_norm(self.final_norm, hidden)
        # Only the final position is sampled; history buckets need KV updates.
        logits = self.proj_out(hidden[:, -1:, :])
        return (logits, *present)


# --------------------------------------------------------------------------- #
# Export driver                                                               #
# --------------------------------------------------------------------------- #


def _cast_attention_masks(onnx_path, onnx_dtype):
    """Post-export ONNX surgery: cast each fused ``Attention`` op's additive mask
    input (input #4) to ``onnx_dtype``.

    The dynamo exporter emits the key-padding mask as float32 even for an fp16
    model (the `0.0`/`-inf` literals default to float32). TensorRT-RTX's Myelin
    backend then rejects the masked-attention add with "Input 0's element type
    (half) differs from input 1's element type (float)". A single Cast on the
    mask makes the operands match. Returns the number of distinct masks fixed.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph
    new_nodes, casted = [], {}
    for node in graph.node:
        if node.op_type == "Attention" and len(node.input) >= 4 and node.input[3]:
            mask = node.input[3]
            if mask not in casted:
                cast_out = mask + "_io_cast"
                new_nodes.append(onnx.helper.make_node(
                    "Cast", [mask], [cast_out], to=onnx_dtype, name=mask + "_io_cast"))
                casted[mask] = cast_out
            node.input[3] = casted[mask]
        new_nodes.append(node)
    if casted:
        del graph.node[:]
        graph.node.extend(new_nodes)
        # External-data refs are untouched, so re-save just the proto (the
        # <name>.onnx.data file is left as-is — no expensive weight rewrite).
        onnx.save_model(model, str(onnx_path))
    return len(casted)


def _sanitize_fp16_initializers(onnx_path):
    """Replace non-finite values in fp16 weight initializers with the tensor's
    own finite range. Returns a list of (name, count) for what was fixed.

    WORKAROUND. With ``do_constant_folding=True`` the dynamo exporter folds some
    constant subexpressions in fp16, and a few *individual* elements overflow to
    +/-inf (observed on whisper-medium: `layers.13.fc2.bias` and a folded
    [1024,1024] weight, one inf each). The HF fp16 weights are all finite
    (absmax ~40) and the fp32 export has zero non-finite initializers, so this is
    purely an fp16 constant-fold overflow. A single inf weight poisons its MatMul
    -> NaN -> garbage logits, which is why medium/large produced "!!!!" in fp16
    while tiny/base (smaller folds) were fine.

    We clamp the offending elements into the tensor's finite magnitude (sign
    preserved; NaN -> 0). They are 1-in-1e6 elements, so the perturbation is
    negligible and it removes the inf that breaks the whole graph.

    TODO: this is a band-aid. Proper fixes to evaluate later:
      * export the affected constants/folds in fp32 (mixed precision), or
      * disable folding for the overflowing nodes (do_constant_folding=False
        regressed nothing here but wasn't fully validated), or
      * file/track the dynamo fp16 constant-fold overflow upstream.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))  # with external data (the inf lives in the .data file)
    data_name = Path(onnx_path).name + EXT_SUFFIX[len(".onnx"):]
    fixed = []
    for init in model.graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT16:
            continue
        arr = numpy_helper.to_array(init)
        f32 = arr.astype(np.float32)
        bad = ~np.isfinite(f32)
        if not bad.any():
            continue
        finite_absmax = float(np.max(np.abs(np.where(np.isfinite(f32), f32, 0.0))))
        fill = finite_absmax if finite_absmax > 0 else 0.0
        f32 = np.where(np.isnan(f32), 0.0, f32)
        f32 = np.where(np.isposinf(f32), fill, f32)
        f32 = np.where(np.isneginf(f32), -fill, f32)
        init.CopyFrom(numpy_helper.from_array(f32.astype(np.float16), init.name))
        fixed.append((init.name, int(bad.sum())))
    if fixed:
        onnx.save_model(model, str(onnx_path), save_as_external_data=True,
                        all_tensors_to_one_file=True, location=data_name)
    return fixed


def _onnx_export(model, args_tuple, output_path, input_names, output_names, opset, mask_cast_dtype=None,
                 dynamic_shapes=None):
    """Export to ONNX (dynamo) and rewrite weights as a single external-data file."""
    import onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        args_tuple,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        # Folding disabled: with fp16 it overflowed individual elements of folded
        # constants to +/-inf (e.g. whisper-medium `layers.13.fc2.bias` / a folded
        # [1024,1024] weight), poisoning the graph. HF's fp16 weights are all
        # finite, so this was purely a fold artifact. _sanitize_fp16_initializers
        # below stays as a belt-and-suspenders guard.
        do_constant_folding=False,
        dynamo=True,
        opset_version=opset,
        dynamic_shapes=dynamic_shapes,
    )
    # Consolidate weights into a single <name>.onnx.data file next to the model
    # (e.g. encoder.onnx -> encoder.onnx.data), matching the C++ runtime layout.
    data_name = output_path.name + EXT_SUFFIX[len(".onnx"):]  # "encoder.onnx" -> "encoder.onnx.data"
    onnx_model = onnx.load(str(output_path))
    onnx.save_model(
        onnx_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
    )
    print(f"  saved {output_path} (opset={opset})")
    # Fix fp16 constant-fold overflow on the written file (see fn docstring).
    sanitized = _sanitize_fp16_initializers(output_path)
    if sanitized:
        print(f"  workaround: clamped non-finite fp16 initializers {sanitized}")
    if mask_cast_dtype is not None:
        fixed = _cast_attention_masks(output_path, mask_cast_dtype)
        if fixed:
            print(f"  surgery: cast {fixed} Attention mask(s) to the model dtype (TRT-RTX fix)")


def _layer_names(prefix: str, n_layers: int):
    return [f"{prefix}_{i}" for i in range(n_layers)]


def _mask_cast_dtype(dtype):
    """ONNX dtype to cast fused-Attention masks to, or None when no fix is needed.

    Only fp16 exports hit the float32-mask vs fp16-scores mismatch (see
    _cast_attention_masks). For an fp32 model the mask is already float32, so the
    surgery is unnecessary and we skip it.
    """
    import onnx

    return onnx.TensorProto.FLOAT16 if dtype == torch.float16 else None


def export_mel(model_id, out_dir, dtype, opset):
    import numpy as np
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor.from_pretrained(model_id)
    mel_fb = np.asarray(fe.mel_filters)
    if mel_fb.shape[0] != fe.feature_size:  # HF stores [n_freq, n_mels]; we want [n_mels, n_freq]
        mel_fb = mel_fb.T
    wrapper = WhisperMel(mel_fb, dtype).eval()
    samples = torch.zeros(1, MEL_SAMPLES)  # fp32 PCM
    print(f"[mel] n_mels={fe.feature_size}, samples [1, {MEL_SAMPLES}] -> audio_features [1, {fe.feature_size}, {MEL_FRAMES}]")
    with torch.inference_mode():
        _onnx_export(wrapper, (samples,), out_dir / "mel.onnx", ["samples"], ["audio_features"], opset)


def export_encoder(model, out_dir, device, dtype, n_mels, mel_frames, opset):
    cfg = model.config
    n_layers = cfg.decoder_layers
    wrapper = EncoderExport(model).to(device).eval()

    audio = torch.zeros(1, n_mels, mel_frames, dtype=dtype, device=device)
    output_names = (
        ["hidden_states"]
        + _layer_names("present_key_cross", n_layers)
        + _layer_names("present_value_cross", n_layers)
    )
    print(f"[encoder] {n_layers} layers, audio_features [1, {n_mels}, {mel_frames}]")
    with torch.inference_mode():
        _onnx_export(wrapper, (audio,), out_dir / "encoder.onnx", ["audio_features"], output_names, opset,
                     mask_cast_dtype=_mask_cast_dtype(dtype))


def export_decoder(model, out_dir, device, dtype, enc_frames, use_sdpa, opset):
    cfg = model.config
    max_length = cfg.max_target_positions
    n_layers, n_heads = cfg.decoder_layers, cfg.decoder_attention_heads
    head_dim = cfg.d_model // n_heads
    self_shape = (1, n_heads, max_length, head_dim)
    cross_shape = (1, n_heads, enc_frames, head_dim)
    input_ids = torch.zeros(1, 4, dtype=torch.int32, device=device)
    write_indices = torch.arange(4, dtype=torch.int64, device=device)
    nonpad = torch.tensor([4], dtype=torch.int64, device=device)
    cache, inputs = [], ["input_ids", "write_indices", "nonpad_kv_seqlen"]
    for i in range(n_layers):
        cache += [torch.zeros(shape, dtype=dtype, device=device)
                  for shape in (self_shape, self_shape, cross_shape, cross_shape)]
        inputs += [f"past_key_self_{i}", f"past_value_self_{i}",
                   f"past_key_cross_{i}", f"past_value_cross_{i}"]
    outputs = ["logits"] + [name for i in range(n_layers)
                            for name in (f"present_key_self_{i}", f"present_value_self_{i}")]
    wrapper = DecoderExport(model, use_sdpa).to(device).eval()
    export_args = (input_ids, write_indices, nonpad, cache)
    sequence = torch.export.Dim("sequence_length", min=1, max=min(220, max_length))
    shapes = torch.export.ShapesCollection()
    shapes[input_ids] = {1: sequence}
    shapes[write_indices] = {0: sequence}
    print(f"[decoder] {n_layers} layers, self cache {self_shape}, cross cache {cross_shape}, dynamic sequence")
    with torch.inference_mode():
        _onnx_export(wrapper, export_args, out_dir / "decoder.onnx", inputs, outputs, opset,
                     mask_cast_dtype=_mask_cast_dtype(dtype),
                     dynamic_shapes=shapes.dynamic_shapes(wrapper, export_args))


def _save_tokenizer(model_id, out_dir):
    import json

    from transformers import GenerationConfig, WhisperTokenizer

    tok = WhisperTokenizer.from_pretrained(model_id)
    tok.save_pretrained(out_dir)
    GenerationConfig.from_pretrained(model_id).save_pretrained(out_dir)
    vocab = out_dir / "vocab.json"
    if not vocab.exists():
        # Transformers 5 writes a unified tokenizer.json for Whisper instead of
        # the legacy GPT-2 vocab.json. The C++ decoder consumes the same token-to-id
        # mapping, so materialize it from the tokenizer API when needed.
        vocab.write_text(
            json.dumps(tok.get_vocab(), ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"  saved tokenizer ({vocab.name} + friends)")


def load_model(model_id, device, dtype, use_sdpa):
    from transformers import WhisperForConditionalGeneration

    # Eager attention => the encoder's self-attention also exports as decomposed
    # MatMul/Softmax (no fused ONNX Attention op); sdpa lets it fuse.
    impl = "sdpa" if use_sdpa else "eager"
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, attn_implementation=impl, torch_dtype=dtype
    )
    return model.to(device).eval()

def _verify(out_dir, which):
    import onnx

    path = out_dir / f"{which}.onnx"
    m = onnx.load(str(path), load_external_data=False)

    def shape(value):
        return [dimension.dim_param or dimension.dim_value for dimension in value.type.tensor_type.shape.dim]

    ins = [(value.name, shape(value)) for value in m.graph.input]
    outs = [(value.name, shape(value)) for value in m.graph.output]
    fused = [n.op_type for n in m.graph.node if n.op_type == "Attention"]
    print(f"  [{which}] {len(ins)} inputs, {len(outs)} outputs, "
          f"opsets={[(o.domain, o.version) for o in m.opset_import]}, "
          f"fused Attention nodes={len(fused)}")
    if which == "decoder":
        inputs, outputs = dict(ins), dict(outs)
        sequence = inputs["input_ids"][1]
        if not isinstance(sequence, str) or inputs["write_indices"][0] != sequence or outputs["logits"][1] != 1:
            raise RuntimeError("decoder must support dynamic tokens and return last-position logits")
    return ins, outs


def main():
    _configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="openai/whisper-medium",
                        help="HF model id or local path (default: openai/whisper-medium).")
    parser.add_argument("--output", type=Path, default=Path("D:/models/whisper-medium-onnx"),
                        help="Output directory for encoder.onnx / decoder.onnx / vocab.json.")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                        help="Model and cache precision (default: fp16).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device used to trace the export.")
    parser.add_argument("--attention", choices=["math", "sdpa"], default="sdpa",
                        help="Attention export for both encoder and decoder.")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--only", choices=["encoder", "decoder", "mel"], default=None,
                        help="Export only one graph (default: encoder + decoder + mel).")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    use_sdpa = args.attention == "sdpa"
    device = torch.device(args.device)
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # The mel preprocessor only needs the feature extractor, not the model weights.
    if args.only in (None, "encoder", "decoder"):
        print(f"Loading {args.model} ({dtype}, attn={args.attention}) on {device} ...")
        model = load_model(args.model, device, dtype, use_sdpa)
        cfg = model.config
        print(f"  d_model={cfg.d_model} layers={cfg.decoder_layers} heads={cfg.decoder_attention_heads} "
              f"mels={cfg.num_mel_bins} vocab={cfg.vocab_size}")

    if args.only in (None, "encoder"):
        # TODO also accept opset, blocked by TRT support for attention dimensions
        export_encoder(model, out_dir, device, dtype, cfg.num_mel_bins, 3000, 22)
    if args.only in (None, "decoder"):
        export_decoder(model, out_dir, device, dtype, 1500, use_sdpa, args.opset)
    if args.only in (None, "mel"):
        export_mel(args.model, out_dir, dtype, args.opset)
    if args.only is None:
        _save_tokenizer(args.model, out_dir)

    if args.only in (None, "encoder"):
        _verify(out_dir, "encoder")
    if args.only in (None, "decoder"):
        _verify(out_dir, "decoder")
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
