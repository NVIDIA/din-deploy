# SPDX-License-Identifier: Apache-2.0
"""Export Qwen3-ASR with explicit tensor interfaces; checkpoint BF16 is the default."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import MethodType

import onnx
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from transformers import (
    AttentionInterface,
    AutoProcessor,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRForTokenClassification,
)
from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.model_export.kv_cache import FixedKVCache  # noqa: E402
from common.model_export.log_mel import LogMel as WhisperMel  # noqa: E402


class LogMel(WhisperMel):
    def __init__(self, processor):
        fe = processor.feature_extractor
        if (fe.n_fft, fe.hop_length, fe.feature_size, fe.sampling_rate, fe.dither) != (400, 160, 128, 16000, 0):
            raise ValueError("Native frontend requires the standard Qwen3 16 kHz configuration")
        super().__init__(fe.mel_filters.T.copy(), torch.float32, frames=None)


def save_native_assets(processor, output, task):
    tokenizer = processor.tokenizer

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    data = {
        "audio_start": encode(processor.audio_bos_token),
        "audio_end": encode(processor.audio_eos_token),
    }
    if task == "asr":
        prefixes = {}
        suffixes = {}
        languages = {}
        for language in [None, *LANGUAGE_CODE_TO_NAME.values()]:
            messages = [{"role": "user", "content": [{"type": "audio"}]}]
            rendered = tokenizer.apply_chat_template(
                messages, chat_template=processor.chat_template, tokenize=False, add_generation_prompt=True
            )
            prefix, suffix = rendered.split(processor.audio_token)
            prefixes[language or "auto"] = encode(prefix)
            suffixes[language or "auto"] = encode(suffix + (f"language {language}<asr_text>" if language else ""))
            languages[language or "auto"] = language or ""
        for code, language in LANGUAGE_CODE_TO_NAME.items():
            prefixes[code] = prefixes[language]
            suffixes[code] = suffixes[language]
            languages[code] = language
        data["prefixes"] = prefixes
        data["suffixes"] = suffixes
        data["languages"] = languages
        data["suffix"] = suffixes["auto"]
    (output / "native.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def decode_attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    if query.dtype == torch.float32:
        return export_attention(module, query, key, value, attention_mask, scaling, **kwargs)
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    return output.transpose(1, 2), None


def attention(q, k, v, bias, scale):
    scores = q @ k.transpose(-1, -2) * scale + bias
    return torch.softmax(scores.float(), dim=-1).to(v.dtype) @ v


def audio_attention(module, hidden_states, attention_bias, **kwargs):
    q, k, v = [
        p(hidden_states).reshape(1, -1, module.num_heads, module.head_dim).transpose(1, 2)
        for p in (module.q_proj, module.k_proj, module.v_proj)
    ]
    y = attention(q, k, v, attention_bias, module.scaling)
    return module.out_proj(y.transpose(1, 2).flatten(2))


class AudioEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        if model.config.audio_config.n_window != 50 or model.config.audio_config.num_mel_bins != 128:
            raise ValueError("This export contract requires n_window=50 and 128 mel bins")
        self.encoder = model.model.audio_tower
        self.projector = model.model.multi_modal_projector
        # Replace only HF's data-dependent attention splits; keep its encoder layers.
        for layer in self.encoder.layers:
            layer.self_attn.forward = MethodType(audio_attention, layer.self_attn)

    def forward(self, mel_chunks, valid_indices, attention_bias):
        enc = self.encoder
        x = F.gelu(enc.conv2d1(mel_chunks.unsqueeze(1)))
        x = F.gelu(enc.conv2d2(x))
        x = F.gelu(enc.conv2d3(x))
        x = x.permute(0, 3, 1, 2).flatten(2)
        x = enc.conv_out(x)
        x = x + enc.positional_embedding.positional_embedding[: x.shape[1]].to(x.dtype)
        x = x.flatten(0, 1).index_select(0, valid_indices).unsqueeze(0)
        for layer in enc.layers:
            x = layer(x, cu_seqlens=None, attention_bias=attention_bias)[0]
        return self.projector(enc.ln_post(x)).squeeze(0)


def export_attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    # Fold query groups into the query axis. TensorRT RTX otherwise lowers GQA
    # to full-size KV-head replication. This is ordinary MHA with identical math
    # and the original smaller KV tensors; only the small query/mask is repeated.
    batch, heads, sequence, width = query.shape
    kv_heads = key.shape[1]
    groups = heads // kv_heads
    query = query.reshape(batch, kv_heads, groups * sequence, width)
    if attention_mask is not None:
        attention_mask = attention_mask.repeat(1, 1, groups, 1)
    # TensorRT RTX 1.6 has no fused FP32 Attention kernel for this graph.
    if query.dtype == torch.float32:
        output = attention(query, key, value, attention_mask, scaling)
    else:
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=scaling)
    output = output.reshape(batch, heads, sequence, width)
    return output.transpose(1, 2), None


AttentionInterface.register("qwen3_onnx", export_attention)


class TextBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        if model.config.text_config.rope_parameters["rope_type"] != "default":
            raise ValueError("Only default RoPE is supported")
        self.decoder = model.model.language_model
        self.decoder.config._attn_implementation = "qwen3_onnx"

    def hidden(self, input_ids, audio_embeddings, audio_mask, position_ids, attention_bias, *past):
        dec = self.decoder
        x = torch.where(audio_mask, audio_embeddings, dec.embed_tokens(input_ids))
        rotary = dec.rotary_emb(x, position_ids)
        cache = FixedKVCache(past, position_ids, tensor_scatter=True) if past else None
        for layer in dec.layers:
            x = layer(x, attention_mask=attention_bias, position_embeddings=rotary, past_key_values=cache)
        return dec.norm(x), cache.present if cache else []


class TextDecoder(TextBackbone):
    """Prompt prefill and single-token decoding with fixed-capacity KV tensors."""

    def __init__(self, model):
        super().__init__(model)
        self.lm_head = model.lm_head
        AttentionInterface.register("qwen3_onnx_decode", decode_attention)
        self.decoder.config._attn_implementation = "qwen3_onnx_decode"

    def forward(self, input_ids, audio_embeddings, audio_mask, position_ids, attention_bias, logits_index, *past):
        hidden, present = self.hidden(input_ids, audio_embeddings, audio_mask, position_ids, attention_bias, *past)
        logits = self.lm_head(hidden.index_select(1, logits_index).squeeze(1))
        return logits, logits.argmax(-1), *present


class ForcedAligner(TextBackbone):
    """Predict all timestamp slots together, without an autoregressive cache."""

    def __init__(self, model):
        super().__init__(model)
        self.score = model.score

    def forward(self, input_ids, audio_embeddings, audio_mask, position_ids, attention_bias, timestamp_indices):
        hidden, _ = self.hidden(input_ids, audio_embeddings, audio_mask, position_ids, attention_bias)
        logits = self.score(hidden.index_select(1, timestamp_indices))
        return logits, logits.argmax(-1)


def export(module, args, path, names, outputs, shapes):
    print(f"Exporting {path.name}", flush=True)
    path.unlink(missing_ok=True)
    path.with_suffix(".onnx.data").unlink(missing_ok=True)
    torch.onnx.export(
        module.eval(),
        args,
        str(path),
        input_names=names,
        output_names=outputs,
        dynamo=True,
        dynamic_shapes=shapes,
        opset_version=24,
        external_data=True,
    )
    onnx.checker.check_model(str(path))
    print(f"Checked {path}", flush=True)


def save_metadata(output, metadata):
    metadata["graphs"] = {}
    for name in ("mel", "encoder", "decoder" if metadata["task"] == "asr" else "aligner"):
        if not (output / (name + ".onnx")).is_file():
            continue
        digest = hashlib.sha256()
        for suffix in (".onnx", ".onnx.data"):
            path = output / (name + suffix)
            if path.exists():
                with path.open("rb") as file:
                    while data := file.read(8 * 1024 * 1024):
                        digest.update(data)
        metadata["graphs"][name] = digest.hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def export_mel(processor, output):
    export(
        LogMel(processor),
        (torch.zeros(1, 16000),),
        output / "mel.onnx",
        ["samples"],
        ["features"],
        {"samples": {1: torch.export.Dim("samples", min=8000, max=1205 * 16000)}},
    )


@torch.inference_mode()
def export_encoder(model, output):
    dtype = model.dtype
    export(
        AudioEncoder(model),
        (torch.zeros(2, 128, 100, dtype=dtype), torch.arange(26), torch.zeros(1, 1, 26, 26, dtype=dtype)),
        output / "encoder.onnx",
        ["mel_chunks", "valid_indices", "attention_bias"],
        ["audio_embeddings"],
        {
            "mel_chunks": {0: torch.export.Dim("chunks", min=1)},
            "valid_indices": {0: torch.export.Dim("audio_tokens", min=1)},
            "attention_bias": {2: torch.export.Dim("audio_tokens", min=1), 3: torch.export.Dim("audio_tokens", min=1)},
        },
    )


@torch.inference_mode()
def export_decoder(model, output, cache_capacity):
    dtype = model.dtype
    c = model.config.text_config
    past = tuple(
        torch.zeros(1, c.num_key_value_heads, cache_capacity, c.head_dim, dtype=dtype)
        for _ in range(2 * c.num_hidden_layers)
    )
    seq = torch.export.Dim("sequence", min=1, max=512)
    inputs = (
        torch.ones(1, 3, dtype=torch.int64),
        torch.zeros(1, 3, c.hidden_size, dtype=dtype),
        torch.zeros(1, 3, 1, dtype=torch.bool),
        torch.arange(3)[None],
        torch.zeros(1, 1, 3, cache_capacity, dtype=dtype),
        torch.tensor([2]),
        *past,
    )
    names = ["input_ids", "audio_embeddings", "audio_mask", "position_ids", "attention_bias", "logits_index"]
    export(
        TextDecoder(model),
        inputs,
        output / "decoder.onnx",
        names + [f"past_{i}" for i in range(len(past))],
        ["logits", "next_token"] + [f"present_{i}" for i in range(len(past))],
        {
            "input_ids": {1: seq},
            "audio_embeddings": {1: seq},
            "audio_mask": {1: seq},
            "position_ids": {1: seq},
            "attention_bias": {2: seq},
            "logits_index": {},
            "past": tuple({} for _ in past),
        },
    )


@torch.inference_mode()
def export_aligner(model, output):
    dtype = model.dtype
    seq = torch.export.Dim("sequence", min=1)
    inputs = (
        torch.ones(1, 3, dtype=torch.int64),
        torch.zeros(1, 3, model.config.text_config.hidden_size, dtype=dtype),
        torch.zeros(1, 3, 1, dtype=torch.bool),
        torch.arange(3)[None],
        torch.zeros(1, 1, 3, 3, dtype=dtype),
        torch.tensor([1, 2]),
    )
    export(
        ForcedAligner(model),
        inputs,
        output / "aligner.onnx",
        ["input_ids", "audio_embeddings", "audio_mask", "position_ids", "attention_bias", "timestamp_indices"],
        ["timestamp_logits", "timestamp_bins"],
        {
            "input_ids": {1: seq},
            "audio_embeddings": {1: seq},
            "audio_mask": {1: seq},
            "position_ids": {1: seq},
            "attention_bias": {2: seq, 3: seq},
            "timestamp_indices": {0: torch.export.Dim("timestamp_slots", min=1)},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--checkpoint", dest="model", help="HF model ID or local checkpoint directory")
    parser.add_argument("--size", choices=["0.6B", "1.7B"], default="0.6B", help="ASR model size")
    parser.add_argument("--revision", help="Optional HF revision or commit")
    parser.add_argument("--output", type=Path, help="Defaults to the ONNX artifact directory for --task")
    parser.add_argument("--task", choices=["asr", "aligner"], default="asr")
    parser.add_argument("--dtype", choices=["original", "fp16", "fp32"], default="original")
    parser.add_argument("--only", choices=["mel", "encoder", "decoder", "aligner"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-capacity", type=int, choices=range(512, 16385, 512), default=8192, metavar="TOKENS")
    args = parser.parse_args()
    prefix = "aligner-" if args.task == "aligner" else ""
    args.model = args.model or (
        "Qwen/Qwen3-ForcedAligner-0.6B-hf" if args.task == "aligner" else f"Qwen/Qwen3-ASR-{args.size}-hf"
    )
    precision = "bf16" if args.dtype == "original" else args.dtype
    size_suffix = "-1.7b" if args.task == "asr" and "1.7b" in args.model.lower() else ""
    args.output = args.output or Path(f"artifacts/qwen3/{prefix}onnx-{precision}{size_suffix}")
    if args.only in ("decoder", "aligner") and args.only != ("decoder" if args.task == "asr" else "aligner"):
        parser.error("--only must match --task")
    metadata_path = args.output / "metadata.json"
    existing = json.loads(metadata_path.read_text(encoding="utf-8")) if args.only and metadata_path.exists() else {}
    requested = {"original": "bfloat16", "fp16": "float16", "fp32": "float32"}[args.dtype]
    if (
        existing
        and args.only != "mel"
        and (existing["dtype"] != requested or existing["task"] != args.task or existing.get("quantization"))
    ):
        parser.error("Partial export requires the same task and precision, without quantization; use a new directory")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    revision = args.revision
    if args.only != "mel":
        model_class = Qwen3ASRForConditionalGeneration if args.task == "asr" else Qwen3ASRForTokenClassification
        model = model_class.from_pretrained(
            args.model,
            dtype={"original": "auto", "fp16": torch.float16, "fp32": torch.float32}[args.dtype],
            attn_implementation="eager",
            revision=args.revision,
        ).eval()
        revision = model.config._commit_hash or args.revision
    processor = AutoProcessor.from_pretrained(args.model, revision=revision)
    processor.save_pretrained(args.output / "processor")
    save_native_assets(processor, args.output, args.task)
    if args.only in (None, "mel"):
        export_mel(processor, args.output)
    if args.only == "mel":
        if existing:
            save_metadata(args.output, existing)
        return
    cfg = model.config
    dtype = model.dtype
    if args.only in (None, "encoder"):
        export_encoder(model, args.output)
    if args.task == "asr" and args.only in (None, "decoder"):
        export_decoder(model, args.output, args.cache_capacity)
    if args.task == "aligner" and args.only in (None, "aligner"):
        export_aligner(model, args.output)
    eos = cfg.eos_token_id
    metadata = {
        "format_version": 3 if args.task == "asr" else 2,
        "cache_capacity": args.cache_capacity if args.task == "asr" else None,
        "prefill_block": 512 if args.task == "asr" else None,
        "task": args.task,
        "dtype": str(dtype).removeprefix("torch."),
        "opset": 24,
        "batch_size": 1,
        "audio_config": cfg.audio_config.to_dict(),
        "text_config": cfg.text_config.to_dict(),
        "audio_token_id": cfg.audio_token_id,
        "eos_token_ids": list(eos) if isinstance(eos, (list, tuple)) else [eos],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if args.task == "aligner":
        metadata.update(
            timestamp_bins=True,
            timestamp_token_id=cfg.timestamp_token_id,
            num_labels=cfg.num_labels,
            timestamp_segment_time=processor.timestamp_segment_time,
        )
    metadata["source"] = {"model": args.model, "revision": revision}
    if args.only == "encoder" and existing:
        for key in ("cache_capacity", "format_version", "prefill_block", "timestamp_bins"):
            if key in existing:
                metadata[key] = existing[key]
    save_metadata(args.output, metadata)


if __name__ == "__main__":
    main()
