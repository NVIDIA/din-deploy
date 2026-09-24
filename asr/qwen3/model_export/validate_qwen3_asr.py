# SPDX-License-Identifier: Apache-2.0
"""Validate Qwen3 ASR or forced alignment against HF in checkpoint precision."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import onnxruntime as ort
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration, Qwen3ASRForTokenClassification
from transformers.modeling_outputs import CausalLMOutputWithPast

_REGISTERED_EP = None


def session_options(provider, threads, extra_options=None):
    global _REGISTERED_EP
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    if provider == "cpu":
        return options, ["CPUExecutionProvider"]
    if provider != "trt-rtx":
        raise ValueError(f"Unknown provider: {provider}")
    import onnxruntime_ep_nv_tensorrt_rtx as ep

    if _REGISTERED_EP is None:
        _REGISTERED_EP = ep.get_ep_name()
        ort.register_execution_provider_library(_REGISTERED_EP, ep.get_library_path())
    devices = [device for device in ort.get_ep_devices() if device.ep_name == _REGISTERED_EP]
    if not devices:
        raise RuntimeError("TensorRT RTX registered but no GPU was discovered")
    options.add_provider_for_devices(devices, {"enable_cuda_graph": "0", **(extra_options or {})})
    return options, None


def decoder_inputs(ids, embeddings, audio_token_id, hidden_size, past_length, capacity=None):
    ids = ids.cpu().long().reshape(1, -1)
    seq = ids.shape[1]
    if not seq or past_length < 0 or (capacity is not None and past_length + seq > capacity):
        raise ValueError("Input exceeds KV cache capacity or has invalid length")
    mask = (ids == audio_token_id)[..., None] if embeddings is not None else torch.zeros(1, seq, 1, dtype=torch.bool)
    padded = torch.zeros(1, seq, hidden_size)
    if embeddings is not None:
        if int(mask.sum()) != len(embeddings):
            raise ValueError("Audio placeholder count differs from encoder output length")
        padded[mask[..., 0]] = embeddings.float().cpu()
    positions = torch.arange(past_length, past_length + seq)[None]
    allowed = torch.arange(capacity or past_length + seq)[None] <= positions.T
    return {
        "input_ids": ids,
        "audio_embeddings": padded,
        "audio_mask": mask,
        "position_ids": positions,
        "attention_bias": torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, -1e4)[None, None],
    }


class OnnxAudioModel:
    def __init__(self, directory, task, threads=4, provider=None):
        text_graph = "decoder.onnx" if task == "asr" else "aligner.onnx"
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata["task"] != task or self.metadata["format_version"] != (3 if task == "asr" else 2):
            raise ValueError("Incompatible export; re-export the model with the current exporter")
        self.processor = AutoProcessor.from_pretrained(directory / "processor", local_files_only=True)
        self.dtype = getattr(torch, self.metadata["dtype"])
        self.provider = provider or ("trt-rtx" if self.dtype in (torch.float16, torch.bfloat16) else "cpu")
        encoder_shape = "mel_chunks:8x128x100,valid_indices:104,attention_bias:1x1x104x104"
        options, providers = session_options(
            self.provider, threads, {f"nv_profile_{bound}_shapes": encoder_shape for bound in ("min", "opt", "max")}
        )
        self.encoder = ort.InferenceSession(str(directory / "encoder.onnx"), options, providers=providers)
        if text_graph == "decoder.onnx":
            c = self.metadata["text_config"]
            capacity = self.metadata["cache_capacity"]

            def shapes(sequence):
                return (
                    f"input_ids:1x{sequence},audio_embeddings:1x{sequence}x{c['hidden_size']},"
                    f"audio_mask:1x{sequence}x1,position_ids:1x{sequence},"
                    f"attention_bias:1x1x{sequence}x{capacity},logits_index:1"
                )

            sessions = []
            for sequence in (1, self.metadata["prefill_block"]):
                options, providers = session_options(
                    self.provider,
                    threads,
                    {f"nv_profile_{bound}_shapes": shapes(sequence) for bound in ("min", "opt", "max")},
                )
                sessions.append(ort.InferenceSession(str(directory / text_graph), options, providers=providers))
            self.decoder, self.prefill = sessions
        else:
            options, providers = session_options(self.provider, threads)
            self.decoder = ort.InferenceSession(str(directory / text_graph), options, providers=providers)

    def run(self, session, feed, inplace=False):
        values = {}
        for name, tensor in feed.items():
            tensor = tensor.detach().to("cuda" if inplace else "cpu").contiguous()
            if tensor.is_floating_point():
                tensor = tensor.to(self.dtype)
            # DLPack preserves BF16; ORT expects boolean capsules encoded as uint8.
            if tensor.dtype == torch.bool:
                capsule = torch.utils.dlpack.to_dlpack(tensor.view(torch.uint8))
                values[name] = ort.OrtValue(ort.capi._pybind_state.OrtValue.from_dlpack(capsule, True))
            else:
                values[name] = ort.OrtValue.from_dlpack(tensor)
        if inplace:
            # TensorRT requires TensorScatter past/present to alias, even for validation.
            binding = session.io_binding()
            for name, value in values.items():
                binding.bind_ortvalue_input(name, value)
            for output in session.get_outputs():
                if output.name.startswith("present_"):
                    binding.bind_ortvalue_output(output.name, values[output.name.replace("present_", "past_")])
                else:
                    binding.bind_output(output.name)
            torch.cuda.synchronize()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            outputs = binding.get_outputs()
        else:
            outputs = session.run_with_ort_values(None, values)
        return [torch.from_dlpack(value).cpu().clone() for value in outputs]

    def encode(self, inputs):
        config = self.metadata["audio_config"]
        window = config["n_window_infer"]
        features, mask = inputs["input_features"], inputs["input_features_mask"]
        outputs = []
        for start in range(0, int(mask.sum()), window):
            chunk_size = config["n_window"] * 2
            chunks = features[..., start : start + window].reshape(128, -1, chunk_size).permute(1, 0, 2)
            lengths = mask[..., start : start + window].reshape(-1, chunk_size).sum(1)
            valid_tokens = int(((lengths + 7) // 8).sum())
            tokens = window // 100 * 13
            bias = torch.zeros(1, 1, tokens, tokens)
            bias[..., valid_tokens:] = -1e4
            packed = torch.zeros(window // 100, 128, 100)
            packed[: len(chunks)] = chunks
            outputs.append(
                self.run(
                    self.encoder, {"mel_chunks": packed, "valid_indices": torch.arange(tokens), "attention_bias": bias}
                )[0][:valid_tokens]
            )
        return torch.cat(outputs)

    def empty_cache(self, capacity):
        c = self.metadata["text_config"]
        return [
            torch.zeros(1, c["num_key_value_heads"], capacity, c["head_dim"], dtype=self.dtype)
            for _ in range(2 * c["num_hidden_layers"])
        ]

    def step(self, ids, embeddings, cache, length):
        ids = ids.reshape(1, -1)
        block = self.metadata["prefill_block"] if ids.numel() > 1 else 1
        audio_offset = 0
        for offset in range(0, ids.numel(), block):
            chunk = ids[:, offset : offset + block]
            valid = chunk.numel()
            padded = torch.zeros(1, block, dtype=torch.int64)
            padded[:, :valid] = chunk
            count = int((chunk == self.metadata["audio_token_id"]).sum())
            audio = embeddings[audio_offset : audio_offset + count] if embeddings is not None else None
            audio_offset += count
            feed = decoder_inputs(
                padded,
                audio,
                self.metadata["audio_token_id"],
                self.metadata["text_config"]["hidden_size"],
                length + offset,
                cache[0].shape[2],
            )
            feed["logits_index"] = torch.tensor([valid - 1])
            feed.update({f"past_{i}": value for i, value in enumerate(cache)})
            logits, _, *cache = self.run(
                self.prefill if block > 1 else self.decoder, feed, inplace=self.provider == "trt-rtx"
            )
        return logits, cache


def compare(actual, expected, atol):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    error = (actual - expected).abs().max().item()
    return {
        "passed": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all() and error <= atol),
        "max_abs_error": error,
    }


def compare_cache(actual, expected, length, atol):
    values = [value for layer in expected.layers for value in (layer.keys, layer.values)]
    checks = [compare(a[:, :, :length], b, atol) for a, b in zip(actual, values, strict=True)]
    return {"passed": all(c["passed"] for c in checks), "max_abs_error": max(c["max_abs_error"] for c in checks)}


def generate_onnx(reference, runner, inputs, embeddings, capacity, max_new_tokens):
    cache, length = runner.empty_cache(capacity), 0

    def forward(input_ids, input_features=None, input_features_mask=None, attention_mask=None, **kwargs):
        nonlocal cache, length
        ids = input_ids[:, length:]
        logits, cache = runner.step(ids, embeddings if length == 0 else None, cache, length)
        length += ids.shape[1]
        return CausalLMOutputWithPast(logits=logits[:, None].to(input_ids.device))

    # HF owns token selection and stopping. Only graph execution is replaced.
    with patch.object(reference, "forward", forward):
        return reference.generate(**inputs, use_cache=False, do_sample=False, max_new_tokens=max_new_tokens)


def validate_asr(args, runner, reference, inputs, ref_inputs):
    length = inputs["input_ids"].shape[1]
    required = length + args.max_new_tokens
    metadata = runner.metadata
    capacity = metadata["cache_capacity"]
    if required > capacity:
        raise ValueError("Prompt and generation budget exceed exported cache capacity")
    audio = runner.encode(inputs)
    expected_audio = reference.get_audio_features(
        ref_inputs["input_features"], ref_inputs["input_features_mask"]
    ).pooler_output
    expected = reference(**ref_inputs, use_cache=True, logits_to_keep=1)
    logits, cache = runner.step(inputs["input_ids"], audio, runner.empty_cache(capacity), 0)
    checks = {
        "encoder": compare(audio, expected_audio, args.atol),
        "prefill_logits": compare(logits, expected.logits[:, -1], args.atol),
        "prefill_kv": compare_cache(cache, expected.past_key_values, length, args.atol),
    }
    token = expected.logits[:, -1].argmax(-1, keepdim=True)
    next_expected = reference(input_ids=token, past_key_values=expected.past_key_values, use_cache=True)
    next_logits, present = runner.step(token, None, cache, length)
    checks["cached_logits"] = compare(next_logits, next_expected.logits[:, -1], args.atol)
    checks["cached_kv"] = compare_cache(present, next_expected.past_key_values, length + 1, args.atol)
    expected_ids = reference.generate(**ref_inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    actual_ids = generate_onnx(reference, runner, ref_inputs, audio, capacity, args.max_new_tokens)
    expected_tokens, actual_tokens = expected_ids[0, length:].tolist(), actual_ids[0, length:].tolist()
    eos = metadata["eos_token_ids"]
    exact = expected_tokens == actual_tokens
    stopped = bool(expected_tokens and actual_tokens and expected_tokens[-1] in eos and actual_tokens[-1] in eos)
    return {
        "checks": checks,
        "exact_token_match": exact,
        "reached_eos": stopped,
        "reference_tokens": expected_tokens,
        "onnx_tokens": actual_tokens,
        "passed": all(c["passed"] for c in checks.values()) and exact and stopped,
    }


def validate_aligner(args, runner, reference, inputs, ref_inputs, words):
    audio = runner.encode(inputs)
    expected_audio = reference.model.get_audio_features(
        ref_inputs["input_features"], ref_inputs["input_features_mask"]
    ).pooler_output
    expected = reference(**ref_inputs, use_cache=False).logits.cpu()
    timestamp_id = reference.config.timestamp_token_id
    indices = (inputs["input_ids"][0] == timestamp_id).nonzero().flatten()
    feed = decoder_inputs(
        inputs["input_ids"], audio, runner.metadata["audio_token_id"], runner.metadata["text_config"]["hidden_size"], 0
    )
    feed["timestamp_indices"] = indices
    logits = runner.run(runner.decoder, feed)[0]
    processor = runner.processor
    expected_spans = processor.decode_forced_alignment(expected.float(), inputs["input_ids"], words, timestamp_id)[0]
    actual_spans = processor.decode_forced_alignment(
        logits.float(), torch.full((1, len(indices)), timestamp_id), words, timestamp_id
    )[0]
    checks = {
        "encoder": compare(audio, expected_audio, args.atol),
        "timestamp_logits": compare(logits, expected[:, indices], args.atol),
    }
    exact = actual_spans == expected_spans
    return {
        "checks": checks,
        "exact_spans": exact,
        "reference_spans": expected_spans,
        "onnx_spans": actual_spans,
        "passed": all(c["passed"] for c in checks.values()) and exact,
    }


@torch.inference_mode()
def validate(args):
    runner = OnnxAudioModel(args.onnx_dir, args.task, args.threads, args.provider)
    model_type = Qwen3ASRForConditionalGeneration if args.task == "asr" else Qwen3ASRForTokenClassification
    reference = (
        model_type.from_pretrained(args.model, revision=args.revision, dtype=runner.dtype, attn_implementation="eager")
        .to(args.reference_device)
        .eval()
    )
    reports = []
    for path in args.audio:
        if args.task == "asr":
            inputs = runner.processor.apply_transcription_request(
                audio=str(path), language=args.language, return_tensors="pt"
            )
        else:
            inputs, words = runner.processor.prepare_forced_aligner_inputs(
                audio=str(path),
                transcript=args.transcript.read_text(encoding="utf-8-sig").strip(),
                language=args.language,
                return_tensors="pt",
            )
            if not words[0]:
                raise ValueError("Transcript must contain alignable words")
        ref_inputs = {
            k: v.to(device=args.reference_device, dtype=runner.dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }
        report = (
            validate_asr(args, runner, reference, inputs, ref_inputs)
            if args.task == "asr"
            else validate_aligner(args, runner, reference, inputs, ref_inputs, words)
        )
        reports.append({"audio": str(path), **report})
    payload = {
        "passed": all(r["passed"] for r in reports),
        "atol": args.atol,
        "provider": runner.provider,
        "dtype": str(runner.dtype),
        "source": runner.metadata["source"],
        "cases": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(0 if payload["passed"] else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["asr", "aligner"], default="asr")
    parser.add_argument("--onnx-dir", type=Path)
    parser.add_argument(
        "--model", "--checkpoint", dest="model", help="HF model ID or local checkpoint; defaults to export metadata"
    )
    parser.add_argument("--revision", help="Optional HF revision; defaults to export metadata")
    parser.add_argument("--audio", type=Path, nargs="+", default=[Path("assets/sample.wav")])
    parser.add_argument("--transcript", type=Path, help="UTF-8 transcript file, required for alignment")
    parser.add_argument("--language")
    parser.add_argument("--provider", choices=["cpu", "trt-rtx"])
    parser.add_argument("--reference-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=0.005)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.max_new_tokens < 1 or args.atol <= 0:
        parser.error("threads, max-new-tokens and atol must be positive")
    if args.task == "aligner" and (args.transcript is None or len(args.audio) != 1):
        parser.error("Alignment requires one --audio and a --transcript")
    prefix = "aligner-" if args.task == "aligner" else ""
    args.onnx_dir = args.onnx_dir or Path(f"artifacts/qwen3/{prefix}onnx-bf16")
    args.report = args.report or Path(f"artifacts/qwen3/validation-{prefix}bf16.json")
    metadata = json.loads((args.onnx_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["task"] != args.task:
        parser.error("--task does not match the exported model")
    source = metadata["source"]
    if args.model is None:
        args.model = source["model"]
        args.revision = args.revision or source.get("revision")
    torch.set_num_threads(args.threads)
    if args.task == "aligner":
        args.language = args.language or "English"
    validate(args)


if __name__ == "__main__":
    main()
