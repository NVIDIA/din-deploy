# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Validate exported Whisper ONNX graphs with real audio only.

The script generates a HuggingFace reference transcription and a transcription
using ONNX Runtime encoder/decoder wrappers, then reports WER between them.
It supports the CPU EP and the TensorRT-RTX plugin EP, including an explicitly
supplied locally built provider DLL.

Example:
    python validate_whisper.py --model openai/whisper-tiny \
        --onnx-dir D:/models/whisper-tiny-onnx-fp16 \
        --audio audio.mp3 --truth transcript.txt --provider trt-rtx --dtype fp16
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import scipy.signal as sps
import soundfile as sf
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_whisper as ew  # noqa: E402


EP_NAME = "nv_tensorrt_rtx"
SR = 16000
NP_DT = {"fp16": np.float16, "fp32": np.float32}


def register_trt_rtx(args) -> str:
    """Register TensorRT-RTX, preferring an explicit local provider DLL."""
    if args.ep_lib:
        name, library = EP_NAME, str(args.ep_lib)
        for directory in (args.ep_dll_dir, args.trt_bin):
            if directory and os.path.isdir(directory):
                os.add_dll_directory(os.path.abspath(directory))
        print(f"  using explicit TensorRT-RTX EP library: {library}")
    else:
        try:
            import onnxruntime_ep_nv_tensorrt_rtx as ep  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "trt-rtx needs onnxruntime-ep-nv-tensorrt-rtx or --ep-lib"
            ) from exc
        name, library = ep.get_ep_name(), ep.get_library_path()
        print(f"  using pip package onnxruntime-ep-nv-tensorrt-rtx ({name})")
    ort.register_execution_provider_library(name, library)
    return name


def make_sessions(onnx_dir: Path, provider: str, args):
    options = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    if provider == "trt-rtx":
        name = register_trt_rtx(args)
        devices = [device for device in ort.get_ep_devices() if device.ep_name == name]
        if not devices:
            raise SystemExit(f"EP '{name}' registered but no devices were discovered")
        options.add_provider_for_devices(devices, {})
        providers = None
    encoder = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), sess_options=options, providers=providers)
    decoder = ort.InferenceSession(str(onnx_dir / "decoder.onnx"), sess_options=options, providers=providers)
    print(f"  encoder providers: {encoder.get_providers()}")
    print(f"  decoder providers: {decoder.get_providers()}")
    return encoder, decoder


def ep_vendor_id() -> int:
    devices = [device for device in ort.get_ep_devices() if device.ep_name == EP_NAME]
    if not devices:
        raise SystemExit("TensorRT-RTX EP exposes no device for IO binding")
    return devices[0].memory_info(ort.OrtDeviceMemoryType.DEFAULT).device_vendor_id


def device_value_from_numpy(array, vendor):
    return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(array), "gpu", 0, vendor)


def device_value(shape, np_dtype, vendor):
    return ort.OrtValue.ortvalue_from_shape_and_type(list(shape), np_dtype, "gpu", 0, vendor)


def load_audio(path: str, seconds: float) -> np.ndarray:
    data, sample_rate = sf.read(path, always_2d=True)
    data = data.mean(axis=1).astype(np.float32)
    if seconds:
        data = data[: int(sample_rate * seconds)]
    if sample_rate != SR:
        data = sps.resample(data, int(len(data) * SR / sample_rate)).astype(np.float32)
    return data


class OrtEncoder(nn.Module):
    """Whisper encoder wrapper that also retains projected cross-attention K/V."""

    def __init__(self, session, n_layers, np_dtype, device_io, device, vendor=None):
        super().__init__()
        self.session, self.n_layers, self.np_dtype = session, n_layers, np_dtype
        self.device_io, self.device, self.vendor = device_io, device, vendor
        self.cross_k = self.cross_v = self._keep = None

    def forward(self, input_features=None, **_):
        if not self.device_io:
            features = input_features.detach().cpu().numpy().astype(self.np_dtype)
            outputs = self.session.run(None, {"audio_features": features})
            values = {output.name: value for output, value in zip(self.session.get_outputs(), outputs)}
            self.cross_k = [values[f"present_key_cross_{i}"].astype(self.np_dtype) for i in range(self.n_layers)]
            self.cross_v = [values[f"present_value_cross_{i}"].astype(self.np_dtype) for i in range(self.n_layers)]
            hidden = torch.from_numpy(values["hidden_states"].astype(np.float32)).to(input_features.device)
            return BaseModelOutput(last_hidden_state=hidden)

        features = input_features.detach().cpu().numpy().astype(self.np_dtype)
        input_value = device_value_from_numpy(features, self.vendor)
        binding = self.session.io_binding()
        binding.bind_ortvalue_input("audio_features", input_value)
        outputs = {output.name: device_value(output.shape, self.np_dtype, self.vendor)
                   for output in self.session.get_outputs()}
        for name, value in outputs.items():
            binding.bind_ortvalue_output(name, value)
        self.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        self._keep = (input_value, outputs)
        self.cross_k = [outputs[f"present_key_cross_{i}"] for i in range(self.n_layers)]
        self.cross_v = [outputs[f"present_value_cross_{i}"] for i in range(self.n_layers)]
        hidden = torch.from_numpy(outputs["hidden_states"].numpy()).to(input_features.device)
        return BaseModelOutput(last_hidden_state=hidden)


class OrtDecoder(nn.Module):
    """Whisper fixed-KV-cache decoder wrapper returning baked-in LM logits."""

    def __init__(self, session, encoder, config, np_dtype, device_io, vendor=None):
        super().__init__()
        self.session, self.encoder, self.np_dtype = session, encoder, np_dtype
        self.device_io, self.vendor = device_io, vendor
        self.n_layers = config.decoder_layers
        self.n_heads = config.decoder_attention_heads
        self.head_dim = config.d_model // self.n_heads
        self.max_len, self.vocab_size = config.max_target_positions, config.vocab_size
        if device_io:
            cache_shape = (1, self.n_heads, self.max_len, self.head_dim)
            self.self_kv = [device_value(cache_shape, np_dtype, vendor) for _ in range(2 * self.n_layers)]
            self.logits_value = device_value((1, 1, self.vocab_size), np_dtype, vendor)

    def forward(self, input_ids=None, **_):
        return self._forward_device(input_ids) if self.device_io else self._forward_host(input_ids)

    def _forward_host(self, input_ids):
        ids = input_ids.detach().cpu().numpy()
        cache_shape = (1, self.n_heads, self.max_len, self.head_dim)
        self_k = [np.zeros(cache_shape, self.np_dtype) for _ in range(self.n_layers)]
        self_v = [np.zeros(cache_shape, self.np_dtype) for _ in range(self.n_layers)]
        logits = []
        for position in range(ids.shape[1]):
            feeds = {
                "input_ids": ids[:, position:position + 1].astype(np.int32),
                "write_indices": np.array([position], np.int64),
                "nonpad_kv_seqlen": np.array([position + 1], np.int64),
            }
            for layer in range(self.n_layers):
                feeds.update({
                    f"past_key_self_{layer}": self_k[layer],
                    f"past_value_self_{layer}": self_v[layer],
                    f"past_key_cross_{layer}": self.encoder.cross_k[layer],
                    f"past_value_cross_{layer}": self.encoder.cross_v[layer],
                })
            outputs = self.session.run(None, feeds)
            values = {output.name: value for output, value in zip(self.session.get_outputs(), outputs)}
            logits.append(values["logits"].astype(np.float32))
            for layer in range(self.n_layers):
                self_k[layer] = values[f"present_key_self_{layer}"]
                self_v[layer] = values[f"present_value_self_{layer}"]
        result = torch.from_numpy(np.concatenate(logits, axis=1)).to(input_ids.device)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=result, past_key_values=None)

    def _forward_device(self, input_ids):
        ids = input_ids.detach().cpu().numpy()
        binding = self.session.io_binding()
        for layer in range(self.n_layers):
            binding.bind_ortvalue_input(f"past_key_cross_{layer}", self.encoder.cross_k[layer])
            binding.bind_ortvalue_input(f"past_value_cross_{layer}", self.encoder.cross_v[layer])
            binding.bind_ortvalue_input(f"past_key_self_{layer}", self.self_kv[2 * layer])
            binding.bind_ortvalue_input(f"past_value_self_{layer}", self.self_kv[2 * layer + 1])
            binding.bind_ortvalue_output(f"present_key_self_{layer}", self.self_kv[2 * layer])
            binding.bind_ortvalue_output(f"present_value_self_{layer}", self.self_kv[2 * layer + 1])
        binding.bind_ortvalue_output("logits", self.logits_value)
        logits = []
        for position in range(ids.shape[1]):
            binding.bind_cpu_input("input_ids", ids[:, position:position + 1].astype(np.int32))
            binding.bind_cpu_input("write_indices", np.array([position], np.int64))
            binding.bind_cpu_input("nonpad_kv_seqlen", np.array([position + 1], np.int64))
            self.session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            logits.append(torch.from_numpy(self.logits_value.numpy()))
        result = torch.cat(logits, dim=1).float().to(input_ids.device)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=result, past_key_values=None)


def normalize(text: str) -> str:
    return " ".join("".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in text).split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--truth", default=None)
    parser.add_argument("--provider", choices=["cpu", "trt-rtx"], default="cpu")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--ep-lib", type=Path, default=None,
                        help="Local TensorRT-RTX EP DLL; overrides the installed EP package.")
    parser.add_argument("--ep-dll-dir", type=Path, default=None,
                        help="Directory containing dependencies of --ep-lib.")
    parser.add_argument("--trt-bin", type=Path, default=None,
                        help="TensorRT-RTX bin directory used to resolve --ep-lib dependencies.")
    parser.add_argument("--seconds", type=float, default=30.0, help="Audio window to transcribe.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    ew._configure_stdio()

    np_dtype = NP_DT[args.dtype]
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[audio] {args.audio} (first {args.seconds}s)")
    audio = load_audio(args.audio, args.seconds)
    processor = WhisperProcessor.from_pretrained(args.model)
    features = processor(audio, sampling_rate=SR, return_tensors="pt").input_features.to(device=device, dtype=torch_dtype)

    print(f"[hf] reference generate ({args.dtype})")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch_dtype).to(device).eval()
    generate_kwargs = dict(language=args.language, task="transcribe", max_new_tokens=args.max_new_tokens)
    with torch.inference_mode():
        hf_text = processor.batch_decode(model.generate(features, **generate_kwargs), skip_special_tokens=True)[0].strip()

    print(f"[onnx] real-audio generate ({args.provider})")
    encoder_session, decoder_session = make_sessions(args.onnx_dir, args.provider, args)
    device_io = args.provider == "trt-rtx" and device.type == "cuda"
    vendor = ep_vendor_id() if device_io else None
    print(f"[onnx] device IO binding: {'on' if device_io else 'off'}")
    encoder = OrtEncoder(encoder_session, model.config.decoder_layers, np_dtype, device_io, device, vendor)
    encoder.conv1, encoder.conv2 = model.model.encoder.conv1, model.model.encoder.conv2
    model.model.encoder = encoder
    model.model.decoder = OrtDecoder(decoder_session, encoder, model.config, np_dtype, device_io, vendor)
    model.proj_out = nn.Identity()
    model.generation_config.use_cache = False
    with torch.inference_mode():
        onnx_text = processor.batch_decode(
            model.generate(features, use_cache=False, **generate_kwargs), skip_special_tokens=True
        )[0].strip()

    import jiwer
    print("\n--- HF reference ---\n" + hf_text)
    print("\n--- ONNX (" + args.provider + ") ---\n" + onnx_text)
    print(f"\nWER(onnx vs hf) = {jiwer.wer(normalize(hf_text), normalize(onnx_text)):.4f}")
    if args.truth:
        truth = Path(args.truth).read_text(encoding="utf-8", errors="ignore")
        count = len(normalize(onnx_text).split())
        truth_prefix = " ".join(normalize(truth).split()[:count])
        print(f"WER(onnx vs truth-prefix) = {jiwer.wer(truth_prefix, normalize(onnx_text)):.4f}")
        print(f"WER(hf   vs truth-prefix) = {jiwer.wer(truth_prefix, normalize(hf_text)):.4f}")


if __name__ == "__main__":
    main()
