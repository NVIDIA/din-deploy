# SPDX-License-Identifier: Apache-2.0
"""Compare exported diarization probabilities to the official NeMo streaming pipeline."""

import argparse
import json
import sys
from pathlib import Path

import onnxruntime as ort
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mel import LogMel  # noqa: E402
from model import DiarizationStep, load_model  # noqa: E402
from torch.nn import functional as F


def run(session, inputs):
    values = {name: ort.OrtValue.from_dlpack(tensor.contiguous()) for name, tensor in inputs.items()}
    return [torch.from_dlpack(value).clone() for value in session.run_with_ort_values(None, values)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=95)
    parser.add_argument("--provider", choices=("cpu", "trt-rtx"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-output", type=Path, help="Check C++ JSON output for the same audio and precision")
    args = parser.parse_args()
    with sf.SoundFile(args.audio) as file:
        if file.samplerate != 16000:
            raise ValueError("Validation audio must be 16 kHz; the C++ pipeline handles resampling")
        samples = (
            torch.frombuffer(
                bytearray(file.buffer_read(int(args.seconds * file.samplerate), dtype="float32")), dtype=torch.float32
            )
            .reshape(-1, file.channels)
            .mean(1)
        )
        samples = samples[None]
    metadata = json.loads((args.model_dir / "metadata.json").read_text())
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[metadata["dtype"]]
    if args.provider == "cpu" and dtype != torch.float32:
        raise ValueError("Use an FP32 export for CPU validation")
    model = load_model(args.checkpoint).cuda()
    model.encoder.to(dtype=dtype)
    model.sortformer_modules.to(dtype=dtype)
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        reference = model(
            audio_signal=samples.cuda(), audio_signal_length=torch.tensor([samples.numel()], device="cuda")
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    if args.provider == "trt-rtx":
        import onnxruntime_ep_nv_tensorrt_rtx as ep

        ort.register_execution_provider_library(ep.get_ep_name(), ep.get_library_path())
        devices = [d for d in ort.get_ep_devices() if d.ep_name == ep.get_ep_name()]
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_provider_for_devices(devices, {"enable_cuda_graph": "0"})
    session = ort.InferenceSession(str(args.model_dir / "diarization.onnx"), options)
    length = samples.numel() // 160
    history, history_preds = torch.zeros(1, 304, 512), torch.zeros(1, 264, 8)
    results = []
    for offset in range(0, length, 2720):
        count = min(3040, length - offset)
        begin = offset * 160 - 257
        lo, hi = max(0, begin), min(samples.numel(), begin + LogMel.samples)
        pcm = F.pad(samples[:, lo:hi], (lo - begin, begin + LogMel.samples - hi))
        probabilities, history, history_preds = run(
            session,
            dict(
                zip(
                    DiarizationStep.input_names,
                    (
                        pcm,
                        torch.tensor([lo - begin, hi - begin]),
                        torch.tensor([count]),
                        history,
                        history_preds,
                        torch.tensor([304 if offset else 0]),
                    ),
                    strict=True,
                )
            ),
        )
        results.append(probabilities[:, : min(2720, int(length) - offset)])
    actual = torch.cat(results, dim=1)
    reference = reference.cpu()
    error = (actual - reference).abs()
    speech = (actual.amax(-1) > 0.5) | (reference.amax(-1) > 0.5)
    speaker_agreement = (actual.argmax(-1)[speech] == reference.argmax(-1)[speech]).float()
    active_union = (actual > 0.5) | (reference > 0.5)
    active_disagreement = ((actual > 0.5) != (reference > 0.5)).sum() / active_union.sum().clamp_min(1)
    report = {
        "dtype": metadata["dtype"],
        "provider": args.provider,
        "frames": actual.shape[1],
        "max_probability_error": error.max().item(),
        "mean_probability_error": error.mean().item(),
        "activity_agreement": ((actual > 0.5) == (reference > 0.5)).float().mean().item(),
        "active_disagreement": active_disagreement.item(),
        "speaker_agreement": speaker_agreement.mean().item() if speaker_agreement.numel() else 1.0,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if dtype == torch.float32:
        torch.testing.assert_close(actual.float(), reference.float(), atol=3e-3, rtol=3e-3)
    else:
        # Reduced-precision fusion and cache ranking can move individual boundary probabilities.
        assert report["mean_probability_error"] < 0.005
        assert report["active_disagreement"] < 0.01
        assert report["speaker_agreement"] >= 0.99
    if args.native_output:
        from nemo.collections.asr.parts.utils.vad_utils import binarization_vectorized

        expected = []
        for speaker in range(8):
            spans = binarization_vectorized(
                actual[0, :, speaker], {"onset": 0.5, "offset": 0.5, "frame_length_in_sec": 0.01}
            )
            expected.extend((float(a), float(b), speaker) for a, b in spans)
        expected.sort(key=lambda span: (span[0], span[2]))
        text = args.native_output.read_text(encoding="utf-8")
        native = json.loads(text[text.index("{") :])
        assert abs(native["audio_seconds"] - samples.numel() / 16000) < 0.01
        observed = [(s["start_time"], s["end_time"], s["speaker"]) for s in native["segments"]]
        torch.testing.assert_close(torch.tensor(observed), torch.tensor(expected), rtol=0, atol=0.011)
        print("Native timestamps match ONNX / NeMo postprocessing")


if __name__ == "__main__":
    main()
