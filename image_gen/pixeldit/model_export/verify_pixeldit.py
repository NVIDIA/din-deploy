# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Verify PixelDiT T2I ONNX graphs against the vendored PyTorch model, and optionally verify the
vendored PyTorch model + sampler against the upstream NVlabs/PixelDiT repo.

Checks (see .claude/plans/pixeldit-export-plan.md, "Validation plan"):
    --validate_only    V5: ONNX structure + dummy run of each graph, then exit
    (default)          V5, V6: per-model ONNX Runtime vs PyTorch, V7: end-to-end image comparison
    --reference_repo   V2-V4: vendored model / text path / sampler vs upstream PixelDiT

Outputs (written inside --output_dir if given, otherwise --onnx_dir):
    pytorch_output.png  - image from the PyTorch pipeline
    ort_output.png      - image from the ONNX Runtime-backed pipeline
    comparison.png      - PyTorch (left) | ONNX Runtime (right) side-by-side
    upstream_output.png - image from upstream PixelDiT (--reference_repo only)

Pass/fail is decided by the per-model checks (V2, V3, V6). End-to-end images only fail if clearly
broken: 50-step bf16 sampling amplifies tiny per-step differences, so correct runs can differ in fine
detail or even composition depending on the seed. Review comparison.png visually.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from pixeldit import pipeline as P
from pixeldit.weights import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEXT_ENCODER_NAME,
    load_gemma_text_encoder,
    load_pixeldit,
    resolve_checkpoint,
)

EP_NAME = "nv_tensorrt_rtx"
_REGISTERED_EP_NAME: str | None = None
DEFAULT_PROMPT = (
    "Close-up portrait of a beautiful Baltic model wearing white flower-shaped earrings, emphasis on the earrings, "
    "everything is in full focus, pores and skin imperfections are visible, neutral lighting from a large studio "
    "softbox, realistic, high resolution, natural beauty, detailed texture, professional studio shooting."
)
BANNED_OPS = {"Im2Col", "Col2Im", "DFT", "STFT"}

# All comparisons are bf16 vs bf16. Thresholds are ~2x the bf16-vs-fp32 noise measured once on the
# 1024x1024 export (see .claude/plans/pixeldit-export-plan.md, "Findings"); bf16 rounding of the
# timestep makes the DiT far noisier near t*1000 = 1000 than elsewhere.
DIT_MAX_REL_L2 = {999.75: 0.55, 996.0: 0.025, 500.0: 0.025, 50.0: 0.025}  # DiT timestep input -> max rel L2
CHECK_TIMESTEPS = (999.75, 500.0, 50.0)
TEXT_MIN_COS = 0.999  # Gemma-2 bf16 vs fp32 is itself 0.9997-0.9998
# 50-step images diverge chaotically from tiny per-step bf16 differences: correct runs measured 18-40 dB
# depending on seed (seed 1 even changes the pose). The image checks only catch broken output; the
# per-model checks are the real gate. PSNR >= 12 dB rejects black (~5 dB) and noise (~9 dB) but not a
# flat image (~14 dB, these portraits have smooth backgrounds), so detail (mean |Laplacian|) must also
# stay within 0.5-2x of the reference (correct runs measured 0.72-1.01x).
MIN_E2E_PSNR = 12.0
DETAIL_RATIO_RANGE = (0.5, 2.0)

_failures: list[str] = []


def configure_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        _failures.append(name)


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.detach().double().flatten(), b.detach().double().flatten().to(a.device)
    return {
        "max_abs": (a - b).abs().max().item(),
        "rel_l2": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
        "cos": F.cosine_similarity(a, b, dim=0).item(),
        "bitwise": torch.equal(a, b),
    }


def _fmt(s: dict) -> str:
    return f"cos={s['cos']:.6f} rel_l2={s['rel_l2']:.3e} max_abs={s['max_abs']:.3e}"


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse < 1e-12 else 10.0 * math.log10(255.0**2 / mse)


# ---------------------------------------------------------------------------
# ONNX Runtime session helpers (same as verify_flux2.py)
# ---------------------------------------------------------------------------


def _np_dtype(type_name: str) -> np.dtype:
    return {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(bool)": np.bool_,
    }[type_name]


def _register_trt_rtx(args) -> str:
    global _REGISTERED_EP_NAME
    if _REGISTERED_EP_NAME is not None:
        return _REGISTERED_EP_NAME

    try:
        import onnxruntime_ep_nv_tensorrt_rtx as ep  # noqa: F401

        name, lib = ep.get_ep_name(), ep.get_library_path()
        print(f"  [ORT] Using onnxruntime-ep-nv-tensorrt-rtx ({name})")
    except ImportError:
        if not args.ep_lib:
            raise SystemExit("trt-rtx needs onnxruntime-ep-nv-tensorrt-rtx or --ep_lib")
        name, lib = EP_NAME, str(args.ep_lib)
        for d in (args.ep_dll_dir, args.trt_bin):
            if d and os.path.isdir(d):
                os.add_dll_directory(os.path.abspath(d))
    import onnxruntime as ort

    ort.register_execution_provider_library(name, lib)
    _REGISTERED_EP_NAME = name
    return name


def _make_session_options(provider: str, args):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if provider != "trt-rtx":
        return so, ["CPUExecutionProvider"]
    name = _register_trt_rtx(args)
    devices = [d for d in ort.get_ep_devices() if d.ep_name == name]
    if not devices:
        raise SystemExit(f"EP '{name}' registered but no devices were discovered")
    so.add_provider_for_devices(devices, {})
    return so, None


def _make_session(onnx_path: Path, provider: str, args):
    import onnxruntime as ort

    so, providers = _make_session_options(provider, args)
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)


class OrtSessionRunner:
    def __init__(self, session):
        self.session = session
        self.input_names = [i.name for i in session.get_inputs()]
        self.output_names = [o.name for o in session.get_outputs()]
        self.input_dtypes = {i.name: _np_dtype(i.type) for i in session.get_inputs()}
        self.output_dtypes = {o.name: _np_dtype(o.type) for o in session.get_outputs()}

    @classmethod
    def from_path(cls, onnx_path: Path, provider: str, args):
        return cls(_make_session(onnx_path, provider, args))

    def run(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self.session.run(None, feed)


def _to_numpy(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().to("cpu", torch.float32 if dtype == np.float32 else tensor.dtype).numpy().astype(dtype, copy=False))


def ort_encode_fn(runner: OrtSessionRunner, device):
    def encode(input_ids, attention_mask, select_index):
        feed = {
            "input_ids": _to_numpy(input_ids, runner.input_dtypes["input_ids"]),
            "attention_mask": _to_numpy(attention_mask, runner.input_dtypes["attention_mask"]),
            "select_index": _to_numpy(select_index, runner.input_dtypes["select_index"]),
        }
        return torch.from_numpy(runner.run(feed)[0]).to(device)

    return encode


def ort_dit_fn(runner: OrtSessionRunner, device):
    def dit(x, timestep, y):
        feed = {
            "hidden_states": _to_numpy(x, np.float32),
            "timestep": _to_numpy(timestep, np.float32),
            "encoder_hidden_states": _to_numpy(y, np.float32),
        }
        return torch.from_numpy(runner.run(feed)[0]).to(device)

    return dit


def torch_encode_fn(text_encoder):
    def encode(input_ids, attention_mask, select_index):
        return text_encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)[0][:, select_index]

    return encode


# ---------------------------------------------------------------------------
# V5: graph structure + dummy run
# ---------------------------------------------------------------------------

EXPECTED_IO = {
    "text_encoder": (["input_ids", "attention_mask", "select_index"], ["prompt_embeds"]),
    "transformer": (["hidden_states", "timestep", "encoder_hidden_states"], ["velocity"]),
}


def _onnx_io_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    shapes = {}
    for value in list(model.graph.input) + list(model.graph.output):
        if value.name not in initializer_names:
            shapes[value.name] = tuple(int(d.dim_value) for d in value.type.tensor_type.shape.dim)
    return shapes


def validate_onnx_dir(onnx_dir: Path, provider: str, args, expected_shapes: dict) -> dict[str, OrtSessionRunner]:
    """V5 checks; returns the ORT sessions so later checks don't rebuild TensorRT engines."""
    print(f"\n[V5 Validate] {onnx_dir}")
    runners = {}
    for name, (inputs, outputs) in EXPECTED_IO.items():
        onnx_path = onnx_dir / name / "model.onnx"
        if not onnx_path.exists():
            continue
        print(f"  [Validate] {name}")
        onnx.checker.check_model(str(onnx_path))
        model = onnx.load(str(onnx_path), load_external_data=False)
        shapes = _onnx_io_shapes(model)
        op_types = {node.op_type for node in model.graph.node}
        for function in model.functions:
            op_types.update(node.op_type for node in function.node)
        io_names = ([i.name for i in model.graph.input], [o.name for o in model.graph.output])
        _check(f"{name} io names", io_names == (inputs, outputs), f"{io_names}")
        for io_name, shape in expected_shapes.get(name, {}).items():
            _check(f"{name} {io_name} shape", shapes.get(io_name) == shape, f"{shapes.get(io_name)} (expected {shape})")
        banned = sorted(op_types & BANNED_OPS)
        _check(f"{name} no Im2Col/Col2Im/complex ops", not banned, f"found {banned}" if banned else "none")
        has_bf16 = any(t.data_type == onnx.TensorProto.BFLOAT16 for t in model.graph.initializer)
        _check(f"{name} bf16 weights", has_bf16, "bf16 initializers present" if has_bf16 else "no bf16 initializers")
        data_gb = (onnx_path.parent / "model.onnx_data").stat().st_size / 1e9
        print(f"    model.onnx_data: {data_gb:.2f} GB")

        runner = OrtSessionRunner.from_path(onnx_path, provider, args)
        float_io = [d for d in list(runner.input_dtypes.values()) + list(runner.output_dtypes.values()) if d in (np.float16, np.float32)]
        _check(f"{name} fp32 float IO", all(d == np.float32 for d in float_io), str(float_io))
        feed = {}
        for inp in runner.input_names:
            dtype = runner.input_dtypes[inp]
            shape = shapes[inp]
            if dtype == np.float32:
                feed[inp] = np.random.default_rng(0).standard_normal(shape).astype(dtype)
            elif inp == "attention_mask":
                feed[inp] = np.ones(shape, dtype)  # an all-zero mask has fully masked attention rows
            else:
                feed[inp] = np.zeros(shape, dtype)
            print(f"    input {inp}: shape={shape}, dtype={dtype}")
        for out_name, output in zip(runner.output_names, runner.run(feed), strict=True):
            finite = bool(np.isfinite(output).all())
            print(f"    output {out_name}: shape={tuple(output.shape)}, dtype={output.dtype}, finite={finite}")
            _check(f"{name} dummy run output finite", finite, out_name)
        runners[name] = runner
    if not runners:
        raise RuntimeError(f"No model.onnx files found under {onnx_dir}")
    return runners


# ---------------------------------------------------------------------------
# V6: per-model ONNX Runtime vs PyTorch
# ---------------------------------------------------------------------------


@torch.inference_mode()
def compare_text_encoder(tokenizer, text_encoder, runner, layout, prompt, negative_prompt, device):
    print("\n[V6] text_encoder: ONNX Runtime vs PyTorch (bf16)")
    torch_neg, torch_pos = P.encode_prompts(tokenizer, torch_encode_fn(text_encoder), layout, prompt, negative_prompt, device)
    ort_neg, ort_pos = P.encode_prompts(tokenizer, ort_encode_fn(runner, device), layout, prompt, negative_prompt, device)
    for label, ort, ref in (("positive", ort_pos, torch_pos), ("negative", ort_neg, torch_neg)):
        s = _stats(ort, ref)
        per_token = F.cosine_similarity(ort[0].double(), ref[0].double(), dim=-1).min().item()
        _check(f"text_encoder {label} prompt_embeds", s["cos"] >= TEXT_MIN_COS,
               f"{_fmt(s)} min_token_cos={per_token:.6f} (need cos >= {TEXT_MIN_COS})")


def dit_check(name: str, tv: float, out: torch.Tensor, ref: torch.Tensor) -> None:
    s = _stats(out, ref)
    limit = DIT_MAX_REL_L2[tv]
    _check(f"{name} t={tv}", s["rel_l2"] <= limit, f"{_fmt(s)} (need rel_l2 <= {limit})")


@torch.inference_mode()
def compare_dit(dit, runner, y_neg, y_pos, height, width, device):
    print("\n[V6] transformer: ONNX Runtime vs PyTorch (bf16)")
    g = torch.Generator(device=device).manual_seed(1)
    x = torch.randn(1, 3, height, width, device=device, generator=g).repeat(2, 1, 1, 1)
    y = torch.cat([y_neg, y_pos]).float()
    ort_dit = ort_dit_fn(runner, device)
    for tv in CHECK_TIMESTEPS:
        t = torch.full((2,), tv, device=device)
        dit_check("transformer velocity", tv, ort_dit(x, t, y), dit(x, t, y))


# ---------------------------------------------------------------------------
# V7: end to end
# ---------------------------------------------------------------------------


def _detail(image: np.ndarray) -> float:
    """High-frequency energy: mean |Laplacian| of luminance."""
    g = image.astype(np.float64).mean(-1)
    return float(np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).mean())


def check_image_sanity(name: str, image: np.ndarray, ref: np.ndarray) -> None:
    """Fail only on clearly broken output (black, noise, flat); otherwise report metrics for visual review."""
    psnr = _psnr(ref, image)
    mae = float(np.mean(np.abs(ref.astype(np.float32) - image.astype(np.float32))))
    ratio = _detail(image) / max(_detail(ref), 1e-6)
    lo, hi = DETAIL_RATIO_RANGE
    _check(f"{name} not broken: PSNR", psnr >= MIN_E2E_PSNR, f"PSNR={psnr:.2f} dB (need >= {MIN_E2E_PSNR} dB)")
    _check(f"{name} not broken: detail", lo <= ratio <= hi, f"detail ratio={ratio:.2f} (need {lo}-{hi})")
    print(f"  [INFO] {name}: PSNR={psnr:.2f} dB MAE={mae:.2f}. Fine details can differ between correct "
          "bf16 runs (18-40 dB measured across seeds); compare the images visually.")


def compare_images(ref: np.ndarray, ort: np.ndarray, output_dir: Path) -> None:
    check_image_sanity("end-to-end image", ort, ref)
    canvas = PILImage.new("RGB", (ref.shape[1] + ort.shape[1], max(ref.shape[0], ort.shape[0])), (255, 255, 255))
    canvas.paste(PILImage.fromarray(ref), (0, 0))
    canvas.paste(PILImage.fromarray(ort), (ref.shape[1], 0))
    cmp_path = output_dir / "comparison.png"
    canvas.save(str(cmp_path))
    print(f"[Comparison] Side-by-side saved -> {cmp_path}")


def _progress(label):
    def report(step):
        if step % 10 == 0:
            print(f"  [{label}] step {step}")

    return report


# ---------------------------------------------------------------------------
# V2-V4: vendored code vs upstream PixelDiT (--reference_repo)
# ---------------------------------------------------------------------------


def run_reference_checks(args, tokenizer, text_encoder, layout, dit, height, width, device, pytorch_image, output_dir) -> None:
    """pytorch_image() returns the vendored-pipeline image shared with V7 (generated once)."""
    repo = Path(args.reference_repo).resolve()
    sys.path[:0] = [str(repo), str(repo / "t2i")]
    import yaml
    from diffusion.model.dpm_solver import DPM_Solver, NoiseScheduleFlow
    from diffusion.model.flow_dpm import DPMS
    from diffusion.model.trainer import PixDiTTrainer
    from pixdit_core import modules as upstream_modules

    from pixeldit.model import apply_rotary_emb, pixel_sincos_pos_embed, rope_2d_angles

    print(f"\n[Reference] Upstream PixelDiT at {repo}")
    cfg = yaml.safe_load(open(repo / "t2i" / "configs" / "PixelDiT_1024px_pixel_diffusion_stage3.yaml", encoding="utf-8"))
    _check("chi_prompt matches upstream config", "\n".join(cfg["text_encoder"]["chi_prompt"]) == layout.chi_prompt, "")

    # V2 building blocks
    q, k = torch.randn(2, 4096, 24, 64), torch.randn(2, 4096, 24, 64)
    uq, uk = upstream_modules.apply_rotary_emb(q, k, upstream_modules.precompute_freqs_cis_2d(64, 64, 64))
    ang = rope_2d_angles(64, 64, 64)
    mq, mk = apply_rotary_emb(q, k, ang.cos(), ang.sin())
    err = max(_stats(mq, uq)["max_abs"], _stats(mk, uk)["max_abs"])
    _check("V2 real-valued RoPE vs complex", err < 1e-6, f"max_abs={err:.3e}")
    x = torch.randn(2, 3, height, width)
    hs, ws = height // 16, width // 16
    patches = x.view(2, 3, hs, 16, ws, 16).permute(0, 2, 4, 1, 3, 5).reshape(2, hs * ws, -1)
    _check("V2 patchify vs F.unfold", torch.equal(patches, F.unfold(x, 16, stride=16).transpose(1, 2)), "bitwise")
    xp = torch.randn(2, hs * ws, 256, 3)
    folded = F.fold(xp.permute(0, 3, 2, 1).reshape(2, 768, hs * ws), (height, width), kernel_size=16, stride=16)
    unpatched = xp.view(2, hs, ws, 16, 16, 3).permute(0, 5, 1, 3, 2, 4).reshape(2, 3, height, width)
    _check("V2 unpatchify vs F.fold", torch.equal(unpatched, folded), "bitwise")
    if height == width:
        up_pos = upstream_modules.get_2d_sincos_pos_embed(16, height)
        _check("V2 pixel pos embed", np.array_equal(up_pos, pixel_sincos_pos_embed(16, height, width)), "bitwise")

    # V2 full forward: upstream PixDiTTrainer vs vendored, both bf16
    checkpoint, _ = resolve_checkpoint(args.model_name, args.local_files_only)
    full_state = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=True)["state_dict"]
    upstream = PixDiTTrainer(image_size=1024, extra=cfg["model"]["extra"], caption_channels=2304, model_max_length=300)
    upstream.load_state_dict(full_state, strict=True)
    upstream = upstream.to(device).eval().requires_grad_(False).to(torch.bfloat16)
    g = torch.Generator(device=device).manual_seed(1)
    xb = torch.randn(2, 3, height, width, device=device, generator=g)
    yb = torch.randn(2, 300, 2304, device=device, generator=g)
    with torch.inference_mode():
        for tv in CHECK_TIMESTEPS + (996.0,):
            t = torch.full((2,), tv, device=device)
            dit_check("V2 DiT vendored vs upstream", tv, dit(xb, t, yb), upstream.forward_with_dpmsolver(xb, t, yb))

    # V3 text path
    prompt, negative_prompt = args.prompt, args.negative_prompt
    with torch.inference_mode():
        sel = [0] + list(range(-300 + 1, 0))
        up_tok = tokenizer([layout.chi_prompt + prompt.strip()], max_length=layout.text_seq_len, padding="max_length",
                           truncation=True, return_tensors="pt").to(device)
        up_pos = text_encoder(up_tok.input_ids, up_tok.attention_mask)[0][:, sel]
        up_neg_tok = tokenizer(negative_prompt, max_length=300, padding="max_length", truncation=True, return_tensors="pt").to(device)
        up_neg = text_encoder(up_neg_tok.input_ids, up_neg_tok.attention_mask)[0]
        my_neg, my_pos = P.encode_prompts(tokenizer, torch_encode_fn(text_encoder), layout, prompt, negative_prompt, device)
    s_pos, s_neg = _stats(my_pos, up_pos), _stats(my_neg, up_neg)
    _check("V3 positive embeds vs upstream", s_pos["bitwise"], _fmt(s_pos))
    # The 506-token trick is exact in fp32 (rel L2 3.8e-6, measured once); in bf16 it differs only by Gemma's bf16 noise.
    _check("V3 negative embeds 506 + select vs upstream 300", s_neg["cos"] >= TEXT_MIN_COS,
           f"{_fmt(s_neg)} (need cos >= {TEXT_MIN_COS})")

    # V4 sampler
    ns = NoiseScheduleFlow(schedule="discrete_flow")
    for steps in (2, 20, args.steps):
        up_t = DPM_Solver(lambda x_, t_: x_, ns).get_time_steps("time_uniform_flow", 1, 1.0 / 1000, steps, device, shift=args.flow_shift)
        _check(f"V4 timesteps steps={steps}", torch.equal(P.flow_timesteps(steps, args.flow_shift, device), up_t), "bitwise")

    def fake_model(x_, t_, y_, **_):
        return torch.sin(3 * x_) * (t_ / 1000).view(-1, 1, 1, 1) + y_.reshape(y_.shape[0], -1).mean(1).view(-1, 1, 1, 1) - 0.3 * x_

    z = torch.randn(1, 3, 64, 64, device=device, generator=g, dtype=torch.float64)
    yc = torch.randn(1, 1, 8, 4, device=device, dtype=torch.float64)
    yu = torch.randn(1, 1, 8, 4, device=device, dtype=torch.float64)
    up = DPMS(fake_model, condition=yc, uncondition=yu, cfg_scale=args.cfg_scale, model_type="flow", model_kwargs={},
              schedule="FLOW", interval_guidance=[0, 1]).sample(
        z, steps=args.steps, order=2, skip_type="time_uniform_flow", method="multistep", flow_shift=args.flow_shift)
    mine = P.FlowDPMSolver(fake_model, yu[:, 0], yc[:, 0], args.cfg_scale).sample(z, args.steps, args.flow_shift)
    s = _stats(mine, up)
    _check("V4 solver math (fake model, fp64)", s["max_abs"] < 1e-10 and torch.isfinite(mine).all().item(), _fmt(s))

    # V4 end to end: upstream DPMS + PixDiTTrainer (bf16) vs vendored pipeline
    print("  [V4] end-to-end upstream sample...")
    with torch.inference_mode():
        z = P.initial_noise(height, width, args.seed, device)
        hw = torch.tensor([[float(height), float(width)]], device=device)
        kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": (hw[:, 0] / hw[:, 1]).unsqueeze(1)}, mask=up_tok.attention_mask[:, sel])
        up_sample = DPMS(upstream.forward_with_dpmsolver, condition=up_pos[:, None], uncondition=up_neg[:, None],
                         guidance_type="classifier-free", cfg_scale=args.cfg_scale, model_type="flow", model_kwargs=kwargs,
                         schedule="FLOW", interval_guidance=[0, 1]).sample(
            z, steps=args.steps, order=2, skip_type="time_uniform_flow", method="multistep", flow_shift=args.flow_shift)
    del upstream
    torch.cuda.empty_cache()
    upstream_image = P.to_uint8_image(up_sample)
    PILImage.fromarray(upstream_image).save(str(output_dir / "upstream_output.png"))
    print(f"  [V4] Upstream image saved -> {output_dir / 'upstream_output.png'}")
    check_image_sanity("V4 end-to-end vendored vs upstream", pytorch_image(), upstream_image)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _image_size(onnx_dir: Path) -> tuple[int, int]:
    config_path = onnx_dir / "pipeline_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return int(config["height"]), int(config["width"])
    transformer = onnx_dir / "transformer" / "model.onnx"
    shape = _onnx_io_shapes(onnx.load(str(transformer), load_external_data=False))["hidden_states"]
    return shape[2], shape[3]


def main():
    p = argparse.ArgumentParser(description="Verify PixelDiT T2I ONNX graphs with PyTorch and ONNX Runtime.")
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model repo ID, local dir, or .pth")
    p.add_argument("--text_encoder_name", type=str, default=DEFAULT_TEXT_ENCODER_NAME)
    p.add_argument("--local_files_only", action="store_true", help="Resolve models from the local Hugging Face cache only")
    p.add_argument("--onnx_dir", type=str, default="./pixeldit_onnx")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--negative_prompt", type=str, default=P.DEFAULT_NEGATIVE_PROMPT)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=2.75)
    p.add_argument("--flow_shift", type=float, default=4.0)
    p.add_argument("--provider", choices=["cpu", "trt-rtx"], default="trt-rtx")
    p.add_argument("--ep_lib", default=None, help="Path to onnxruntime_providers_nv_tensorrt_rtx.dll/.so")
    p.add_argument("--ep_dll_dir", default=None, help="Dir with the plugin EP's bundled runtime DLLs")
    p.add_argument("--trt_bin", default=None, help="TensorRT-RTX bin dir, used as a DLL search path")
    p.add_argument("--validate_only", action="store_true", help="Only run ONNX structure checks and a dummy run")
    p.add_argument("--skip_e2e", action="store_true", help="Skip the end-to-end image comparison (V7)")
    p.add_argument("--reference_repo", type=str, default=None, help="Path to an NVlabs/PixelDiT clone for V2-V4 parity checks")
    args = p.parse_args()
    configure_stdio()

    onnx_dir = Path(args.onnx_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else onnx_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    height, width = _image_size(onnx_dir)
    print(f"Image size from export: {height}x{width}")

    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, local_files_only=args.local_files_only)
    hidden = AutoConfig.from_pretrained(args.text_encoder_name, local_files_only=args.local_files_only).hidden_size
    layout = P.TextLayout.from_tokenizer(tokenizer)
    expected_shapes = {
        "text_encoder": {
            "input_ids": (1, layout.text_seq_len),
            "attention_mask": (1, layout.text_seq_len),
            "select_index": (layout.txt_max_length,),
            "prompt_embeds": (1, layout.txt_max_length, hidden),
        },
        "transformer": {
            "hidden_states": (2, 3, height, width),
            "timestep": (2,),
            "encoder_hidden_states": (2, layout.txt_max_length, hidden),
            "velocity": (2, 3, height, width),
        },
    }

    runners = validate_onnx_dir(onnx_dir, args.provider, args, expected_shapes)
    if args.validate_only:
        _finish()
        return

    print(f"Loading tokenizer + Gemma from {args.text_encoder_name}...")
    tokenizer, text_encoder = load_gemma_text_encoder(args.text_encoder_name, device, args.local_files_only)

    print(f"Loading PixelDiT from {args.model_name} (bf16)...")
    dit = load_pixeldit(args.model_name, device, torch.bfloat16, args.local_files_only)
    dit.set_image_size(height, width)

    sampler = P.SamplerConfig(args.steps, args.cfg_scale, args.flow_shift, args.negative_prompt)
    encode_torch = torch_encode_fn(text_encoder)
    cache = {}

    def pytorch_image() -> np.ndarray:
        """The PyTorch-pipeline image, generated once and shared by V4 and V7."""
        if "image" not in cache:
            print("[PyTorch] Running pipeline...")
            cache["image"] = P.to_uint8_image(P.generate(tokenizer, encode_torch, dit, layout, args.prompt, height, width,
                                                         args.seed, sampler, device, _progress("PyTorch")))
            PILImage.fromarray(cache["image"]).save(str(output_dir / "pytorch_output.png"))
            print(f"[PyTorch] Image saved -> {output_dir / 'pytorch_output.png'}")
        return cache["image"]

    if args.reference_repo:
        run_reference_checks(args, tokenizer, text_encoder, layout, dit, height, width, device, pytorch_image, output_dir)

    te_runner, tr_runner = runners.get("text_encoder"), runners.get("transformer")
    if te_runner is not None:
        compare_text_encoder(tokenizer, text_encoder, te_runner, layout, args.prompt, args.negative_prompt, device)
    if tr_runner is not None:
        with torch.inference_mode():
            y_neg, y_pos = P.encode_prompts(tokenizer, encode_torch, layout, args.prompt, args.negative_prompt, device)
        compare_dit(dit, tr_runner, y_neg, y_pos, height, width, device)

    if not args.skip_e2e:
        print("\n[V7] End to end")
        pt_image = pytorch_image()
        print("[ONNX Runtime] Running pipeline...")
        encode_ort = ort_encode_fn(te_runner, device) if te_runner is not None else encode_torch
        dit_ort = ort_dit_fn(tr_runner, device) if tr_runner is not None else dit
        ort_image = P.to_uint8_image(P.generate(tokenizer, encode_ort, dit_ort, layout, args.prompt, height, width,
                                                args.seed, sampler, device, _progress("ORT")))
        PILImage.fromarray(ort_image).save(str(output_dir / "ort_output.png"))
        print(f"[ONNX Runtime] Image saved -> {output_dir / 'ort_output.png'}")
        compare_images(pt_image, ort_image, output_dir)

    _finish()


def _finish() -> None:
    if _failures:
        print(f"\nFAILED checks ({len(_failures)}): {', '.join(_failures)}")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
