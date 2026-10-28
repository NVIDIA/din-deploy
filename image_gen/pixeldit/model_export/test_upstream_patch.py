# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pixeldit/upstream_patch.py on a small random upstream PixDiT_T2I (CPU, no checkpoint needed).

Run from this directory:  python -m pytest test_upstream_patch.py   (or: python test_upstream_patch.py)
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from pixdit_core.pixeldit_t2i import PixDiT_T2I

import pixdit_core.modules as upstream_modules
import torch.nn.functional as F
from export_pixeldit import PixelDiTExportWrapper
from pixeldit.upstream_patch import export_patches

HEIGHT, WIDTH, TXT_LEN, TXT_DIM = 96, 128, 12, 64


def small_model() -> PixDiT_T2I:
    torch.manual_seed(0)
    model = PixDiT_T2I(in_channels=3, num_groups=4, hidden_size=128, pixel_hidden_size=8, pixel_attn_hidden_size=96,
                       pixel_num_groups=4, patch_depth=2, pixel_depth=2, patch_size=16, txt_embed_dim=TXT_DIM,
                       txt_max_length=TXT_LEN, repa_encoder_index=1).eval().requires_grad_(False)
    with torch.no_grad():
        for p in model.parameters():  # initialize_weights zeroes the final layer; make outputs non-trivial
            p.normal_(0, 0.05)
    return model


def inputs():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 3, HEIGHT, WIDTH, generator=g)
    t = torch.tensor([999.75, 500.0])
    y = torch.randn(2, TXT_LEN, TXT_DIM, generator=g)
    return x, t, y


def rel_l2(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


class UpstreamPatchTest(unittest.TestCase):
    def test_export_runs_and_matches(self):
        model, (x, t, y) = small_model(), inputs()
        with torch.no_grad():
            reference = model(x, t, y)  # unpatched upstream
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "dit.onnx"
                with export_patches(model, HEIGHT, WIDTH, TXT_LEN):
                    patched = model(x, t, y)
                    torch.onnx.export(PixelDiTExportWrapper(model, torch.float32).eval(), (x, t, y), str(path),
                                      dynamo=True, opset_version=25,
                                      input_names=["hidden_states", "timestep", "encoder_hidden_states"],
                                      output_names=["velocity"])
                ops = {node.op_type for node in onnx.load(str(path)).graph.node}
                session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                exported = session.run(None, {"hidden_states": x.numpy(), "timestep": t.numpy(),
                                              "encoder_hidden_states": y.numpy()})[0]
            after = model(x, t, y)

        self.assertNotIn("Col2Im", ops)
        self.assertLess(rel_l2(patched, exported), 1e-6)  # ONNX Runtime vs patched PyTorch (fp32)
        self.assertLess(rel_l2(reference, patched), 1e-6)  # patched vs upstream: RoPE rounding only
        self.assertTrue(torch.equal(reference, after))  # upstream restored exactly

    def test_everything_is_restored(self):
        model, (x, t, y) = small_model(), inputs()
        rope_fns = [block._rope_fn for block in model.pixel_blocks]
        apply_rotary_emb, fold, unfold = upstream_modules.apply_rotary_emb, F.fold, F.unfold
        with torch.no_grad(), export_patches(model, HEIGHT, WIDTH, TXT_LEN):
            self.assertIsNot(F.fold, fold)
            self.assertEqual(next(iter(model.precompute_pos.values())).dtype, torch.float32)  # real tables
        self.assertIs(upstream_modules.apply_rotary_emb, apply_rotary_emb)
        self.assertIs(F.fold, fold)
        self.assertIs(F.unfold, unfold)
        self.assertEqual([block._rope_fn for block in model.pixel_blocks], rope_fns)
        self.assertNotIn("fetch_pos_text", vars(model))  # class method again, no instance override
        self.assertEqual(len(model.precompute_pos), 0)  # upstream rebuilds its complex tables
        with torch.no_grad():
            model(x, t, y)
        self.assertTrue(next(iter(model.precompute_pos.values())).is_complex())

    def test_guard_reports_upstream_changes(self):
        model = small_model()
        del model.pixel_blocks[0]._rope_fn  # as if upstream renamed it
        with self.assertRaisesRegex(RuntimeError, "upstream PixelDiT changed: .*_rope_fn"):
            with export_patches(model, HEIGHT, WIDTH, TXT_LEN):
                pass
        self.assertEqual(upstream_modules.apply_rotary_emb.__module__, "pixdit_core.modules")  # nothing was patched

    def test_unsupported_fold_arguments_are_rejected(self):
        model = small_model()
        with export_patches(model, HEIGHT, WIDTH, TXT_LEN):
            with self.assertRaises(NotImplementedError):
                F.unfold(torch.zeros(1, 3, 32, 32), kernel_size=4, stride=2)


if __name__ == "__main__":
    unittest.main()
