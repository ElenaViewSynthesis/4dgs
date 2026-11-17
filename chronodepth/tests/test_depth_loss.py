"""Unit tests for utils/depth_loss.py on synthetic tensors (CPU only).

Run from the repository root:  python -m pytest tests/test_depth_loss.py -q
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import depth_loss as dl  # noqa: E402

H, W = 24, 32


def _pair(s=2.5, b=-0.7, seed=0):
    g = torch.Generator().manual_seed(seed)
    prior = torch.rand(H, W, generator=g)
    target = (s * prior + b).clone().requires_grad_(True)
    mask = torch.ones(H, W, dtype=torch.bool)
    return prior, target, mask


def test_alignment_recovers_known_scale_shift():
    for mode in ("cumulative", "ema"):
        al = dl.AffineAligner(mode=mode, decay=0.9)
        for seed in range(5):
            prior, target, mask = _pair(seed=seed)
            al.update(0, prior, target, mask)
        s, b, ok = al.solve(0)
        assert ok
        assert abs(s - 2.5) < 1e-4 and abs(b + 0.7) < 1e-4


def test_alignment_respects_mask():
    prior, target, mask = _pair()
    target = target.detach().clone()
    mask[:, : W // 2] = False
    target[:, : W // 2] = 123.0          # garbage outside Omega
    s, b, ok = dl.align_per_frame(prior, target, mask)
    assert ok and abs(s - 2.5) < 1e-4 and abs(b + 0.7) < 1e-4


def test_alignment_degenerate_is_flagged():
    prior = torch.full((H, W), 0.3)
    target = torch.rand(H, W)
    s, b, ok = dl.align_per_frame(prior, target, torch.ones(H, W, dtype=torch.bool))
    assert not ok and (s, b) == (1.0, 0.0)


def test_depth_loss_zero_for_affine_pair():
    prior, target, mask = _pair()
    s, b, ok = dl.align_per_frame(prior, target, mask)
    loss = dl.depth_l1_loss(prior, target, mask, s, b)
    assert ok and float(loss) < 1e-5


def test_depth_loss_alignment_is_constant_in_backprop():
    prior, target, mask = _pair()
    noisy = (target + 0.1 * torch.rand(H, W)).detach().requires_grad_(True)
    s, b, _ = dl.align_per_frame(prior, noisy, mask)
    assert isinstance(s, float) and isinstance(b, float)
    dl.depth_l1_loss(prior, noisy, mask, s, b).backward()
    assert torch.isfinite(noisy.grad).all() and noisy.grad.abs().sum() > 0


def test_pearson_invariant_to_positive_affine_maps():
    g = torch.Generator().manual_seed(1)
    prior = torch.rand(H, W, generator=g)
    target = torch.rand(H, W, generator=g)
    mask = torch.ones(H, W, dtype=torch.bool)
    base = float(dl.pearson_loss(prior, target, mask))
    for a, c in ((3.0, 0.0), (0.2, 5.0), (17.0, -4.0)):
        assert abs(float(dl.pearson_loss(a * prior + c, target, mask)) - base) < 1e-5
        assert abs(float(dl.pearson_loss(prior, a * target + c, mask)) - base) < 1e-5
    assert float(dl.pearson_loss(prior, 2 * prior + 1, mask)) < 1e-5
    assert abs(float(dl.pearson_loss(prior, -prior, mask)) - 2.0) < 1e-5


def test_temporal_loss_independent_of_shift():
    g = torch.Generator().manual_seed(2)
    p0, p1 = torch.rand(H, W, generator=g), torch.rand(H, W, generator=g)
    s = 1.7
    mask = torch.ones(H, W, dtype=torch.bool)
    vals = []
    for b in (0.0, 3.0, -11.0):
        t0, t1 = s * p0 + b, s * p1 + b
        vals.append(float(dl.temporal_loss(p0, p1, t0, t1, mask, mask, s)))
    assert max(vals) < 1e-5
    # a generic pair: adding the same shift to both rendered frames changes nothing
    t0, t1 = torch.rand(H, W, generator=g), torch.rand(H, W, generator=g)
    a = float(dl.temporal_loss(p0, p1, t0, t1, mask, mask, s))
    c = float(dl.temporal_loss(p0, p1, t0 + 9.0, t1 + 9.0, mask, mask, s))
    assert abs(a - c) < 1e-5 and a > 0


def test_temporal_loss_uses_intersection_of_omegas():
    p0, p1 = torch.zeros(H, W), torch.zeros(H, W)
    t0, t1 = torch.zeros(H, W), torch.zeros(H, W)
    t1[:, : W // 2] = 5.0
    m0 = torch.ones(H, W, dtype=torch.bool)
    m1 = m0.clone()
    m1[:, : W // 2] = False
    assert float(dl.temporal_loss(p0, p1, t0, t1, m0, m1, 1.0)) == 0.0


def test_all_terms_finite_zero_gradient_on_empty_omega():
    prior, target, _ = _pair()
    empty = torch.zeros(H, W, dtype=torch.bool)
    t1 = target.detach().clone().requires_grad_(True)
    losses = [
        dl.depth_l1_loss(prior, target, empty, 1.0, 0.0),
        dl.pearson_loss(prior, target, empty),
        dl.temporal_loss(prior, prior, target, t1, empty, empty, 1.0),
    ]
    for loss in losses:
        assert math.isfinite(float(loss)) and float(loss) == 0.0
        loss.backward()
    assert target.grad is not None and float(target.grad.abs().sum()) == 0.0
    assert float(t1.grad.abs().sum()) == 0.0
    al = dl.AffineAligner()
    al.update(0, prior, target, empty)
    assert al.solve(0)[2] is False


def test_pearson_zero_variance_is_finite():
    prior = torch.rand(H, W)
    flat = torch.full((H, W), 2.0, requires_grad=True)
    mask = torch.ones(H, W, dtype=torch.bool)
    loss = dl.pearson_loss(prior, flat, mask)
    loss.backward()
    assert float(loss) == 0.0 and float(flat.grad.abs().sum()) == 0.0


def test_render_target_spaces():
    depth = torch.full((H, W), 4.0) * 0.5      # alpha-weighted: z=4, alpha=0.5
    alpha = torch.full((H, W), 0.5)
    assert torch.allclose(dl.render_target(depth, alpha, "depth"), depth)
    assert torch.allclose(dl.render_target(depth, alpha, "disparity"), torch.full((H, W), 0.25))


def test_lambda_schedule():
    assert dl.lambda_schedule("const", 0.1, 0.0, 500, 1000) == 0.1
    assert dl.lambda_schedule("linear_decay", 0.1, 0.0, 1, 1000) == 0.1
    assert abs(dl.lambda_schedule("linear_decay", 0.1, 0.0, 1000, 1000)) < 1e-12
    mid = dl.lambda_schedule("exp_decay", 0.1, 0.001, 500, 999)
    assert abs(mid - 0.01) < 1e-3
