"""Video-depth regulariser (paper Sec. 4.3): alignment, L_depth, L_corr, L_temp.

All functions are pure PyTorch and run on CPU, so they can be unit-tested
without the rasteriser (tests/test_depth_loss.py).

Notation
    prior   d   the estimator's map for one frame, HxW (affine-invariant)
    target  r   the rendered quantity the prior is affine-related to, HxW
    mask    Omega, boolean HxW

`render_target` decides what r is. If the estimator predicts depth, r is the
rendered depth. If it predicts disparity, the affine relation holds against
inverse depth, so r is the alpha-normalised inverse rendered depth.
"""
import math

import torch

EPS = 1e-8


def _zero(*tensors):
    """Finite zero that stays attached to the graph (zero gradient)."""
    out = None
    for t in tensors:
        if torch.is_tensor(t) and t.requires_grad:
            z = t.sum() * 0.0
            out = z if out is None else out + z
    if out is None:
        ref = next(t for t in tensors if torch.is_tensor(t))
        out = torch.zeros((), dtype=ref.dtype, device=ref.device)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def render_target(depth, alpha, space="depth", eps=1e-6):
    """Rendered quantity that the prior is assumed affine-related to.

    depth: sum_i z_i alpha_i T_i (not normalised by accumulated alpha, Eq. depth)
    alpha: accumulated opacity sum_i alpha_i T_i
    """
    if space == "depth":
        return depth
    if space == "disparity":
        # 1 / (depth / alpha): inverse of the alpha-normalised expected depth
        return alpha / depth.clamp_min(eps)
    raise ValueError(f"unknown prior space: {space}")


def omega_mask(alpha, thresh, prior=None, target=None):
    """Omega: pixels with accumulated opacity above `thresh` (and finite inputs)."""
    mask = alpha.detach() > thresh
    if prior is not None:
        mask = mask & torch.isfinite(prior)
    if target is not None:
        mask = mask & torch.isfinite(target.detach())
    return mask


def sufficient_stats(prior, target, mask):
    """[N, sum d, sum d^2, sum D, sum d*D] over the mask, float64, detached."""
    d = prior.detach()[mask].double()
    r = target.detach()[mask].double()
    return torch.stack([
        torch.tensor(float(d.numel()), dtype=torch.float64, device=d.device),
        d.sum(), (d * d).sum(), r.sum(), (d * r).sum()])


def solve_affine(stats, min_pixels=16, rel_tol=1e-10):
    """Closed-form least squares for (s, b) in s*d + b ~ D (Eq. align).

    Returns (s, b, ok). ok is False when the 2x2 system is degenerate
    (too few pixels or zero variance of the prior); then (s, b) = (1, 0)
    and the caller must not use the alignment.
    """
    n, sd, sdd, sr, sdr = [float(x) for x in stats]
    if not all(math.isfinite(v) for v in (n, sd, sdd, sr, sdr)) or n < min_pixels:
        return 1.0, 0.0, False
    det = n * sdd - sd * sd          # = n^2 * Var(d)
    if det <= rel_tol * max(n * sdd, EPS):
        return 1.0, 0.0, False
    s = (n * sdr - sd * sr) / det
    b = (sr - s * sd) / n
    if not (math.isfinite(s) and math.isfinite(b)):
        return 1.0, 0.0, False
    return s, b, True


class AffineAligner:
    """Running per-camera alignment from sufficient statistics.

    mode "cumulative": statistics of every frame sampled so far are summed
                       (literal reading of Eq. align).
    mode "ema":        statistics are decayed by `decay` before each update,
                       so frames rendered early in training fade out.
    """

    def __init__(self, mode="cumulative", decay=0.99):
        if mode not in ("cumulative", "ema"):
            raise ValueError(f"unknown align_stats mode: {mode}")
        self.mode = mode
        self.decay = decay
        self.stats = {}

    def update(self, key, prior, target, mask):
        new = sufficient_stats(prior, target, mask).cpu()
        if not torch.isfinite(new).all() or new[0] == 0:
            return
        old = self.stats.get(key)
        if old is None:
            self.stats[key] = new
        elif self.mode == "ema":
            self.stats[key] = self.decay * old + new
        else:
            self.stats[key] = old + new

    def solve(self, key):
        if key not in self.stats:
            return 1.0, 0.0, False
        return solve_affine(self.stats[key])

    def state_dict(self):
        return {str(k): v.tolist() for k, v in self.stats.items()}


def align_per_frame(prior, target, mask):
    """Separate (s, b) for one frame (ablation of the per-camera assumption)."""
    if int(mask.sum()) == 0:
        return 1.0, 0.0, False
    return solve_affine(sufficient_stats(prior, target, mask).cpu())


def depth_l1_loss(prior, target, mask, s, b):
    """L_depth (Eq. ldepth): mean |s*d + b - D| over Omega; s, b are constants."""
    if int(mask.sum()) == 0:
        return _zero(target)
    diff = (float(s) * prior[mask] + float(b)) - target[mask]
    return diff.abs().mean()


def pearson_loss(prior, target, mask, sign=1.0):
    """L_corr (Eq. lcorr): 1 - Pearson(D, sign*d) over Omega, per frame."""
    if int(mask.sum()) < 2:
        return _zero(target)
    d = float(sign) * prior[mask]
    r = target[mask]
    d = d - d.mean()
    r = r - r.mean()
    var_d = (d * d).mean()
    var_r = (r * r).mean()
    if float(var_d) <= EPS or float(var_r.detach()) <= EPS:
        return _zero(target)
    return 1.0 - (d * r).mean() / torch.sqrt(var_d * var_r)


def temporal_loss(prior_t, prior_t1, target_t, target_t1, mask_t, mask_t1, s):
    """L_temp (Eq. ltemp): L1 between rendered change and s * prior change
    over Omega_t ∩ Omega_{t+1}. The shift b cancels."""
    mask = mask_t & mask_t1
    if int(mask.sum()) == 0:
        return _zero(target_t, target_t1)
    d_render = target_t1[mask] - target_t[mask]
    d_prior = prior_t1[mask] - prior_t[mask]
    return (d_render - float(s) * d_prior).abs().mean()


def lambda_schedule(kind, start, end, k, total):
    """lambda_d(k) for global iteration k in [1, total]."""
    if kind == "const" or total <= 1:
        return float(start)
    frac = min(max((k - 1) / float(total - 1), 0.0), 1.0)
    if kind == "linear_decay":
        return float(start + (end - start) * frac)
    if kind == "exp_decay":
        lo = max(float(end), 1e-12)
        if start <= 0:
            return 0.0
        return float(start * (lo / start) ** frac)
    raise ValueError(f"unknown lambda_depth_schedule: {kind}")
