"""Is the depth prior affine in depth or in inverse depth? (training environment, GPU)

Uses a trained baseline model (no depth term) as the reference geometry:

    python scripts/check_prior_space.py --model_path <baseline run> --configs <its config> \
        --prior_dir <datadir>/depth/2x

For evenly spaced TRAINING frames it renders depth and accumulated opacity,
and over pixels with opacity > 0.9 compares the prior with
    (a) the alpha-normalised rendered depth          z = D / A
    (b) the inverse of it                            1 / z
Two fits are reported for each, per frame and pooled over all frames with one
(s, b): the Pearson correlation and the relative residual
    mean|s d + b - y| / std(y)
The space with the higher pooled |r| (lower residual) is the one in which the
affine relation of Eq. (align) holds better. Writes <prior_dir>/prior_space_check.json.
"""
import json
import os
import sys

import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene


def fit(d, y):
    """Least-squares s, b for s*d + b ~ y; returns r and relative L1 residual."""
    d = d.double()
    y = y.double()
    dm, ym = d.mean(), y.mean()
    var_d = ((d - dm) ** 2).mean()
    cov = ((d - dm) * (y - ym)).mean()
    s = cov / var_d
    b = ym - s * dm
    r = cov / torch.sqrt(var_d * ((y - ym) ** 2).mean())
    resid = (s * d + b - y).abs().mean() / y.std()
    return float(s), float(b), float(r), float(resid)


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--prior_dir", type=str, required=True)
    parser.add_argument("--n_frames", type=int, default=32)
    parser.add_argument("--omega_thresh", type=float, default=0.9)
    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, mmcv.Config.fromfile(args.configs))

    with torch.no_grad():
        dataset = model.extract(args)
        gaussians = GaussianModel(dataset.sh_degree, hyperparam.extract(args))
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        train = scene.getTrainCameras()
        data = train.dataset
        bg = torch.zeros(3, device="cuda")
        picks = np.linspace(0, len(train) - 1, min(args.n_frames, len(train))).round().astype(int)
        per_frame, pool_d, pool_z = [], [], []
        for idx in picks:
            cam = train[int(idx)]
            pkg = render(cam, gaussians, pipeline.extract(args), bg, stage="fine", cam_type=scene.dataset_type,
                         return_depth_alpha=True)
            alpha = pkg["alpha"]
            mask = alpha > args.omega_thresh
            frame_id = data.all_img_origin[int(data.i_train[int(idx)])]
            prior = torch.from_numpy(np.load(os.path.join(args.prior_dir, frame_id + ".npy"))).cuda()
            if int(mask.sum()) < 1000:
                continue
            z = (pkg["depth_diff"] / alpha.clamp_min(1e-6))[mask]
            d = prior[mask]
            fz, fi = fit(d, z), fit(d, 1.0 / z)
            per_frame.append({"frame_id": frame_id, "omega_frac": float(mask.float().mean()),
                              "depth": {"s": fz[0], "b": fz[1], "r": fz[2], "rel_l1": fz[3]},
                              "inverse_depth": {"s": fi[0], "b": fi[1], "r": fi[2], "rel_l1": fi[3]}})
            keep = torch.arange(0, d.numel(), 16, device=d.device)      # deterministic subsample for pooling
            pool_d.append(d[keep])
            pool_z.append(z[keep])
        d_all, z_all = torch.cat(pool_d), torch.cat(pool_z)
        pz, pi = fit(d_all, z_all), fit(d_all, 1.0 / z_all)

    med = lambda key, k: float(np.median([f[key][k] for f in per_frame]))
    out = {
        "definition": __doc__.split("For evenly")[1].split("Writes")[0].strip(),
        "reference_model": os.path.basename(os.path.normpath(args.model_path)),
        "n_frames": len(per_frame), "omega_thresh": args.omega_thresh,
        "pooled_single_scale_shift": {
            "depth": {"s": pz[0], "b": pz[1], "r": pz[2], "rel_l1": pz[3]},
            "inverse_depth": {"s": pi[0], "b": pi[1], "r": pi[2], "rel_l1": pi[3]}},
        "per_frame_median": {
            "depth": {"r": med("depth", "r"), "rel_l1": med("depth", "rel_l1")},
            "inverse_depth": {"r": med("inverse_depth", "r"), "rel_l1": med("inverse_depth", "rel_l1")}},
        "per_frame_scale_spread": {
            "depth_s_min_max": [min(f["depth"]["s"] for f in per_frame), max(f["depth"]["s"] for f in per_frame)],
            "inverse_depth_s_min_max": [min(f["inverse_depth"]["s"] for f in per_frame),
                                        max(f["inverse_depth"]["s"] for f in per_frame)]},
        "per_frame": per_frame,
    }
    better = "depth" if abs(pz[2]) >= abs(pi[2]) else "disparity"
    out["better_pooled_space"] = better
    out["larger_is"] = "farther" if pz[2] > 0 else "nearer"
    with open(os.path.join(args.prior_dir, "prior_space_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("pooled_single_scale_shift", "per_frame_median",
                                           "per_frame_scale_spread", "better_pooled_space", "larger_is")}, indent=1))


if __name__ == "__main__":
    main()
