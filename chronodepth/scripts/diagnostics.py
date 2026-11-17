"""Diagnostics for one regularised run (training environment, needs a GPU).

    python scripts/diagnostics.py --model_path <regularised run> --baseline_path <baseline run> \
        --configs <the regularised run's config>

Writes under <model_path>/diagnostics/:
    curves.png           s, b and every loss term over iterations (from train_log.csv)
    depth_<k>.png        test frame | rendered depth | aligned prior depth  (6 frames)
    rgb_<k>.png          ground truth | baseline render | regularised render (same frames)
    diagnostics.json     frames used, alignment constants, colour ranges
Rendered images are read from the runs' own test/ours_*/renders folders.
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene
from utils import depth_loss as dl


def curves(model_path, out_dir):
    rows = list(csv.DictReader(open(os.path.join(model_path, "train_log.csv"))))
    coarse_n = sum(1 for r in rows if r["stage"] == "coarse")
    x = [int(r["iteration"]) + (coarse_n if r["stage"] == "fine" else 0) for r in rows]
    keys = ["s", "b", "l1", "depth_unweighted", "depth_weighted", "temp_unweighted", "temp_weighted", "omega_frac"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for ax, k in zip(axes.ravel(), keys):
        pts = [(xi, float(r[k])) for xi, r in zip(x, rows) if r.get(k, "") != ""]
        if pts:
            ax.plot(*zip(*pts), lw=0.6)
        ax.axvline(coarse_n, color="gray", ls="--", lw=0.8)
        ax.set_title(k)
        ax.set_xlabel("iteration (coarse then fine)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=110)
    plt.close(fig)


def renders_dir(path):
    hits = sorted(glob.glob(os.path.join(path, "test", "ours_*", "renders")))
    return hits[-1] if hits else None


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--baseline_path", type=str, default="")
    parser.add_argument("--n_frames", type=int, default=6)
    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    out_dir = os.path.join(args.model_path, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    curves(args.model_path, out_dir)

    summary = json.load(open(os.path.join(args.model_path, "train_summary.json")))
    resolved = json.load(open(os.path.join(args.model_path, "resolved_config.json")))
    s, b = summary.get("align_final_s"), summary.get("align_final_b")
    space = resolved.get("resolved_depth_prior_space") or "depth"
    prior_dir = resolved.get("resolved_depth_prior_dir")

    with torch.no_grad():
        dataset = model.extract(args)
        gaussians = GaussianModel(dataset.sh_degree, hyperparam.extract(args))
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        test = scene.getTestCameras()
        data = test.dataset
        picks = np.linspace(0, len(test) - 1, args.n_frames).round().astype(int)
        bg = torch.zeros(3, device="cuda")
        info = {"frames": [], "align_s": s, "align_b": b, "prior_space": space,
                "note": "the prior of a test frame never entered training; it is shown with the final "
                        "per-camera (s, b) of the training frames"}
        ours_dir, base_dir = renders_dir(args.model_path), renders_dir(args.baseline_path) if args.baseline_path else None
        for k, idx in enumerate(picks):
            cam = test[int(idx)]
            pkg = render(cam, gaussians, pipeline.extract(args), bg, stage="fine", cam_type=scene.dataset_type,
                         return_depth_alpha=True)
            target = dl.render_target(pkg["depth_diff"], pkg["alpha"], space).cpu().numpy()
            alpha = pkg["alpha"].cpu().numpy()
            frame_id = data.all_img_origin[int(data.i_test[int(idx)])]
            panels = [("test frame " + frame_id, cam.original_image.permute(1, 2, 0).cpu().numpy(), None)]
            vmin, vmax = np.percentile(target[alpha > 0.5], [2, 98]) if (alpha > 0.5).any() else (0, 1)
            panels.append(("rendered ({})".format(space), target, (vmin, vmax)))
            prior_file = os.path.join(prior_dir, frame_id + ".npy") if prior_dir else ""
            if prior_file and os.path.exists(prior_file) and s is not None:
                panels.append(("aligned prior s*d+b", s * np.load(prior_file) + b, (vmin, vmax)))
            fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
            for ax, (title, img, rng) in zip(np.atleast_1d(axes), panels):
                if rng is None:
                    ax.imshow(np.clip(img, 0, 1))
                else:
                    ax.imshow(img, cmap="Spectral", vmin=rng[0], vmax=rng[1])
                ax.set_title(title)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "depth_{}.png".format(k)), dpi=110)
            plt.close(fig)

            name = "{0:05d}.png".format(int(idx))
            rgb = [np.array(Image.open(os.path.join(os.path.dirname(ours_dir), "gt", name)))] if ours_dir else []
            for d in (base_dir, ours_dir):
                if d and os.path.exists(os.path.join(d, name)):
                    rgb.append(np.array(Image.open(os.path.join(d, name))))
            if rgb:
                Image.fromarray(np.concatenate([r[..., :3] for r in rgb], axis=1)).save(
                    os.path.join(out_dir, "rgb_{}.png".format(k)))
            info["frames"].append({"test_index": int(idx), "frame_id": frame_id,
                                   "colour_range": [float(vmin), float(vmax)]})
        info["rgb_panels"] = "ground truth | baseline | regularised"
    with open(os.path.join(out_dir, "diagnostics.json"), "w") as f:
        json.dump(info, f, indent=1)
    print("diagnostics ->", out_dir)


if __name__ == "__main__":
    main()
