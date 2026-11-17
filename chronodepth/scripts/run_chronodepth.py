"""ChronoDepth priors for one HyperNeRF scene (run in the depth environment,
not in the Python 3.7 training environment).

Thin wrapper around the official inference code: it imports `run_pipeline`
and `seed_all` from the official `run_infer.py` and builds the pipeline the
same way that script does. The only differences are the input (the scene's
frames in temporal order instead of an mp4) and the output (one float32
.npy per frame).

    python scripts/run_chronodepth.py --datadir data/hypernerf/misc/americano \
        --chronodepth_repo /path/to/ChronoDepth --seed 1234

Writes  <datadir>/depth/2x/{frame_id}.npy          (H x W float32, RGB resolution)
        <datadir>/depth/2x/chronodepth_settings.json
        <datadir>/depth/2x/preview/*.png            (8 evenly spaced frames)
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image


def load_frames(datadir, factor):
    with open(os.path.join(datadir, "dataset.json")) as f:
        ids = json.load(f)["ids"]          # temporal order, train and test interleaved
    frames = [np.array(Image.open(os.path.join(datadir, "rgb", f"{factor}x", f"{i}.png")).convert("RGB"))
              for i in ids]
    return ids, np.stack(frames, 0)


def repo_commit(path):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
    except Exception:
        return "unknown"


def hub_revision(name):
    try:
        from huggingface_hub import model_info
        return model_info(name).sha
    except Exception:
        return "unknown"


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def prior_space_evidence(datadir, ids, depth, factor, n_frames=24):
    """Is the output depth-like or disparity-like, and which way does it increase?

    Neither the paper nor the inference code states it, so it is measured:
    the dataset's SfM points (points.npy, static scene points) are projected
    into evenly spaced frames with a pinhole model (lens distortion ignored,
    occlusion not handled) and the prior at those pixels is correlated with
    the points' camera-space depth z and with 1/z.
    """
    pts_file = os.path.join(datadir, "points.npy")
    if not os.path.exists(pts_file):
        return None
    pts = np.load(pts_file).astype(np.float64)
    T, H, W = depth.shape
    r_z, r_inv, used = [], [], []
    for t in np.linspace(0, T - 1, min(n_frames, T)).round().astype(int):
        with open(os.path.join(datadir, "camera", f"{ids[t]}.json")) as f:
            cam = json.load(f)
        R = np.array(cam["orientation"])
        local = (pts - np.array(cam["position"])) @ R.T
        z = local[:, 2]
        front = z > 1e-6
        focal = cam["focal_length"] / factor
        cx, cy = np.array(cam["principal_point"]) / factor
        x = local[front, 0] / z[front] * focal + cx
        y = local[front, 1] / z[front] * focal * cam.get("pixel_aspect_ratio", 1.0) + cy
        zi = z[front]
        inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
        if inside.sum() < 50:
            continue
        d = depth[t][y[inside].round().astype(int), x[inside].round().astype(int)].astype(np.float64)
        zi = zi[inside]
        r_z.append(pearson(d, zi))
        r_inv.append(pearson(d, 1.0 / zi))
        used.append(int(t))
    if not r_z:
        return None
    m_z, m_inv = float(np.nanmedian(r_z)), float(np.nanmedian(r_inv))
    return {
        "method": "Pearson r between the prior and SfM point depth z (and 1/z) at projected "
                  "points.npy locations; pinhole projection, no distortion, no occlusion test; "
                  "median over frames",
        "frames_used": used,
        "median_r_prior_vs_z": m_z,
        "median_r_prior_vs_inverse_z": m_inv,
        "per_frame_r_prior_vs_z": r_z,
        "per_frame_r_prior_vs_inverse_z": r_inv,
    }


def log_to_wandb(settings, scene, ids, video, depth, colored, picks, out_dir):
    """Track the prediction maps and every inference setting in W&B.

    Only if WANDB_API_KEY is in the environment; the key is never written anywhere.
    Logged: the settings as run config, the 8 preview frames (RGB | depth), a depth
    video of the whole sequence, per-frame statistics, and the settings file plus
    previews as an artifact (the .npy maps stay on disk).
    """
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set: depth maps are logged to local files only")
        return
    try:
        import wandb
        config = {k: v for k, v in settings.items() if k != "prior_space_evidence"}
        config["scene"] = scene
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "chronodepth-reg"),
                         name="chronodepth-prior/" + scene, job_type="depth-prior", config=config)
        run.log({"preview": [wandb.Image(np.concatenate([video[t], colored[k]], axis=1),
                                         caption="frame {} (index {})".format(ids[t], int(t)))
                             for k, t in enumerate(picks)]})
        # whole sequence at reduced size: every 2nd frame, half resolution, Spectral colour map
        try:
            from src.utils.video_utils import colorize_video_depth
            small = colorize_video_depth(depth[::2, ::2, ::2])
            run.log({"depth_video": wandb.Video(small.transpose(0, 3, 1, 2), fps=15, format="mp4")})
        except Exception as e:
            print("W&B depth video skipped:", type(e).__name__, str(e)[:200])
        for t in range(depth.shape[0]):
            row = {"frame_index": t, "prior/mean": float(depth[t].mean()), "prior/std": float(depth[t].std()),
                   "prior/min": float(depth[t].min()), "prior/max": float(depth[t].max())}
            if t > 0:
                row["prior/mean_abs_change_from_previous_frame"] = float(np.abs(depth[t] - depth[t - 1]).mean())
            run.log(row)
        evidence = settings.get("prior_space_evidence") or {}
        for key in ("median_r_prior_vs_z", "median_r_prior_vs_inverse_z"):
            if key in evidence:
                run.summary[key] = evidence[key]
        artifact = wandb.Artifact("chronodepth-prior-" + scene, type="depth-prior", metadata=config)
        artifact.add_file(os.path.join(out_dir, "chronodepth_settings.json"))
        artifact.add_dir(os.path.join(out_dir, "preview"), name="preview")
        run.log_artifact(artifact)
        run.finish()
    except Exception as e:      # tracking must never lose the computed maps
        print("W&B logging failed:", type(e).__name__, str(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--chronodepth_repo", required=True, help="checkout of the official repository")
    ap.add_argument("--repo_url", default="https://github.com/jhaoshao/ChronoDepth")
    ap.add_argument("--factor", type=int, default=2, help="image downsampling used by the loader")
    ap.add_argument("--unet", default="jhshao/ChronoDepth-v1")
    ap.add_argument("--model_base", default="stabilityai/stable-video-diffusion-img2vid-xt")
    # the defaults below are the official script's defaults
    ap.add_argument("--denoise_steps", type=int, default=5)
    ap.add_argument("--n_tokens", type=int, default=10, help="window length F")
    ap.add_argument("--chunk_size", type=int, default=5, help="new frames per window; overlap = F - chunk_size")
    ap.add_argument("--sigma_epsilon", type=float, default=-4.0)
    ap.add_argument("--max_res", type=int, default=1024)
    ap.add_argument("--decode_chunk_size", type=int, default=8)
    ap.add_argument("--infer_mode", default="ours")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--scene_name", default="")
    ap.add_argument("--wandb_only", action="store_true",
                    help="do not run inference; log the maps already on disk to W&B")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.chronodepth_repo))
    if args.wandb_only:
        from src.utils.video_utils import colorize_video_depth
        ids, video = load_frames(args.datadir, args.factor)
        out_dir = os.path.join(args.datadir, "depth", f"{args.factor}x")
        with open(os.path.join(out_dir, "chronodepth_settings.json")) as f:
            settings = json.load(f)
        depth = np.stack([np.load(os.path.join(out_dir, f"{i}.npy")) for i in ids], 0)
        picks = np.linspace(0, len(ids) - 1, 8).round().astype(int)
        log_to_wandb(settings, args.scene_name or os.path.basename(os.path.normpath(args.datadir)),
                     ids, video, depth, colorize_video_depth(depth[picks]), picks, out_dir)
        return
    import torch
    from easydict import EasyDict
    from run_infer import run_pipeline, seed_all                      # official code
    from src.utils.video_utils import resize_max_res, colorize_video_depth
    from chronodepth.unet_chronodepth import DiffusersUNetSpatioTemporalConditionModelChronodepth
    from chronodepth.chronodepth_pipeline import ChronoDepthPipeline

    ids, video = load_frames(args.datadir, args.factor)
    T, H, W, _ = video.shape
    infer_h, infer_w = resize_max_res(video[:1], args.max_res).shape[1:3]
    print(f"{T} frames, {H}x{W} -> inference at {infer_h}x{infer_w}")

    device = torch.device("cuda")
    seed_all(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    unet = DiffusersUNetSpatioTemporalConditionModelChronodepth.from_pretrained(
        args.unet, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    pipeline = ChronoDepthPipeline.from_pretrained(
        args.model_base, unet=unet, torch_dtype=torch.float16, variant="fp16")
    pipeline.n_tokens = args.n_tokens
    pipeline.chunk_size = args.chunk_size
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        xformers = True
    except Exception:
        xformers = False
    pipeline.to(device)

    cfg = EasyDict(vars(args))
    t0 = time.time()
    depth = run_pipeline(pipeline, cfg, video, generator, device).float().cpu().numpy()   # [T,H,W] in [0,1]
    seconds = time.time() - t0
    assert depth.shape == (T, H, W), depth.shape

    out_dir = os.path.join(args.datadir, "depth", f"{args.factor}x")
    os.makedirs(os.path.join(out_dir, "preview"), exist_ok=True)
    for i, d in zip(ids, depth):
        np.save(os.path.join(out_dir, f"{i}.npy"), d.astype(np.float32))

    picks = np.linspace(0, T - 1, 8).round().astype(int)
    colored = colorize_video_depth(depth[picks])
    for k, t in enumerate(picks):
        Image.fromarray(np.concatenate([video[t], colored[k]], axis=1)).save(
            os.path.join(out_dir, "preview", f"{k}_{ids[t]}.png"))

    evidence = prior_space_evidence(args.datadir, ids, depth, args.factor)
    if evidence is None:
        space, larger = "undetermined", "undetermined"
    else:
        m_z, m_inv = evidence["median_r_prior_vs_z"], evidence["median_r_prior_vs_inverse_z"]
        space = "depth" if abs(m_z) >= abs(m_inv) else "disparity"
        larger = "farther" if m_z > 0 else "nearer"

    settings = {
        "estimator": "ChronoDepth",
        "repo_url": args.repo_url,
        "repo_commit": repo_commit(args.chronodepth_repo),
        "checkpoint": args.unet,
        "checkpoint_revision": hub_revision(args.unet),
        "model_base": args.model_base,
        "model_base_revision": hub_revision(args.model_base),
        "seed": args.seed,
        "window_length_F": args.n_tokens,
        "new_frames_per_window": args.chunk_size,
        "window_overlap": args.n_tokens - args.chunk_size,
        "denoise_steps": args.denoise_steps,
        "sigma_epsilon": args.sigma_epsilon,
        "infer_mode": args.infer_mode,
        "max_res": args.max_res,
        "inference_resolution_hw": [int(infer_h), int(infer_w)],
        "output_resolution_hw": [int(H), int(W)],
        "decode_chunk_size": args.decode_chunk_size,
        "fp16": True,
        "xformers": xformers,
        "num_frames": int(T),
        "frames": "all frames of dataset.json['ids'] in order (train and test interleaved)",
        "output_range": "clipped to [0,1] by the official pipeline; affine-invariant",
        "output_min": float(depth.min()),
        "output_max": float(depth.max()),
        "fraction_at_clip_bounds": float(((depth <= 0) | (depth >= 1)).mean()),
        "prior_space": space,
        "larger_is": larger,
        "prior_space_evidence": evidence,
        "prior_space_note": "Not stated in the ChronoDepth paper or code; decided by the measurement in "
                            "prior_space_evidence. 'depth': the map is affine in depth. 'disparity': "
                            "the map is affine in inverse depth.",
        "inference_seconds": seconds,
        "torch": torch.__version__,
    }
    with open(os.path.join(out_dir, "chronodepth_settings.json"), "w") as f:
        json.dump(settings, f, indent=1)
    log_to_wandb(settings, args.scene_name or os.path.basename(os.path.normpath(args.datadir)),
                 ids, video, depth, colored, picks, out_dir)
    print(json.dumps({k: settings[k] for k in ("prior_space", "larger_is", "inference_resolution_hw",
                                                "inference_seconds", "fraction_at_clip_bounds")}))


if __name__ == "__main__":
    main()
