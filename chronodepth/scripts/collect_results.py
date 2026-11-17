"""Collect one row per run into results.csv.

    python scripts/collect_results.py <output_root> [results.csv]

Every value is read from a file the run wrote; a missing file leaves the
cell empty and is reflected in `status`. Nothing is estimated or rounded.
"""
import csv
import json
import os
import re
import sys

METRICS = ["PSNR", "SSIM", "MS-SSIM", "D-SSIM", "LPIPS-vgg", "LPIPS-alex"]
DEPTH_FLAGS = ["depth_loss", "lambda_depth", "lambda_depth_end", "lambda_depth_schedule", "lambda_temp",
               "omega_thresh", "align", "align_stats", "align_ema_decay", "depth_prior_space",
               "resolved_depth_prior_space", "resolved_depth_prior_dir", "train_every", "val_holdout"]
FIELDS = (["run_dir", "status", "scene", "seed", "seeding", "commit", "config", "iterations",
           "coarse_iterations", "batch_size", "method"] + METRICS +
          ["val_PSNR", "final_gaussians", "training_time_seconds", "wall_clock_seconds_total",
           "iter_ms_coarse", "iter_ms_fine", "render_fps_test", "nan_restarts", "depth_nonfinite_skips",
           "n_train_frames", "n_test_frames", "align_final_s", "align_final_b",
           "frac_gaussians_outside_grid_aabb"] + DEPTH_FLAGS)


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (IOError, ValueError):
        return None


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except IOError:
        return ""


def collect(run_dir, root):
    row = {"run_dir": os.path.relpath(run_dir, root).replace(os.sep, "/")}
    cfg = load(os.path.join(run_dir, "resolved_config.json")) or {}
    summ = load(os.path.join(run_dir, "train_summary.json")) or {}
    res = load(os.path.join(run_dir, "results.json")) or {}
    frames = load(os.path.join(run_dir, "train_frames.json")) or {}

    commit = read(os.path.join(run_dir, "git_commit.txt")).strip()
    if not commit or commit == "unknown":
        commit = read(os.path.join(run_dir, "git_commit_launch.txt")).split(" ")[0].strip()
    row["commit"] = commit
    row["scene"] = os.path.basename(os.path.normpath(cfg.get("source_path", ""))) if cfg else ""
    row["config"] = cfg.get("configs", "")
    row["seed"] = cfg.get("seed", "")
    row["seeding"] = cfg.get("seeding", "")
    for k in ("iterations", "coarse_iterations", "batch_size"):
        row[k] = cfg.get(k, "")
    for k in DEPTH_FLAGS:
        row[k] = cfg.get(k, "")

    if res:
        method = sorted(res.keys())[-1]
        row["method"] = method
        for m in METRICS:
            row[m] = res[method].get(m, "")
    for k in ("val_PSNR", "final_gaussians", "training_time_seconds", "wall_clock_seconds_total",
              "nan_restarts", "depth_nonfinite_skips", "align_final_s", "align_final_b",
              "frac_gaussians_outside_grid_aabb"):
        row[k] = summ.get(k, "")
    it = summ.get("mean_iter_time_ms", {})
    row["iter_ms_coarse"] = it.get("coarse", "")
    row["iter_ms_fine"] = it.get("fine", "")
    fps = re.findall(r"FPS:\s*([0-9.eE+-]+)", read(os.path.join(run_dir, "render_stdout.txt")))
    row["render_fps_test"] = fps[0] if fps else ""       # first FPS line = test set (--skip_train)
    row["n_train_frames"] = len(frames.get("train_frame_ids", [])) or ""
    row["n_test_frames"] = len(frames.get("test_frame_ids", [])) or ""

    restarts = summ.get("nan_restarts", 0) or 0
    if not res:
        row["status"] = "failed"
    elif restarts:
        row["status"] = "nan_restart"
    else:
        row["status"] = "ok"
    return row


def main():
    root = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results.csv"
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "cfg_args" in filenames or "train_stdout.txt" in filenames:
            rows.append(collect(dirpath, root))
            dirnames[:] = []
    rows.sort(key=lambda r: r["run_dir"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print("{} runs -> {}".format(len(rows), out))


if __name__ == "__main__":
    main()
