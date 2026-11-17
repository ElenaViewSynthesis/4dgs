"""Per-run provenance and logging for the depth-regulariser experiments.

Everything is written under <model_path>/. Credentials are never written:
W&B is used only if WANDB_API_KEY is present in the environment.
"""
import csv
import json
import os
import subprocess
import time

CSV_FIELDS = [
    "iteration", "stage", "l1", "loss_total",
    "depth_unweighted", "depth_weighted", "lambda_depth",
    "temp_unweighted", "temp_weighted", "lambda_temp",
    "s", "b", "align_ok", "omega_frac",
    "gaussians", "iter_time_ms",
]

MAX_NAN_RESTARTS = 5   # only enforced when a depth term is active


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


def git_commit(repo_dir):
    """Commit hash of the code that is running, '+dirty' if the tree is modified."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                                       stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
        return head + ("+dirty" if dirty else "")
    except Exception:
        # code shipped without .git (e.g. into a container): the launcher passes the hash
        return os.environ.get("GIT_COMMIT", "unknown")


class RunLogger:
    def __init__(self, model_path, repo_dir):
        self.model_path = model_path
        self.repo_dir = repo_dir
        os.makedirs(model_path, exist_ok=True)
        self._csv_file = None
        self._csv = None
        self._wandb = None
        self._t0 = time.time()
        self.depth_nonfinite_skips = 0
        self.iter_time_sum = {}
        self.iter_time_n = {}

    # ---- provenance -------------------------------------------------
    def write_provenance(self, args, extra=None):
        resolved = {k: _jsonable(v) for k, v in sorted(vars(args).items())}
        if extra:
            resolved.update({k: _jsonable(v) for k, v in extra.items()})
        with open(os.path.join(self.model_path, "resolved_config.json"), "w") as f:
            json.dump(resolved, f, indent=1)
        with open(os.path.join(self.model_path, "git_commit.txt"), "w") as f:
            f.write(git_commit(self.repo_dir) + "\n")
        self._resolved = resolved

    def write_train_frames(self, scene):
        out = {}
        for name, cams in (("train", scene.getTrainCameras()), ("val", scene.getValCameras()),
                           ("test", scene.getTestCameras())):
            data = getattr(cams, "dataset", None)
            if data is None or not hasattr(data, "all_img_origin"):
                continue
            idx = {"train": data.i_train, "val": data.i_val, "test": data.i_test}[name]
            out[name + "_frame_ids"] = [data.all_img_origin[int(i)] for i in idx]
            out[name + "_frame_indices"] = [int(i) for i in idx]
        with open(os.path.join(self.model_path, "train_frames.json"), "w") as f:
            json.dump(out, f, indent=1)

    # ---- NaN restarts -----------------------------------------------
    @property
    def _nan_file(self):
        return os.path.join(self.model_path, "nan_restarts.txt")

    def nan_restarts(self):
        try:
            with open(self._nan_file) as f:
                return int(f.read().strip() or 0)
        except (IOError, ValueError):
            return 0

    def count_nan_restart(self):
        n = self.nan_restarts() + 1
        with open(self._nan_file, "w") as f:
            f.write(str(n))
        return n

    def reset_nan_restarts_if_fresh(self):
        """Called by the launcher-independent entry point: a run that is not a
        re-exec starts from zero. A re-exec is recognised by the env marker."""
        if os.environ.get("FDGS_NAN_REEXEC") != "1" and os.path.exists(self._nan_file):
            os.remove(self._nan_file)

    # ---- per-iteration log ------------------------------------------
    def open_csv(self):
        self._csv_file = open(os.path.join(self.model_path, "train_log.csv"), "w", newline="")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
        self._csv.writeheader()

    def log_iteration(self, row):
        if self._csv is None:
            self.open_csv()
        self._csv.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        stage = row["stage"]
        self.iter_time_sum[stage] = self.iter_time_sum.get(stage, 0.0) + float(row["iter_time_ms"])
        self.iter_time_n[stage] = self.iter_time_n.get(stage, 0) + 1
        if row["iteration"] % 500 == 0:
            self._csv_file.flush()
        if self._wandb is not None and row["iteration"] % 50 == 0:
            self._wandb.log({"{}/{}".format(stage, k): v for k, v in row.items()
                             if k not in ("stage", "iteration") and v != ""})

    # ---- optional W&B -----------------------------------------------
    def maybe_init_wandb(self, run_name):
        if not os.environ.get("WANDB_API_KEY"):
            return
        try:
            import wandb
            self._wandb = wandb.init(project=os.environ.get("WANDB_PROJECT", "chronodepth-reg"),
                                     name=run_name, config=self._resolved, reinit=True)
        except Exception as e:  # logging must never stop a run
            print("W&B disabled:", type(e).__name__)
            self._wandb = None

    @property
    def wandb_active(self):
        return self._wandb is not None

    def log_depth_panel(self, key, rgb, target, aligned_prior, mask, step_label):
        """W&B image: training frame | rendered target | aligned prior | Omega (numpy arrays)."""
        if self._wandb is None:
            return
        try:
            import numpy as np
            import wandb
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import cm
            valid = target[mask] if mask.any() else target.ravel()
            lo, hi = np.percentile(valid, [2, 98])
            norm = lambda x: np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
            panel = np.concatenate([np.clip(rgb, 0, 1),
                                    cm.get_cmap("Spectral")(norm(target))[..., :3],
                                    cm.get_cmap("Spectral")(norm(aligned_prior))[..., :3],
                                    np.repeat(mask[..., None].astype(np.float32), 3, axis=2)], axis=1)
            self._wandb.log({key: wandb.Image((panel * 255).astype(np.uint8),
                             caption="frame | rendered | aligned prior | Omega  ({})".format(step_label))})
        except Exception as e:
            print("W&B image logging failed:", type(e).__name__)

    def log_gaussians(self, gaussians, stage, iteration, with_cloud=False, max_points=100000):
        """Track the Gaussian splats in W&B: count, opacity/scale statistics and
        histograms, canonical-centre extent, share outside the deformation grid;
        with_cloud adds a coloured 3D point cloud of the canonical centres.
        Read-only and free of random numbers, so it cannot change training."""
        if self._wandb is None:
            return
        try:
            import numpy as np
            import torch
            import wandb
            with torch.no_grad():
                xyz = gaussians.get_xyz
                opacity = gaussians.get_opacity.flatten()
                scale = gaussians.get_scaling
                aabb = gaussians._deformation.deformation_net.grid.aabb.to(xyz.device)
                lo, hi = torch.min(aabb[0], aabb[1]), torch.max(aabb[0], aabb[1])
                log = {
                    "splats/count": int(xyz.shape[0]),
                    "splats/opacity_mean": opacity.mean().item(),
                    "splats/opacity_below_0.05_frac": (opacity < 0.05).float().mean().item(),
                    "splats/opacity_above_0.9_frac": (opacity > 0.9).float().mean().item(),
                    "splats/scale_max_axis_mean": scale.max(dim=1).values.mean().item(),
                    "splats/scale_max_axis_max": scale.max().item(),
                    "splats/scale_min_axis_mean": scale.min(dim=1).values.mean().item(),
                    "splats/extent_xyz_std": xyz.std(dim=0).mean().item(),
                    "splats/outside_grid_aabb_frac": ((xyz < lo) | (xyz > hi)).any(dim=1).float().mean().item(),
                    "splats/active_sh_degree": int(gaussians.active_sh_degree),
                    "splats/opacity_hist": wandb.Histogram(opacity.cpu().numpy()),
                    "splats/log10_scale_max_axis_hist": wandb.Histogram(
                        torch.log10(scale.max(dim=1).values.clamp_min(1e-12)).cpu().numpy()),
                }
                if with_cloud:
                    step = max(int(xyz.shape[0]) // max_points, 1)          # deterministic stride
                    pts = xyz[::step].cpu().numpy()
                    dc = gaussians._features_dc[::step].reshape(-1, 3).cpu().numpy()
                    rgb = np.clip(0.5 + 0.28209479177387814 * dc, 0, 1) * 255.0
                    log["splats/point_cloud"] = wandb.Object3D(np.concatenate([pts, rgb], axis=1))
                log = {"{}/{}".format(stage, k): v for k, v in log.items()}
                log["{}/iteration".format(stage)] = iteration
                self._wandb.log(log)
        except Exception as e:
            print("W&B splat logging failed:", type(e).__name__)

    def log_model_artifact(self, model_path, name):
        """Upload the final saved Gaussians (point_cloud.ply + deformation weights) as an artifact."""
        if self._wandb is None:
            return
        try:
            import glob
            import wandb
            dirs = sorted(glob.glob(os.path.join(model_path, "point_cloud", "iteration_*")),
                          key=lambda d: int(d.rsplit("_", 1)[1]))
            if not dirs:
                return
            safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in name)
            artifact = wandb.Artifact("gaussians-" + safe, type="gaussian-splats",
                                      metadata={"iteration": int(dirs[-1].rsplit("_", 1)[1])})
            artifact.add_dir(dirs[-1])
            self._wandb.log_artifact(artifact)
        except Exception as e:
            print("W&B model artifact failed:", type(e).__name__)

    # ---- end of run -------------------------------------------------
    def finish(self, summary):
        if self._csv_file is not None:
            self._csv_file.close()
        summary = dict(summary)
        summary["wall_clock_seconds_total"] = time.time() - self._t0
        summary["nan_restarts"] = self.nan_restarts()
        summary["depth_nonfinite_skips"] = self.depth_nonfinite_skips
        summary["mean_iter_time_ms"] = {s: self.iter_time_sum[s] / self.iter_time_n[s]
                                        for s in self.iter_time_sum}
        with open(os.path.join(self.model_path, "train_summary.json"), "w") as f:
            json.dump({k: _jsonable(v) for k, v in summary.items()}, f, indent=1)
        if self._wandb is not None:
            self._wandb.summary.update({k: v for k, v in summary.items() if isinstance(v, (int, float))})
            self._wandb.finish()
