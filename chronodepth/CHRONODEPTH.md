# ChronoDepth video-depth regulariser for 4DGaussians

This folder is the 4DGaussians fork at commit `205090b` plus an optional
video-depth regulariser. With every new flag at its default, training is
meant to behave as at `205090b`; the comparison run that checks this (B0) has
not been completed yet, and no experiment result exists so far.

## What was added

| Area | Files |
|---|---|
| Losses: alignment, L_depth, L_corr, L_temp | `utils/depth_loss.py`, `tests/test_depth_loss.py` |
| Differentiable depth and accumulated-opacity render pass | `gaussian_renderer/__init__.py` |
| Flags (all default off) | `arguments/__init__.py`, `utils/params_utils.py` |
| Depth priors, sparse training split, validation holdout | `scene/hyper_loader.py`, `scene/dataset_readers.py`, `scene/dataset.py`, `scene/__init__.py` |
| Training loop, pair sampling, per-run logs, optional W&B | `train.py`, `utils/run_logging.py` |
| Depth priors (separate environment) | `scripts/run_chronodepth.py`, `scripts/run_depth_anything.py`, `scripts/depth_temporal_stability.py` |
| Results and diagnostics | `scripts/collect_results.py`, `scripts/diagnostics.py` |
| One config file per experiment | `arguments/hypernerf/exp/` |
| Remote GPU launcher | `remote/` |

## Two things that differ from what one would expect

* The rasteriser's `depth` output has no backward pass (`backward.cu` never
  reads a depth gradient). The regulariser therefore renders depth and
  accumulated opacity through a second, differentiable pass
  (`render(..., return_depth_alpha=True)`).
* The repository lists its submodules in `.gitmodules` but does not record
  them in the tree, and `submodules/` is git-ignored. Clone them by hand at
  the commits recorded in `4DGaussians.ipynb`:

  ```bash
  git clone --recursive https://github.com/ingra14m/depth-diff-gaussian-rasterization submodules/depth-diff-gaussian-rasterization
  git -C submodules/depth-diff-gaussian-rasterization checkout f2d8fa9921ea9a6cb9ac1c33a34ebd1b11510657
  git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
  git -C submodules/simple-knn checkout 44f764299fa305faf6ec5ebd99939e0508331503
  ```

## Flags

`--depth_loss {none,l1_aligned,pearson}`, `--lambda_depth`, `--lambda_depth_end`,
`--lambda_depth_schedule {const,linear_decay,exp_decay}`, `--lambda_temp`,
`--omega_thresh`, `--align {per_camera,per_frame}`, `--align_stats {cumulative,ema}`,
`--depth_prior_space {auto,depth,disparity}`, `--depth_prior_dir`, `--seed`,
`--train_every`, `--val_holdout`.

## Credentials

Copy `.env.example` to `.env` (git-ignored) and fill in `WANDB_API_KEY`.
W&B is optional; without the key every run logs to local files only.

## Tests

```bash
python -m pytest tests -q
```
