# ChronoDepth video-depth regulariser for 4D Gaussian Splatting

This folder is a fork of [4DGaussians](https://github.com/hustvl/4DGaussians) (Wu et al., CVPR 2024) at
commit `205090b`, plus an optional regulariser that ties the rendered depth of the dynamic scene to the
output of a temporally consistent video-depth estimator,
[ChronoDepth](https://github.com/jhaoshao/ChronoDepth) (Shao et al., arXiv:2406.01493).

The original 4DGaussians README is kept unchanged as [`README_4DGaussians.md`](README_4DGaussians.md).

> **Status, 17 November 2026.** The regulariser is implemented and unit-tested. Depth priors exist for
> both scenes. **No regularised experiment has been run.** The check that the baseline is unchanged
> when every new flag is off (B0) has been run 23 times and **does not pass as worded: the unmodified
> baseline itself is unstable on *americano*** (section 10). The dense check says the priors are
> affine in depth (section 7). Nothing in this file is a claim that the regulariser helps.

---

## Contents

1. [What is here](#1-what-is-here)
2. [The pipeline, stage by stage](#2-the-pipeline-stage-by-stage)
3. [COLMAP depth and ChronoDepth are different things](#3-colmap-depth-and-chronodepth-are-different-things)
4. [The regulariser](#4-the-regulariser)
5. [Flags](#5-flags)
6. [ChronoDepth priors: settings actually used](#6-chronodepth-priors-settings-actually-used)
7. [Open question: is the prior depth or disparity?](#7-open-question-is-the-prior-depth-or-disparity)
8. [What every run writes](#8-what-every-run-writes)
9. [Weights & Biases tracking](#9-weights--biases-tracking)
10. [Experiment plan and progress](#10-experiment-plan-and-progress)
11. [How to run](#11-how-to-run)
12. [Findings about the baseline repository](#12-findings-about-the-baseline-repository)
13. [Deviations from the specification](#13-deviations-from-the-specification)
14. [File map](#14-file-map)

---

## 1. What is here

| Area | Files |
|---|---|
| Losses: alignment, L_depth, L_corr, L_temp, λ schedule | `utils/depth_loss.py`, `tests/test_depth_loss.py` |
| Differentiable depth and accumulated-opacity render pass | `gaussian_renderer/__init__.py` |
| Flags, all default off | `arguments/__init__.py`, `utils/params_utils.py` |
| Depth priors in the loader, sparse training split, validation holdout | `scene/hyper_loader.py`, `scene/dataset_readers.py`, `scene/dataset.py`, `scene/__init__.py` |
| Training loop, pair sampling, per-run logs, optional W&B | `train.py`, `utils/run_logging.py` |
| Depth priors (separate environment) | `scripts/run_chronodepth.py`, `scripts/run_depth_anything.py` |
| Checks and measurements | `scripts/check_prior_space.py`, `scripts/depth_temporal_stability.py` |
| Results and diagnostics | `scripts/collect_results.py`, `scripts/diagnostics.py` |
| One config file per experiment | `arguments/hypernerf/exp/` |
| Remote GPU launcher | `remote/app.py`, `remote/launch.py`, `remote/specs/` |

With every new flag at its default the code is intended to train exactly as `205090b` does. That is
tested by experiment B0 (section 10), not assumed.

---

## 2. The pipeline, stage by stage

```
HyperNeRF frames ──► COLMAP ──► initial point cloud ──► 4DGaussians training ──► render ──► metrics
        │                                                      ▲
        └──────────► ChronoDepth ──► per-frame depth prior ────┘   (regularised runs only)
```

| # | Stage | What it does | Baseline | Regularised |
|---|---|---|---|---|
| 1 | Data | HyperNeRF release v0.1; the loader uses the 2x-downsampled images | yes | yes |
| 2 | COLMAP sparse | Features, matching and triangulation on at most 200 frames, with the dataset's camera poses | yes | yes |
| 3 | COLMAP dense stereo | A depth map per selected frame from multi-view matching, in two passes (photometric, then geometric) | yes | yes |
| 4 | COLMAP fusion | Merges those depth maps into one 3D point cloud, `fused.ply` | yes | yes |
| 5 | Downsampling | `scripts/downsample_point.py` thins the cloud to at most 40,000 points, `points3D_downsample2.ply` | yes | yes |
| 6 | ChronoDepth | One depth map per frame of the whole video, in temporal order | no | yes |
| 7 | Training | Coarse stage (canonical Gaussians), then fine stage (Gaussians and deformation field) | yes | yes, plus the depth terms |
| 8 | Evaluation | `render.py` on the full test set, then `metrics.py` | yes | yes |

Stages 2 to 5 are the unmodified `colmap.sh` and `scripts/downsample_point.py` of the baseline.

### Dataset split

The sequences are monocular: one moving camera, one image per time step. The split is the loader's:
every 4th frame trains, the frame two after each training frame tests.

| Scene | Frames | Train | Test | Frames given to COLMAP |
|---|---|---|---|---|
| *americano* | 526 | 132 | 131 | 175 |
| *aleks-teapot* | 975 | 244 | 243 | 195 |

---

## 3. COLMAP depth and ChronoDepth are different things

Both produce depth maps, which makes them easy to confuse. They are unrelated in method and in purpose.

| | COLMAP dense stereo and fusion | ChronoDepth |
|---|---|---|
| Method | Multi-view geometry: triangulation from camera motion between frames | A learned video diffusion model; one map per frame |
| Output | One fused 3D point cloud | A depth image for every frame |
| Used for | The initial positions of the Gaussians, once, before training | The regulariser's loss, at every training iteration |
| Scale | The scale of the reconstruction | Unknown scale and shift, so it has to be aligned (section 4) |
| Needed by | Every run, baseline included | Regularised runs only |
| Part of | The unmodified baseline pipeline | The new work |

**Point-cloud fusion** is the last COLMAP step: it merges the per-frame depth maps of the dense stereo
stage into a single 3D point cloud, keeping points that several views agree on. It is not part of
ChronoDepth.

---

## 4. The regulariser

Notation: `D̂` is the rendered depth of a training frame, `d` the prior for that frame, `A` the
accumulated opacity, `k` the iteration.

**Rendered depth and opacity.** The rasteriser composites front to back with weights `α_i T_i`. One
extra rasteriser pass with colour `(z_i, 1, 0)` on a black background gives, per pixel,

```
D̂ = Σ_i z_i α_i T_i        (camera-space depth, not divided by A)
A  = Σ_i α_i T_i            (accumulated opacity)
```

The deformed Gaussians of the main pass are reused, so the deformation network runs once.

**Ω.** The pixels of the rendered frame with `A > omega_thresh` (and a finite prior).

**Alignment (Eq. align).** One scale `s` and one shift `b` per camera, shared by all its frames,
solved in closed form from a 2×2 system:

```
(s, b) = argmin  Σ_frames Σ_{p∈Ω} ( s·d(p) + b − D̂(p) )²
```

Only five running sums are kept (`N, Σd, Σd², ΣD̂, Σd·D̂`), never the maps. `(s, b)` are constants in
back-propagation. HyperNeRF has a single camera, so there is one `(s, b)` per sequence.
`--align_stats cumulative` sums the statistics of every frame sampled so far (the literal reading);
`--align_stats ema` decays them so that frames rendered early in training fade out.

**Depth term (Eq. ldepth).** `L_depth = mean over Ω of | s·d + b − D̂ |`

**Correlation term (Eq. lcorr).** `L_corr = 1 − Pearson(D̂, d)` over Ω, per frame. It needs no
alignment and is invariant to positive affine maps, but it does not tie frames to each other.

**Temporal term (Eq. ltemp).** For two consecutive training frames `t, t+1`:

```
L_temp = mean over Ω_t ∩ Ω_{t+1} of | (D̂_{t+1} − D̂_t) − s·(d_{t+1} − d_t) |
```

The shift cancels. "Consecutive" means consecutive in the training split, four source frames apart.
Both frames must be rendered in the same iteration, so when `lambda_temp > 0` the fine stage samples
pairs `(t, t+1)` instead of a random batch.

**Objective (Eq. full).** `L = L_base + λ_d(k)·L_depth + λ_t·L_temp`. `L_depth` is active from the
coarse stage; `L_temp` only in the fine stage, where the deformation field exists.

**Guards.** A NaN loss makes the baseline script re-execute itself, so a depth term that produced NaN
would loop forever. Every term returns a finite zero with zero gradient when Ω is empty or a variance
is zero, a non-finite depth term is skipped and counted, restarts are counted, and a run with a depth
term stops after 5 restarts.

**Tests.** `tests/test_depth_loss.py` has 12 tests on synthetic tensors: the alignment recovers a
known `(s, b)` and respects the mask; `L_depth` is zero for an affine-related pair; `L_corr` is
invariant to positive affine maps; `L_temp` is independent of `b` and uses the intersection of the two
Ω; every term is a finite zero with zero gradient on an empty Ω. All 12 pass on CPU and in the
training environment (torch 1.13.1+cu117).

---

## 5. Flags

All default to "off". A config file can set them in a `DepthParams = dict(...)` or
`ModelParams = dict(...)` block.

| Flag | Default | Meaning |
|---|---|---|
| `--depth_loss` | `none` | `none`, `l1_aligned` (L_depth) or `pearson` (L_corr) |
| `--lambda_depth` | `0.0` | λ_d at the first iteration |
| `--lambda_depth_end` | `0.0` | λ_d at the last iteration, for the decay schedules |
| `--lambda_depth_schedule` | `const` | `const`, `linear_decay` or `exp_decay` |
| `--lambda_temp` | `0.0` | λ_t, fine stage only |
| `--omega_thresh` | `0.9` | accumulated-opacity threshold that defines Ω |
| `--align` | `per_camera` | `per_camera`, or `per_frame` (ablation: a separate fit for every frame) |
| `--align_stats` | `cumulative` | `cumulative` or `ema` |
| `--align_ema_decay` | `0.99` | decay per update when `--align_stats ema` |
| `--depth_prior_space` | `auto` | `depth`, `disparity`, or `auto` (read from the prior's settings file) |
| `--depth_prior_dir` | `""` | folder of `{frame_id}.npy`; empty means `<source_path>/depth/2x` |
| `--train_every` | `1` | keep every N-th training frame (sparse-view experiments) |
| `--val_holdout` | `0` | N > 0 holds out every N-th training frame as a validation set |
| `--seed` | `-1` | seeds Python, NumPy and PyTorch; below 0 keeps the seeding of `205090b` |

The test split never changes: it is computed before `--train_every` and `--val_holdout` are applied.

---

## 6. ChronoDepth priors: settings actually used

Produced by `scripts/run_chronodepth.py`, a thin wrapper that imports `run_pipeline` and `seed_all`
from the official `run_infer.py` and builds the pipeline the same way. It runs on **all** frames in
temporal order, because the estimator needs the contiguous video; only training frames' priors are
loaded into the loss. Source of every value below: `<scene>/depth/2x/chronodepth_settings.json`.

| Setting | Value |
|---|---|
| Repository | `https://github.com/jhaoshao/ChronoDepth` |
| Repository commit | `2580e891eb33a5c3a95eab27b745f64ac28b3514` |
| Checkpoint | `jhshao/ChronoDepth-v1`, revision `8ed51301cb97ec7a61a94a1bff1f7a107f73e9e8` |
| Base model | `stabilityai/stable-video-diffusion-img2vid-xt`, revision `9e43909513c6714f1bc78bcb44d96e733cd242aa` |
| Window length F | 10 frames |
| Window overlap | 5 frames (5 new frames per window) |
| Denoising steps | 5 |
| Inference resolution | 960 × 512 (height × width); output resized to the 960 × 536 frames |
| σ_ε, inference mode | −4.0, `ours` |
| Precision | fp16, xformers attention |
| Random seed | 1234 |
| Output | float32, clipped to [0, 1] by the official pipeline, affine-invariant |

| Scene | Frames | Inference time | Share of pixels at a clip bound |
|---|---|---|---|
| *americano* | 526 | 264.8 s | 6.19e-06 |
| *aleks-teapot* | 975 | 474.5 s | 4.23e-06 |

These are the official script's defaults. They were not tuned.

---

## 7. Open question: is the prior depth or disparity?

This decides whether the alignment of section 4 is valid as written. If the estimator's output is
affine in **depth**, it is aligned to rendered depth. If it is affine in **inverse depth**
(disparity), the affine relation holds against inverse rendered depth instead, and the paper's
alignment equation needs the same correction. Neither the ChronoDepth paper nor its inference code
states which it is.

**First measurement (sparse, inconclusive).** The dataset's SfM points were projected into 24 evenly
spaced frames and the prior at those pixels was correlated with the points' camera-space depth `z`
and with `1/z` (median Pearson r over frames; pinhole projection, lens distortion ignored, no
occlusion test).

| Scene | r against `z` | r against `1/z` |
|---|---|---|
| *americano* | +0.490 | −0.487 |
| *aleks-teapot* | +0.672 | −0.680 |

What this supports: **larger values mean farther**, on both scenes. What it does not support: any
choice between depth and disparity. The two correlations differ in the third decimal and the label
flips between scenes. The `prior_space` field that the script wrote into the two settings files
(`depth` for *americano*, `disparity` for *aleks-teapot*) is therefore noise and must not be used.

**Second measurement (dense, *americano*).** `scripts/check_prior_space.py` renders depth from a
trained model for 32 training frames and fits the prior against rendered depth and against inverse
rendered depth over all pixels of Ω, per frame and pooled with a single `(s, b)`. The reference was
the first B0 branch run with flags off (30.80 dB, a "degraded" run in the table of section 10). At
least 93.7% of pixels were in Ω in every frame.

| Fit | Against depth | Against inverse depth |
|---|---|---|
| Pooled, one `(s, b)`, Pearson r | +0.864 | −0.824 |
| Pooled, relative L1 residual | 0.376 | 0.459 |
| Per-frame median r | +0.916 | −0.877 |
| Frames where this space fits better | 32 of 32 | 0 of 32 |

The priors are affine in **depth**, larger meaning farther, so the alignment of section 4 is valid as
written and `depth_prior_space='depth'` will be set explicitly in every experiment config rather
than read with `auto`. Two cautions: the reference is a model's own geometry, not ground truth, and
the margin is moderate. And when each frame is fitted separately the scale ranges from 15.4 to 54.0
over the sequence, so a single scale and shift per camera does not fit this sequence closely; that is
what ablation (e) of E2 tests. The check has not been run on *aleks-teapot*.

Both spaces are implemented: with `--depth_prior_space disparity` every term uses the
alpha-normalised inverse rendered depth `A / D̂` in place of `D̂`.

---

## 8. What every run writes

Under `<model_path>/`:

| File | Content |
|---|---|
| `resolved_config.json` | every hyperparameter **after** the config file is merged, all new flags, the resolved prior directory and space, and the ChronoDepth settings of the prior used |
| `git_commit.txt` | commit of the code that ran |
| `train_frames.json` | the exact frame ids of the train, validation and test sets |
| `train_log.csv` | per iteration: stage, L1, total loss, each depth term unweighted and weighted, λ_d, λ_t, `s`, `b`, the fraction of pixels in Ω, Gaussian count, iteration time |
| `train_summary.json` | training time, total wall-clock time, final Gaussian count, NaN restarts, skipped non-finite depth terms, final `(s, b)`, mean iteration time per stage, share of Gaussian centres outside the deformation grid's bounding box |
| `val_metrics.json` | validation PSNR per frame and mean, when `--val_holdout` is set |
| `results.json`, `per_view.json` | output of `metrics.py` on the full test set |
| `train_stdout.txt`, `render_stdout.txt`, `metrics_stdout.txt` | console output; render FPS is read from the second |

`scripts/collect_results.py` turns a folder of runs into `results.csv`, one row per run, with a
`status` column (`ok`, `failed`, `nan_restart`). Every cell comes from a file the run wrote; a missing
file leaves the cell empty.

The hyperparameters are recorded after the merge on purpose. The config file is merged late in
`train.py`, so anything recorded earlier holds the code defaults, not the values that were used.

---

## 9. Weights & Biases tracking

Optional. It is used only when `WANDB_API_KEY` is in the environment; otherwise everything above is
still written to local files. Copy `.env.example` to `.env` (git-ignored) and fill in the key. The
key is never printed, logged or written to any other file.

**Depth-prior run** (`chronodepth-prior/<scene>`, job type `depth-prior`):

- every inference setting of section 6 as the run config;
- 8 evenly spaced preview frames, RGB beside the depth map;
- a depth video of the whole sequence;
- per-frame mean, standard deviation, minimum, maximum and mean absolute change from the previous frame;
- the two correlations of section 7 in the run summary;
- an artifact with the settings file and the previews.

**Training run:**

- the resolved config, including λ_d and its schedule, λ_t, the Ω threshold, the alignment mode and the settings of the prior used;
- every column of `train_log.csv`, every 50 iterations;
- every 1,000 iterations a panel: training frame, rendered depth, aligned prior, Ω;
- **Gaussian splats**, every 500 iterations: count, opacity mean and histogram, share of near-transparent and near-opaque Gaussians, scale statistics and histogram, spatial extent, share of centres outside the deformation grid, active SH degree;
- a coloured 3D point cloud of the canonical Gaussian centres (up to 100,000 points) every 3,500 iterations and at the end of each stage;
- the final `point_cloud.ply` and deformation weights as an artifact.

The tracking code only reads the model and uses no random numbers, so it cannot change training.

---

## 10. Experiment plan and progress

All at 14,000 iterations, 3,000 coarse, batch size 2, the reference HyperNeRF config
(`arguments/hypernerf/default.py`), unless stated. Evaluation is identical for every run: `render.py`
then `metrics.py` on the full test set, reporting PSNR, SSIM, MS-SSIM, D-SSIM, LPIPS-VGG and
LPIPS-Alex, plus Gaussian count, training time and render FPS.

| Block | What | State |
|---|---|---|
| Preprocessing | COLMAP and point cloud, both scenes | done |
| Depth priors | ChronoDepth, both scenes | done |
| **B0** | Baseline at `205090b` against this code with every flag off, *americano*, same seeding | 23 runs done; does not pass as worded, see below |
| Prior space | Dense depth-versus-disparity check (section 7) | done on *americano*: depth |
| **E1** | Hyperparameter selection on *americano*, one seed, selected on validation PSNR (every 5th training frame held out): λ_d ∈ {0.01, 0.03, 0.1, 0.3, 1.0}; then Ω threshold ∈ {0.5, 0.9, 0.99} and schedule ∈ {const, linear decay to 0}; then λ_t ∈ {0, 0.01, 0.1, 1.0} | not started |
| **E2** | Main comparison, 3 seeds, both scenes: (a) baseline, (b) + L_depth, (c) + L_corr instead, (d) + L_depth + L_temp, (e) as (d) with `--align per_frame` | not started |
| **E3** | Sparse views, 3 seeds, both scenes: every 2nd, 4th and 8th training frame; (a) against the best of E2 | not started |
| E4 | Only if time remains: E2 (a) and (d) at 35,000 iterations on *aleks-teapot* | not started |
| Temporal stability | One number for ChronoDepth and for per-frame Depth Anything V2, both scenes | not started |

Rules the results will be reported under:

- every number comes from a file a run wrote; failed and bad-looking runs are listed, not dropped;
- the whole sweep is reported, not only the winner, and selection never uses test metrics;
- a difference smaller than the seed-to-seed spread is not called an improvement;
- "no measurable effect" is a valid outcome and will be stated where true.

### B0 result: the baseline itself is unstable on *americano*

B0 was meant to be one run of each code. The first pair differed by 0.56 dB, so it was repeated:
23 runs in total, all on *americano*, 14,000 iterations, the reference config, default seeding, one
A100-40GB each. Every run is listed. "Healthy" means test PSNR above 31.2 dB, "collapsed" below
27 dB, "degraded" in between; the thresholds are labels chosen after seeing the results.

| Arm | Run | Outcome | PSNR | SSIM | LPIPS-VGG | LPIPS-Alex | Final Gaussians |
|---|---|---|---|---|---|---|---|
| Baseline `205090b` | r1 | healthy | 31.3588 | 0.9167 | 0.1377 | 0.0702 | 213,500 |
|  | r2 | healthy | 31.4222 | 0.9178 | 0.1353 | 0.0686 | 213,648 |
|  | r3 | healthy | 31.3801 | 0.9170 | 0.1364 | 0.0693 | 213,538 |
|  | r4 | degraded | 30.5720 | 0.9033 | 0.1727 | 0.0970 | 219,447 |
|  | r5 | collapsed | 25.7450 | 0.7995 | 0.3060 | 0.1980 | 363,393 |
|  | r6 | crashed at fine 7,970, after collapsing | — | — | — | — | — |
|  | r7 | healthy | 31.3542 | 0.9170 | 0.1362 | 0.0690 | 212,238 |
|  | r8 | healthy | 31.3553 | 0.9171 | 0.1369 | 0.0703 | 213,029 |
| Branch, flags off, W&B tracking on | r1 | degraded | 30.8006 | 0.9071 | 0.1648 | 0.0895 | 212,293 |
|  | r2 | collapsed | 25.8924 | 0.8016 | 0.3043 | 0.1992 | 360,927 |
|  | r3 | collapsed | 25.5033 | 0.7942 | 0.3102 | 0.2072 | 314,524 |
| Branch, flags off, W&B tracking off | r2 | collapsed | 25.9744 | 0.8066 | 0.2984 | 0.1932 | 361,760 |
|  | r3 | healthy | 31.2919 | 0.9163 | 0.1383 | 0.0707 | 214,233 |
|  | r4 | healthy | 31.3692 | 0.9168 | 0.1380 | 0.0705 | 219,484 |
|  | r5 | crashed at fine 1,800 | — | — | — | — | — |
|  | r6 | collapsed | 24.6862 | 0.7648 | 0.3419 | 0.2412 | 360,950 |
|  | r7 | degraded | 30.4624 | 0.9025 | 0.1714 | 0.0915 | 232,603 |
|  | r8 | healthy | 31.3823 | 0.9177 | 0.1366 | 0.0687 | 212,931 |
| Branch, flags off, per-run logging disabled | r4 | healthy | 31.3372 | 0.9170 | 0.1367 | 0.0696 | 213,238 |
|  | r5 | crashed at fine 880 | — | — | — | — | — |
|  | r6 | healthy | 31.2958 | 0.9163 | 0.1373 | 0.0709 | 214,169 |
|  | r7 | collapsed | 25.8065 | 0.8010 | 0.3055 | 0.1991 | 360,969 |
|  | r8 | degraded | 29.3591 | 0.8811 | 0.2028 | 0.1126 | 276,967 |

| Arm | Runs | Healthy | Degraded | Collapsed | Crashed |
|---|---|---|---|---|---|
| Baseline `205090b` | 8 | 5 | 1 | 1 | 1 |
| Branch, flags off, W&B tracking on | 3 | 0 | 1 | 2 | 0 |
| Branch, flags off, W&B tracking off | 7 | 3 | 1 | 2 | 1 |
| Branch, flags off, per-run logging disabled | 5 | 2 | 1 | 1 | 1 |

What the runs show:

- **The unmodified baseline fails too.** Three of its eight runs are not healthy: one degraded, one
  collapsed to 25.7 dB, one crashed after collapsing. The collapses are therefore not introduced by
  this code.
- **The added logging is not the cause.** Collapses occur with W&B tracking on, with it off, and with
  all per-run logging disabled (`FDGS_NO_RUN_LOGGING=1`).
- **Healthy runs of the two codes agree.** Baseline 31.374 ± 0.029 dB (5 runs); branch 31.335 ± 0.041 dB (5 runs).
- **The failure rates cannot be told apart at this sample size.** 5 of 8 baseline runs are healthy
  against 5 of 15 branch runs (Fisher's exact test, two-sided, p = 0.22). The branch looks worse
  and a higher failure rate for it is not ruled out.
- **The collapse.** The coarse stage is the same in every run (test PSNR 20.88–20.91 at iteration
  3,000). In the fine stage the training loss drifts upward over a few hundred iterations, at a
  different point in each run (near fine iteration 1,800, 2,000–2,500 and 6,000 in three that were
  traced), and does not recover. The Gaussian count then grows towards the 360,000 cap instead of
  settling near 213,000. No run had a NaN restart.
- **The crashes.** Three runs, one per arm, stopped with
  `CUDA error: CUBLAS_STATUS_EXECUTION_FAILED`. None of the logs contains an out-of-memory message,
  and two of the three crashed with 103,175 and 127,347 Gaussians, well below the count healthy runs
  reach. The cause is not known. Five baseline runs with `CUDA_LAUNCH_BLOCKING=1` and external GPU
  memory sampling (`remote/specs/b0_debug_blocking.json`) are running to locate it.

Consequences:

- B0 does **not** pass as worded ("flags off behaves as `205090b`") on single runs; both codes are
  bimodal on this scene, and equivalence can only be stated for the healthy runs and as a comparison
  of failure rates.
- The baseline's run-to-run spread among healthy runs is about 0.03 dB, tighter than the 0.13 dB
  assumed in the task; but single runs of either code have landed 0.8 to 6.7 dB lower.
- No regularised run has been started. With this instability, a 3-seed comparison (E2) cannot
  separate an effect of the regulariser from a collapsed seed, so the plan needs either more seeds
  per arm or the instability fixed first. That decision is open.

---

## 11. How to run

### Environments

Two, exchanging data only through `.npy` files on disk.

```bash
# 1. training: the baseline's environment
conda create -n Gaussians4D python=3.7 && conda activate Gaussians4D
pip install -r requirements.txt                      # torch 1.13.1
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn

# 2. depth inference only: ChronoDepth's own requirements (torch 2.1.0, diffusers 0.29.1, ...)
git clone https://github.com/jhaoshao/ChronoDepth && cd ChronoDepth
git checkout 2580e891eb33a5c3a95eab27b745f64ac28b3514 && pip install -r requirements.txt
```

The submodules are not registered in this repository (section 12). Clone them at the recorded commits:

```bash
git clone --recursive https://github.com/ingra14m/depth-diff-gaussian-rasterization submodules/depth-diff-gaussian-rasterization
git -C submodules/depth-diff-gaussian-rasterization checkout f2d8fa9921ea9a6cb9ac1c33a34ebd1b11510657
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
git -C submodules/simple-knn checkout 44f764299fa305faf6ec5ebd99939e0508331503
```

### Steps

```bash
S=data/hypernerf/misc/americano

# preprocessing (baseline, unchanged)
bash colmap.sh $S hypernerf
python scripts/downsample_point.py $S/colmap/dense/workspace/fused.ply $S/points3D_downsample2.ply

# depth priors (depth environment) -> $S/depth/2x/{frame_id}.npy + chronodepth_settings.json
python scripts/run_chronodepth.py --datadir $S --chronodepth_repo /path/to/ChronoDepth --seed 1234

# baseline: no new flag
python train.py -s $S/ --expname hypernerf/americano_base --configs arguments/hypernerf/default.py

# regularised: one config file per experiment, never edit a shared config between runs
python train.py -s $S/ --expname e1/americano_ld0p1 --configs arguments/hypernerf/exp/e1_ld0p1.py --seed 0

# evaluation, with the run's own config
python render.py --model_path output/e1/americano_ld0p1 --skip_train --configs arguments/hypernerf/exp/e1_ld0p1.py
python metrics.py --model_path output/e1/americano_ld0p1/

# unit tests, results table
python -m pytest tests -q
python scripts/collect_results.py output results.csv
```

### Remote GPUs

`remote/app.py` defines the two environments as container images (including COLMAP 3.8 built with
CUDA) and one job per stage. `remote/launch.py` submits a job to the deployed app and returns at once,
so jobs do not depend on the local machine staying awake.

```bash
modal deploy remote/app.py                       # after every code change
python remote/launch.py prep americano           # download + COLMAP + downsampling
python remote/launch.py depth americano          # ChronoDepth
python remote/launch.py runs remote/specs/b0.json
python remote/launch.py status
python remote/launch.py collect results.csv
```

`launch.py runs` refuses to start from a modified working tree, so every run maps to one commit.

At most `FDGS_MAX_RUN_GPUS` training runs (default 4, read at deploy time) hold a GPU at once; further
submissions queue. Every run also writes `gpu_mem.csv` and `gpu_mem_summary.json`: GPU and container
memory sampled every 2 s from outside the training process, so the baseline code is not touched.

---

## 12. Findings about the baseline repository

Verified in the code at `205090b`. None of these was changed, so that the baseline stays reproducible.

1. **The rasteriser's depth output cannot be trained against.** `forward.cu` accumulates
   `D += depths[i] * alpha * T` and writes it out without dividing by accumulated alpha, so it is the
   alpha-weighted expected depth (not a median, not normalised). But `backward.cu` never reads a depth
   gradient, and the Python wrapper ignores `grad_depth`. A loss on `render(...)["depth"]` would
   therefore change nothing. Hence the second render pass of section 4.
2. **The renderer returns no accumulated opacity.** The same second pass provides it.
3. **The submodules are not recorded.** `.gitmodules` lists them, but the tree has no entry for them
   and `submodules/` is git-ignored, so `git submodule update --init` does nothing. The commits used
   here come from the checkout log in `4DGaussians.ipynb`.
4. **`database.py` crashes inside `colmap.sh`.** `scripts/hypernerf2colmap.py` writes the image size
   as a float (`536.0`), and `database.py` calls `int('536.0')`, which raises `ValueError`. The shell
   script does not stop on errors and carries on. This happens with the unmodified code, so earlier
   runs of this pipeline will have hit it too.
5. **The loader already builds a depth path nobody reads** (`{datadir}/depth/2x/{frame_id}.npy`).
   The priors are stored there.
6. **A NaN loss makes `train.py` re-execute itself**, with no limit.
7. **The config file is merged after the argument groups are built**, and `save_iterations` gets the
   code-default iteration count appended before the merge.
8. **`colmap.sh` with Windows line endings corrupts every path it builds.** On a Windows checkout,
   convert it to LF before running it on Linux; the remote launcher does this on a copy.
9. **Training is unstable on *americano* at the reference config.** Of eight runs of the unmodified
   code, one collapsed to 25.7 dB, one crashed with a CUDA error after collapsing and one finished
   0.8 dB low (section 10). Training is also not bit-reproducible under the default seeding.

---

## 13. Deviations from the specification

| # | Deviation | Reason |
|---|---|---|
| 1 | Depth and accumulated opacity come from a second rasteriser pass, not from the rasteriser's `depth` output | Finding 1 above: that output has no gradient |
| 2 | Submodules cloned by hand at commits taken from the repository's notebook | Finding 3 above |
| 3 | With `--align per_frame`, `L_temp` uses one scale fitted jointly on the two frames of the pair | The paper defines `L_temp` with a per-camera scale; a per-frame fit gives two different scales for one pair |
| 4 | `--depth_prior_space` exists and both spaces are implemented | Section 7: the space of the estimator's output is not documented |
| 5 | With `lambda_temp > 0` the fine stage samples pairs of consecutive training frames instead of a random batch | `L_temp` needs both frames in the same iteration |
| 6 | A run with a depth term stops after 5 NaN restarts | The baseline's restart loop has no limit |
| 7 | COLMAP 3.8 is built without its GUI in the remote image | Headless containers; the command-line tools are the same |
| 8 | Remote training runs on an A100 (40 GB), not an RTX A6000 | Availability; timings are comparable only among runs of this project |

---

## 14. File map

```
utils/depth_loss.py               alignment, L_depth, L_corr, L_temp, λ schedule
utils/run_logging.py              provenance files, train_log.csv, optional W&B
tests/test_depth_loss.py          12 unit tests
gaussian_renderer/__init__.py     render(..., return_depth_alpha=True)
arguments/__init__.py             DepthParams; depth_prior_dir, train_every, val_holdout
arguments/hypernerf/exp/          one config per experiment
scene/hyper_loader.py             priors, sparse split, validation holdout
train.py                          depth terms, pair sampling, logging
scripts/run_chronodepth.py        ChronoDepth priors + settings file + previews + W&B
scripts/run_depth_anything.py     per-frame Depth Anything V2 maps (stability comparison only)
scripts/check_prior_space.py      dense depth-versus-disparity check
scripts/depth_temporal_stability.py   one stability number per estimator and scene
scripts/collect_results.py        results.csv
scripts/diagnostics.py            curves, depth beside aligned prior, RGB comparison
remote/app.py, remote/launch.py   remote GPU jobs
README_4DGaussians.md             the original README of the fork
```
