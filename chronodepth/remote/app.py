"""Remote GPU launcher for the depth-regulariser experiments.

Three container images, exchanging data only through files on a volume:
  * train image   Python 3.7 / torch 1.13.1 environment of the baseline
                  (+ COLMAP 3.8 with CUDA for preprocessing)
  * depth image   recent diffusers stack for ChronoDepth inference only

Usage (from the repository root):
  modal run remote/app.py::prep --scene americano
  modal run remote/app.py::depth --scene americano
  modal run remote/app.py::tests
  modal run remote/app.py::runs --spec remote/specs/b0.json

No credential is read or written here. W&B is used only if a secret that
provides WANDB_API_KEY is attached (FDGS_WANDB_SECRET names it).
"""
import json
import os
import subprocess
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BASELINE_EXPORT = os.path.join(os.path.dirname(REPO), "_baseline_205090b")
VOL = "/vol"
PY37 = "/opt/env37/bin/python"
CHRONODEPTH_URL = "https://github.com/jhaoshao/ChronoDepth"
CHRONODEPTH_COMMIT = "2580e891eb33a5c3a95eab27b745f64ac28b3514"
GPU = os.environ.get("FDGS_GPU", "A100-40GB")
# At most this many training runs hold a GPU at once; further submissions queue.
MAX_RUN_GPUS = int(os.environ.get("FDGS_MAX_RUN_GPUS", "4"))

SCENES = {
    "americano": {"zip": "misc_americano.zip", "group": "misc"},
    "aleks-teapot": {"zip": "interp_aleks-teapot.zip", "group": "interp"},
}
RELEASE = "https://github.com/google/hypernerf/releases/download/v0.1"
# Multi-camera DyNeRF (Neural 3D Video) scenes: one mp4 per camera plus poses_bounds.npy
DYNERF_SCENES = {"coffee_martini": "coffee_martini.zip"}
DYNERF_RELEASE = "https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0"

app = modal.App("fdgs-depth-reg")
volume = modal.Volume.from_name("fdgs-depth-reg", create_if_missing=True)


def load_dotenv(path):
    """Load KEY=VALUE lines of the git-ignored .env into the environment (never printed)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if value and key.strip() not in os.environ:
                os.environ[key.strip()] = value


load_dotenv(os.path.join(REPO, ".env"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "chronodepth-reg")
# Exactly one secret object in every environment (the container re-imports this file
# and must see the same dependencies). Locally it carries the key from the environment
# or .env; without a key it is empty and runs log to local files only.
if os.environ.get("FDGS_WANDB_SECRET"):
    secrets = [modal.Secret.from_name(os.environ["FDGS_WANDB_SECRET"])]
else:
    secrets = [modal.Secret.from_dict({"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})]

CODE_IGNORE = [".git", "data", "output", "**/__pycache__", "**/.ipynb_checkpoints", "*.ipynb",
               "assets", "docs", "colmap", "SIBR_viewers", "**/build", "**/*.egg-info", "*.log", "*.png"]

COLMAP_APT = ("git cmake ninja-build build-essential libboost-program-options-dev libboost-filesystem-dev "
              "libboost-graph-dev libboost-system-dev libboost-test-dev libeigen3-dev libflann-dev "
              "libfreeimage-dev libmetis-dev libgoogle-glog-dev libgflags-dev libsqlite3-dev libglew-dev "
              "qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev xvfb").split()

train_image = (
    modal.Image.from_registry("nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04", add_python="3.11")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC", "TORCH_CUDA_ARCH_LIST": "7.0 7.5 8.0 8.6",
          "MPLBACKEND": "Agg", "QT_QPA_PLATFORM": "offscreen"})
    .apt_install(*COLMAP_APT, "wget", "curl", "unzip", "ffmpeg", "libgl1", "libglib2.0-0", "bzip2")
    .run_commands(
        # COLMAP 3.8 with CUDA, as in the baseline pipeline
        "git clone https://github.com/colmap/colmap.git /opt/colmap && cd /opt/colmap && git checkout 3.8 "
        "&& mkdir -p build && cd build "
        "&& cmake .. -GNinja -DCUDA_ENABLED=ON -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN='70 75 80 86' "
        "-DCMAKE_CUDA_ARCHITECTURES='70;75;80;86' "
        "&& ninja && ninja install && cd / && rm -rf /opt/colmap/build",
    )
    .run_commands(
        # the baseline's Python 3.7 / torch 1.13.1 environment
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba",
        "micromamba create -y -p /opt/env37 -c conda-forge python=3.7 pip",
        "/opt/env37/bin/pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1",
        "/opt/env37/bin/pip install --no-cache-dir mmcv==1.6.0 matplotlib lpips plyfile pytorch_msssim open3d "
        "'imageio[ffmpeg]' scipy tqdm pytest wandb",
    )
    .add_local_dir(os.path.join(REPO, "submodules"), "/opt/submodules", copy=True,
                   ignore=["**/.git", "**/build", "**/*.egg-info", "**/third_party/glm/test", "**/third_party/glm/doc"])
    .run_commands(
        "/opt/env37/bin/pip install --no-cache-dir /opt/submodules/depth-diff-gaussian-rasterization",
        "/opt/env37/bin/pip install --no-cache-dir /opt/submodules/simple-knn",
    )
    .add_local_dir(REPO, "/root/4dgs", ignore=CODE_IGNORE + ["submodules"])
)
if os.path.isdir(BASELINE_EXPORT):
    train_image = train_image.add_local_dir(BASELINE_EXPORT, "/root/4dgs_base", ignore=CODE_IGNORE)

depth_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        # the official repository's requirements.txt
        "torch==2.1.0", "torchvision==0.16.0", "xformers==0.0.22.post7", "diffusers==0.29.1",
        "easydict==1.13", "einops==0.8.0", "matplotlib==3.8.4", "mediapy==1.2.2", "numpy==1.26.4",
        "Pillow==10.3.0", "tqdm==4.66.2", "accelerate==0.28.0", "transformers==4.43.3", "opencv-python",
        "huggingface_hub==0.24.6", "wandb", "moviepy==1.0.3", "imageio[ffmpeg]",
    )
    .run_commands(f"git clone {CHRONODEPTH_URL} /opt/ChronoDepth && cd /opt/ChronoDepth && git checkout {CHRONODEPTH_COMMIT}")
    .env({"HF_HOME": f"{VOL}/hf"})
    .add_local_dir(os.path.join(REPO, "scripts"), "/root/scripts")
)


def scene_dir(scene):
    """Directory that holds dataset.json for a scene (created by `prep`)."""
    if scene in DYNERF_SCENES:
        cand = f"{VOL}/data/dynerf/{scene}"
        if os.path.exists(f"{cand}/poses_bounds.npy"):
            return cand
        raise FileNotFoundError(f"scene {scene} not prepared under {cand}")
    root = f"{VOL}/data/hypernerf/{SCENES[scene]['group']}"
    for cand in (f"{root}/{scene}", f"{root}/{scene}/{scene}"):
        if os.path.exists(f"{cand}/dataset.json"):
            return cand
    raise FileNotFoundError(f"scene {scene} not prepared under {root}")


def sh(cmd, cwd=None, env=None, log=None):
    """Run a shell command, echo it, optionally tee the output to a file."""
    print("+", cmd, flush=True)
    full_env = dict(os.environ)
    full_env.update(env or {})
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=full_env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    lines = []
    for line in proc.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    code = proc.wait()
    if log:
        with open(log, "a") as f:
            f.write("+ " + cmd + "\n" + "".join(lines))
    if code != 0:
        raise RuntimeError(f"command failed ({code}): {cmd}")
    return "".join(lines)


# ---------------------------------------------------------------- data
@app.function(image=train_image, volumes={VOL: volume}, gpu=GPU, timeout=10 * 3600, cpu=16, memory=65536)
def prep_remote(scene: str, force: bool = False):
    """Download a HyperNeRF sequence and run the baseline preprocessing
    (COLMAP via colmap.sh, then scripts/downsample_point.py)."""
    info = SCENES[scene]
    root = f"{VOL}/data/hypernerf/{info['group']}"
    os.makedirs(root, exist_ok=True)
    try:
        sdir = scene_dir(scene)
    except FileNotFoundError:
        sh(f"wget -q {RELEASE}/{info['zip']} -O /tmp/{info['zip']} && unzip -q /tmp/{info['zip']} -d {root}")
        sdir = scene_dir(scene)
        volume.commit()
    ply = f"{sdir}/points3D_downsample2.ply"
    log = f"{sdir}/preprocess_log.txt"
    if force or not os.path.exists(ply):
        # the checkout may carry Windows line endings; run an LF copy of the unchanged script
        import shutil
        for root_dir in (os.path.dirname(sdir), sdir):      # leftovers of a run with CRLF endings
            for name in [n for n in os.listdir(root_dir) if chr(13) in n]:
                target = os.path.join(root_dir, name)
                shutil.rmtree(target) if os.path.isdir(target) else os.remove(target)
        with open("/root/4dgs/colmap.sh", "rb") as f:
            script = f.read().replace(b"\r\n", b"\n")
        with open("/tmp/colmap_lf.sh", "wb") as f:
            f.write(script)
        sh(f"bash /tmp/colmap_lf.sh {sdir} hypernerf", cwd="/root/4dgs", env={"PATH": f"/opt/env37/bin:{os.environ['PATH']}"}, log=log)
        sh(f"{PY37} scripts/downsample_point.py {sdir}/colmap/dense/workspace/fused.ply {ply}", cwd="/root/4dgs", log=log)
        volume.commit()
    with open(f"{sdir}/dataset.json") as f:
        n = len(json.load(f)["ids"])
    return {"scene": scene, "dir": sdir, "frames": n, "ply_bytes": os.path.getsize(ply),
            "colmap_version": sh("colmap -h | head -3")}


@app.function(image=train_image, volumes={VOL: volume}, gpu=GPU, timeout=10 * 3600, cpu=16, memory=65536)
def prep_dynerf_remote(scene: str, force: bool = False):
    """Download a DyNeRF sequence and run the baseline preprocessing as its README gives it:
    scripts/preprocess_dynerf.py (frame extraction), colmap.sh <dir> llff, scripts/downsample_point.py."""
    root = f"{VOL}/data/dynerf"
    sdir = f"{root}/{scene}"
    os.makedirs(root, exist_ok=True)
    if not os.path.exists(f"{sdir}/poses_bounds.npy"):
        name = DYNERF_SCENES[scene]
        sh(f"wget -q {DYNERF_RELEASE}/{name} -O /tmp/{name} && unzip -q -o /tmp/{name} -d {root}")
        volume.commit()
    if not os.path.exists(f"{sdir}/poses_bounds.npy"):
        raise FileNotFoundError("unexpected archive layout: " + sh(f"find {root} -maxdepth 3 | head -50"))
    ply = f"{sdir}/points3D_downsample2.ply"
    log = f"{sdir}/preprocess_log.txt"
    env = {"PATH": f"/opt/env37/bin:{os.environ['PATH']}"}
    if force or not os.path.exists(ply):
        sh(f"{PY37} scripts/preprocess_dynerf.py --datadir {sdir}", cwd="/root/4dgs", env=env, log=log)
        volume.commit()
        with open("/root/4dgs/colmap.sh", "rb") as f:      # LF copy of the unchanged script
            script = f.read().replace(b"\r\n", b"\n")
        with open("/tmp/colmap_lf.sh", "wb") as f:
            f.write(script)
        sh(f"bash /tmp/colmap_lf.sh {sdir} llff", cwd="/root/4dgs", env=env, log=log)
        sh(f"{PY37} scripts/downsample_point.py {sdir}/colmap/dense/workspace/fused.ply {ply}", cwd="/root/4dgs", log=log)
        volume.commit()
    cams = sorted(d for d in os.listdir(sdir) if d.startswith("cam") and os.path.isdir(f"{sdir}/{d}/images"))
    return {"scene": scene, "dir": sdir, "cameras": len(cams),
            "frames_first_camera": len(os.listdir(f"{sdir}/{cams[0]}/images")) if cams else 0,
            "ply_bytes": os.path.getsize(ply)}


# --------------------------------------------------------------- depth
@app.function(image=depth_image, volumes={VOL: volume}, gpu=GPU, timeout=6 * 3600, memory=65536, secrets=secrets)
def depth_dynerf_remote(scene: str, camera: str = "cam00", seed: int = 1234):
    """ChronoDepth on the frames of ONE camera of a DyNeRF scene; previews, statistics and the
    depth video are tracked in W&B as chronodepth-prior/<scene>/<camera>."""
    sdir = scene_dir(scene)
    out = f"{sdir}/depth_chronodepth/{camera}"
    sh(f"python /root/scripts/run_chronodepth.py --datadir {sdir} --frames_dir {sdir}/{camera}/images "
       f"--out_dir {out} --chronodepth_repo /opt/ChronoDepth --repo_url {CHRONODEPTH_URL} --seed {seed} "
       f"--scene_name {scene}/{camera}", env={"WANDB_PROJECT": WANDB_PROJECT, "WANDB_SILENT": "true"})
    volume.commit()
    with open(f"{out}/chronodepth_settings.json") as f:
        return json.load(f)


@app.function(image=depth_image, volumes={VOL: volume}, gpu=GPU, timeout=6 * 3600, memory=65536, secrets=secrets)
def depth_remote(scene: str, seed: int = 1234, wandb_only: bool = False):
    sdir = scene_dir(scene)
    sh(f"python /root/scripts/run_chronodepth.py --datadir {sdir} --chronodepth_repo /opt/ChronoDepth "
       f"--repo_url {CHRONODEPTH_URL} --seed {seed} --scene_name {scene}" + (" --wandb_only" if wandb_only else ""),
       env={"WANDB_PROJECT": WANDB_PROJECT, "WANDB_SILENT": "true"})
    volume.commit()
    with open(f"{sdir}/depth/2x/chronodepth_settings.json") as f:
        return json.load(f)


@app.function(image=depth_image, volumes={VOL: volume}, gpu=GPU, timeout=3 * 3600, memory=32768)
def stability_remote(scene: str):
    """Per-frame Depth Anything V2 maps, then the temporal-stability number for both estimators."""
    sdir = scene_dir(scene)
    if not os.path.exists(f"{sdir}/depth_da2/2x/depth_anything_settings.json"):
        sh(f"python /root/scripts/run_depth_anything.py --datadir {sdir}")
    sh(f"python /root/scripts/depth_temporal_stability.py {sdir} depth/2x depth_da2/2x")
    volume.commit()
    with open(f"{sdir}/depth_temporal_stability.json") as f:
        return json.load(f)


@app.function(image=train_image, volumes={VOL: volume}, timeout=1800)
def collect_remote():
    sh(f"{PY37} scripts/collect_results.py {VOL}/output {VOL}/output/results.csv", cwd="/root/4dgs")
    volume.commit()
    with open(f"{VOL}/output/results.csv") as f:
        return f.read()


# --------------------------------------------------------------- tests
@app.function(image=train_image, gpu=GPU, timeout=1800)
def tests_remote():
    out = sh(f"{PY37} -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)\"")
    out += sh(f"{PY37} -m pytest tests -q", cwd="/root/4dgs")
    return out


# ------------------------------------------------------- train + eval
class GpuMemSampler:
    """Samples GPU and container memory from outside the training process, so it works
    for the unmodified baseline too. Writes one CSV row per interval and a summary."""

    def __init__(self, out_dir, interval=2.0):
        import threading
        self.csv, self.summary_path, self.interval = f"{out_dir}/gpu_mem.csv", f"{out_dir}/gpu_mem_summary.json", interval
        self.stop_flag, self.thread = threading.Event(), threading.Thread(target=self._loop, daemon=True)
        self.peak_used, self.total, self.peak_host, self.samples, self.last = 0, None, 0, 0, None

    @staticmethod
    def _host_bytes():
        for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
            try:
                with open(path) as f:
                    return int(f.read().strip())
            except Exception:
                pass
        return -1

    def _loop(self):
        t0 = time.time()
        with open(self.csv, "w") as f:
            f.write("seconds,gpu_used_mib,gpu_total_mib,gpu_util_pct,container_mem_mib\n")
            while not self.stop_flag.is_set():
                try:
                    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                    used, total, util = [int(float(x)) for x in q.stdout.strip().splitlines()[0].split(",")]
                    host = self._host_bytes()
                    host_mib = host // (1024 * 1024) if host >= 0 else -1
                    self.peak_used, self.total = max(self.peak_used, used), total
                    self.peak_host, self.samples = max(self.peak_host, host_mib), self.samples + 1
                    self.last = {"seconds": round(time.time() - t0, 1), "gpu_used_mib": used, "container_mem_mib": host_mib}
                    f.write(f"{time.time() - t0:.1f},{used},{total},{util},{host_mib}\n")
                    f.flush()
                except Exception:
                    pass
                self.stop_flag.wait(self.interval)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_flag.set()
        self.thread.join(timeout=15)
        summary = {"peak_gpu_used_mib": self.peak_used, "gpu_total_mib": self.total,
                   "peak_container_mem_mib": self.peak_host, "samples": self.samples,
                   "interval_seconds": self.interval, "last_sample": self.last,
                   "note": "nvidia-smi memory.used for the whole GPU (includes the PyTorch cache), sampled externally"}
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=1)
        return summary


@app.function(image=train_image, volumes={VOL: volume}, gpu=GPU, timeout=8 * 3600, cpu=8, memory=49152, max_containers=MAX_RUN_GPUS,
              secrets=secrets)
def run_remote(spec: dict):
    """One run: train, render the full test set, compute metrics.

    spec keys: name, scene, config (path inside the repo), seed (int or None),
               code ("branch" | "base"), commit, extra_args (optional string)
    """
    code_dir = "/root/4dgs_base" if spec.get("code") == "base" else "/root/4dgs"
    sdir = scene_dir(spec["scene"])
    out = f"{VOL}/output/{spec['name']}"
    status = {"name": spec["name"], "status": "failed", "model_path": out}
    if os.path.exists(f"{out}/results.json") and not spec.get("force"):
        status["status"] = "ok"
        status["skipped_existing"] = True
        return status
    sh(f"rm -rf {out} && mkdir -p {out}")
    env = {"GIT_COMMIT": spec.get("commit", "unknown"), "WANDB_SILENT": "true", "WANDB_PROJECT": WANDB_PROJECT}
    if spec.get("cuda_blocking"):
        env["CUDA_LAUNCH_BLOCKING"] = "1"     # debugging: CUDA errors are raised at the call that failed
    if spec.get("no_run_logging"):
        env["FDGS_NO_RUN_LOGGING"] = "1"
    if spec.get("no_wandb"):
        env["WANDB_API_KEY"] = ""          # tracking off for this run (local files only)
    seed = "" if spec.get("seed") is None or spec.get("code") == "base" else f"--seed {spec['seed']}"
    cfg = spec["config"]
    sampler = GpuMemSampler(out).start()
    try:
        sh(f"{PY37} train.py -s {sdir}/ --port 6017 --expname {spec['name']} --model_path {out} "
           f"--configs {cfg} {seed} {spec.get('extra_args', '')}", cwd=code_dir, env=env, log=f"{out}/train_stdout.txt")
        sh(f"{PY37} render.py --model_path {out} --skip_train --configs {cfg}", cwd=code_dir, env=env,
           log=f"{out}/render_stdout.txt")
        sh(f"{PY37} metrics.py --model_path {out}/", cwd=code_dir, env=env, log=f"{out}/metrics_stdout.txt")
        status["status"] = "ok" if os.path.exists(f"{out}/results.json") else "failed"
    except Exception as e:
        status["error"] = str(e)[:500]
    status["gpu_mem"] = sampler.stop()
    with open(f"{out}/git_commit_launch.txt", "w") as f:
        f.write(f"{spec.get('commit', 'unknown')} code={spec.get('code', 'branch')}\n")
    volume.commit()
    return status


@app.function(image=train_image, timeout=20 * 3600)
def dynerf_pipeline_remote(scene: str, camera: str, spec: dict):
    """Preprocessing, ChronoDepth on one camera, then one training run, one after the other,
    so the whole chain holds one GPU at a time and does not depend on the local machine."""
    out = {"scene": scene, "camera": camera, "stages": ""}
    out["prep"] = prep_dynerf_remote.remote(scene)
    out["stages"] += "prep ok; "
    try:
        d = depth_dynerf_remote.remote(scene, camera)
        out["depth"] = {k: d.get(k) for k in ("num_frames", "inference_resolution_hw", "inference_seconds")}
        out["stages"] += "depth ok; "
    except Exception as e:                      # the training run does not need the depth maps
        out["stages"] += "depth FAILED: " + str(e)[:300] + "; "
    out["run"] = run_remote.remote(spec)
    out["stages"] += "run " + str(out["run"].get("status")) + " " + str(out["run"].get("error", ""))[:300]
    return out


@app.function(image=train_image, volumes={VOL: volume}, gpu=GPU, timeout=3600, memory=32768)
def space_remote(scene: str, run_name: str, config: str):
    """Dense depth-vs-disparity check of the prior against a trained baseline model."""
    sdir = scene_dir(scene)
    sh(f"{PY37} scripts/check_prior_space.py --model_path {VOL}/output/{run_name} --configs {config} "
       f"--prior_dir {sdir}/depth/2x", cwd="/root/4dgs")
    volume.commit()
    with open(f"{sdir}/depth/2x/prior_space_check.json") as f:
        out = json.load(f)
    out.pop("per_frame", None)
    return out


@app.function(image=train_image, volumes={VOL: volume}, gpu=GPU, timeout=3600, memory=32768)
def diagnostics_remote(run_name: str, baseline_name: str, config: str):
    sh(f"{PY37} scripts/diagnostics.py --model_path {VOL}/output/{run_name} "
       f"--baseline_path {VOL}/output/{baseline_name} --configs {config}", cwd="/root/4dgs")
    volume.commit()
    return {"diagnostics": f"output/{run_name}/diagnostics"}


# ------------------------------------------------------- entry points
def local_commit():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO).decode().strip()
    return head + ("+dirty" if dirty else "")


@app.local_entrypoint()
def prep(scene: str = "americano", force: bool = False):
    print(json.dumps(prep_remote.remote(scene, force), indent=1))


@app.local_entrypoint()
def depth(scene: str = "americano", seed: int = 1234, wandb_only: bool = False):
    s = depth_remote.remote(scene, seed, wandb_only)
    print(json.dumps({k: v for k, v in s.items() if not k.startswith("per_frame")}, indent=1)[:4000])


@app.local_entrypoint()
def stability(scene: str = "americano"):
    print(json.dumps(stability_remote.remote(scene)["results"], indent=1))


@app.local_entrypoint()
def collect(out: str = "results.csv"):
    with open(out, "w", newline="") as f:
        f.write(collect_remote.remote())
    print("wrote", out)


@app.local_entrypoint()
def tests():
    print(tests_remote.remote())


@app.local_entrypoint()
def runs(spec: str):
    """Launch every run of a JSON spec file (a list of run dicts) in parallel."""
    with open(spec) as f:
        specs = json.load(f)
    commit = local_commit()
    for s in specs:
        s.setdefault("commit", "205090b68d8bc58ab0273c2e564c246204532903" if s.get("code") == "base" else commit)
    for result in run_remote.map(specs, return_exceptions=True):
        print(result)
