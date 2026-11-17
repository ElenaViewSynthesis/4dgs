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

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BASELINE_EXPORT = os.path.join(os.path.dirname(REPO), "_baseline_205090b")
VOL = "/vol"
PY37 = "/opt/env37/bin/python"
CHRONODEPTH_URL = "https://github.com/jhaoshao/ChronoDepth"
CHRONODEPTH_COMMIT = "2580e891eb33a5c3a95eab27b745f64ac28b3514"
GPU = os.environ.get("FDGS_GPU", "A100-40GB")

SCENES = {
    "americano": {"zip": "misc_americano.zip", "group": "misc"},
    "aleks-teapot": {"zip": "interp_aleks-teapot.zip", "group": "interp"},
}
RELEASE = "https://github.com/google/hypernerf/releases/download/v0.1"

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


# --------------------------------------------------------------- depth
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
@app.function(image=train_image, volumes={VOL: volume}, gpu=GPU, timeout=8 * 3600, cpu=8, memory=49152,
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
    if spec.get("no_run_logging"):
        env["FDGS_NO_RUN_LOGGING"] = "1"
    if spec.get("no_wandb"):
        env["WANDB_API_KEY"] = ""          # tracking off for this run (local files only)
    seed = "" if spec.get("seed") is None or spec.get("code") == "base" else f"--seed {spec['seed']}"
    cfg = spec["config"]
    try:
        sh(f"{PY37} train.py -s {sdir}/ --port 6017 --expname {spec['name']} --model_path {out} "
           f"--configs {cfg} {seed} {spec.get('extra_args', '')}", cwd=code_dir, env=env, log=f"{out}/train_stdout.txt")
        sh(f"{PY37} render.py --model_path {out} --skip_train --configs {cfg}", cwd=code_dir, env=env,
           log=f"{out}/render_stdout.txt")
        sh(f"{PY37} metrics.py --model_path {out}/", cwd=code_dir, env=env, log=f"{out}/metrics_stdout.txt")
        status["status"] = "ok" if os.path.exists(f"{out}/results.json") else "failed"
    except Exception as e:
        status["error"] = str(e)[:500]
    with open(f"{out}/git_commit_launch.txt", "w") as f:
        f.write(f"{spec.get('commit', 'unknown')} code={spec.get('code', 'branch')}\n")
    volume.commit()
    return status


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
