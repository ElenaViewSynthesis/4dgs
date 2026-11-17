"""Submit jobs to the deployed app and return immediately.

The local process only submits; the jobs run to completion remotely even if
this machine sleeps or the terminal closes. Deploy first, after every code
change:   modal deploy remote/app.py

    python remote/launch.py prep americano
    python remote/launch.py depth americano [--wandb-only]
    python remote/launch.py stability americano
    python remote/launch.py dynerf coffee_martini [cam00]   # prep, ChronoDepth on one camera, one run
    python remote/launch.py runs remote/specs/b0.json
    python remote/launch.py status            # state and result of every submitted job
    python remote/launch.py collect results.csv

Submitted call ids are kept in ../_launches.json (outside the repository).
"""
import json
import os
import subprocess
import sys
import time

import modal

APP = "fdgs-depth-reg"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LEDGER = os.path.join(os.path.dirname(REPO), "_launches.json")
BASE_COMMIT = "205090b68d8bc58ab0273c2e564c246204532903"


def ledger():
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            return json.load(f)
    return []


def record(kind, label, call, extra=None):
    rows = ledger()
    rows.append({"kind": kind, "label": label, "call_id": call.object_id,
                 "submitted": time.strftime("%Y-%m-%d %H:%M:%S"), "extra": extra or {}})
    with open(LEDGER, "w") as f:
        json.dump(rows, f, indent=1)
    print("submitted", kind, label, call.object_id)


def commit():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO).decode().strip()
    return head + ("+dirty" if dirty else "")


def fn(name):
    return modal.Function.from_name(APP, name)


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "prep":
        record("prep", args[0], fn("prep_remote").spawn(args[0], "--force" in args))
    elif cmd == "depth":
        record("depth", args[0], fn("depth_remote").spawn(args[0], 1234, "--wandb-only" in args))
    elif cmd == "dynerf":         # dynerf <scene> [camera]: prep, ChronoDepth on one camera, one training run
        head = commit()
        if head.endswith("+dirty"):
            sys.exit("refusing to launch runs from a modified working tree: commit and redeploy first")
        camera = args[1] if len(args) > 1 else "cam00"
        spec = {"name": "dynerf/{}_branch_flags_off".format(args[0]), "scene": args[0],
                "config": "arguments/dynerf/{}.py".format(args[0]), "code": "branch", "seed": None, "commit": head}
        record("dynerf", args[0], fn("dynerf_pipeline_remote").spawn(args[0], camera, spec),
               {"spec": spec, "camera": camera})
    elif cmd == "depthall":       # depthall <scene> [camera tracked in W&B]: ChronoDepth for every camera, one GPU
        cam = args[1] if len(args) > 1 else "cam16"
        record("depthall", args[0], fn("depth_dynerf_all_remote").spawn(args[0], cam), {"wandb_camera": cam})
    elif cmd == "stability":
        record("stability", args[0], fn("stability_remote").spawn(args[0]))
    elif cmd == "space":          # space <scene> <baseline run name> <config>
        record("space", args[0], fn("space_remote").spawn(args[0], args[1], args[2], int(args[3]) if len(args) > 3 else 32))
    elif cmd == "diagnostics":    # diagnostics <run name> <baseline run name> <config>
        record("diagnostics", args[0], fn("diagnostics_remote").spawn(args[0], args[1], args[2]))
    elif cmd == "tests":
        record("tests", "unit", fn("tests_remote").spawn())
    elif cmd == "runs":
        with open(args[0]) as f:
            specs = json.load(f)
        head = commit()
        if head.endswith("+dirty"):
            sys.exit("refusing to launch runs from a modified working tree: commit and redeploy first")
        only = set(args[1:])
        for s in specs:
            if only and s["name"] not in only:
                continue
            s.setdefault("commit", BASE_COMMIT if s.get("code") == "base" else head)
            record("run", s["name"], fn("run_remote").spawn(s), {"spec": s})
    elif cmd == "collect":
        out = args[0] if args else "results.csv"
        with open(out, "w", newline="") as f:
            f.write(fn("collect_remote").remote())
        print("wrote", out)
    elif cmd == "status":
        for row in ledger():
            call = modal.FunctionCall.from_id(row["call_id"])
            try:
                result = call.get(timeout=0)
                state = "done"
                if isinstance(result, dict):
                    result = {k: v for k, v in result.items() if not isinstance(v, (dict, list))}
                text = json.dumps(result, default=str)[:300] if not isinstance(result, str) else result[-300:]
            except TimeoutError:
                state, text = "running", ""
            except Exception as e:
                state, text = "error", "{}: {}".format(type(e).__name__, str(e)[:300])
            print("{:8} {:10} {:45} {} {}".format(state, row["kind"], row["label"], row["submitted"], text))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
