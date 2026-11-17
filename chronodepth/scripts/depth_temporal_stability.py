"""Temporal stability of a depth prior over one sequence.

    python scripts/depth_temporal_stability.py <datadir> <depth_subdir> [<depth_subdir> ...]
    e.g.  ... data/hypernerf/misc/americano depth/2x depth_da2/2x

Definition (one number per estimator and scene)
    1. Take all frames of dataset.json['ids'] in order.
    2. Apply ONE affine map to the whole sequence: the 2nd and 98th percentile
       of all its values (subsampled) are mapped to 0 and 1.
    3. Report the mean over consecutive frame pairs and pixels of
       |d_{t+1}(p) - d_t(p)|.
The number is computed in each estimator's own output space, so real scene
and camera motion contribute to it for both estimators; it is a relative
measure between estimators on the same sequence, not an absolute one.
"""
import json
import os
import sys

import numpy as np


def stability(datadir, sub):
    with open(os.path.join(datadir, "dataset.json")) as f:
        ids = json.load(f)["ids"]
    load = lambda i: np.load(os.path.join(datadir, sub, f"{i}.npy")).astype(np.float64)
    sample = np.concatenate([load(i)[::8, ::8].ravel() for i in ids[::4]])
    lo, hi = np.percentile(sample, [2, 98])
    per_pair = []
    prev = (load(ids[0]) - lo) / (hi - lo)
    for i in ids[1:]:
        cur = (load(i) - lo) / (hi - lo)
        per_pair.append(float(np.abs(cur - prev).mean()))
        prev = cur
    return {"depth_dir": sub, "num_frames": len(ids), "p2": float(lo), "p98": float(hi),
            "mean_abs_frame_to_frame_change": float(np.mean(per_pair)),
            "median_over_pairs": float(np.median(per_pair)),
            "max_over_pairs": float(np.max(per_pair))}


def main():
    datadir = sys.argv[1]
    out = {"definition": __doc__.split("Definition")[1].strip(),
           "results": [stability(datadir, s) for s in sys.argv[2:]]}
    path = os.path.join(datadir, "depth_temporal_stability.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["results"], indent=1))


if __name__ == "__main__":
    main()
