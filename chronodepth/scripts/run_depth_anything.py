"""Per-frame Depth Anything V2 maps for one HyperNeRF scene (depth environment).

Used only for the temporal-stability comparison, never in training.
Writes <datadir>/depth_da2/2x/{frame_id}.npy and depth_anything_settings.json.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Large-hf")
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).cuda().eval()

    with open(os.path.join(args.datadir, "dataset.json")) as f:
        ids = json.load(f)["ids"]
    out_dir = os.path.join(args.datadir, "depth_da2", f"{args.factor}x")
    os.makedirs(out_dir, exist_ok=True)
    for i in ids:
        image = Image.open(os.path.join(args.datadir, "rgb", f"{args.factor}x", f"{i}.png")).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            pred = model(**inputs).predicted_depth          # [1, h, w], one independent map per frame
        pred = torch.nn.functional.interpolate(pred[None], size=image.size[::-1], mode="bicubic",
                                               align_corners=False)[0, 0]
        np.save(os.path.join(out_dir, f"{i}.npy"), pred.float().cpu().numpy())
    try:
        from huggingface_hub import model_info
        revision = model_info(args.model).sha
    except Exception:
        revision = "unknown"
    with open(os.path.join(out_dir, "depth_anything_settings.json"), "w") as f:
        json.dump({"estimator": "Depth Anything V2", "checkpoint": args.model, "checkpoint_revision": revision,
                   "num_frames": len(ids), "per_frame": True,
                   "output": "relative inverse depth (model card), each frame predicted independently",
                   "processor": "model default"}, f, indent=1)
    print(len(ids), "frames ->", out_dir)


if __name__ == "__main__":
    main()
