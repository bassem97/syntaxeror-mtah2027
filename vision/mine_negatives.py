"""Hard negatives: wherever the current detector fires away from the ball, cut a crop and label it
"no ball". Random negatives are mostly plain grass; the mistakes that matter are socks, boots and
line crossings, and the detector itself is the best way to find them.

    python vision/mine_negatives.py --every 8 --per-frame 2     # appends to crops/labels.csv

Re-run vision/train.py afterwards.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession.games import asset_paths  # noqa: E402
from vision.autolabels import load_labels  # noqa: E402
from vision.dataset import CROPS, crop_at, frames  # noqa: E402
from vision.model import BallNet, topk_peaks  # noqa: E402
from vision.train import WEIGHTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--per-frame", type=int, default=2)
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--far-px", type=float, default=48.0, help="a peak this far from the label is a false one")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BallNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    u, v, _ = load_labels()
    os.makedirs(os.path.join(CROPS, "hard"), exist_ok=True)
    rng = np.random.default_rng(1)
    rows, n_frames = [], 0
    batch, idx = [], []

    def flush():
        nonlocal batch, idx
        if not batch:
            return
        with torch.no_grad():
            t = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device)
            pu, pv, ps = topk_peaks(model(t), 8)
        pu, pv, ps = pu.cpu().numpy(), pv.cpu().numpy(), ps.cpu().numpy()
        for k, i in enumerate(idx):
            taken = 0
            for j in range(pu.shape[1]):
                if ps[k, j] < args.min_score or taken >= args.per_frame:
                    break
                if np.hypot(pu[k, j] - u[i], pv[k, j] - v[i]) < args.far_px:
                    continue
                jx, jy = rng.integers(-args.size // 3, args.size // 3, size=2)
                crop, _, _ = crop_at(batch[k], pu[k, j] + jx, pv[k, j] + jy, args.size)
                name = "hard/{:06d}_{}.jpg".format(i, j)
                cv2.imwrite(os.path.join(CROPS, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append((name, i, -1, -1, 0))
                taken += 1
        batch, idx = [], []

    gen = frames(asset_paths()["video"], step=args.every)
    for i, img in gen:
        if np.isnan(u[i]):
            continue
        batch.append(img)
        idx.append(i)
        n_frames += 1
        if len(batch) == 4:
            flush()
        if args.limit and n_frames >= args.limit:
            break
        if n_frames % 2000 == 0:
            print("  {} frames, {} hard negatives".format(n_frames, len(rows)), flush=True)
    gen.close()
    flush()
    with open(os.path.join(CROPS, "labels.csv"), "a", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print("appended {} hard negatives from {} frames to labels.csv".format(len(rows), n_frames))


if __name__ == "__main__":
    main()
