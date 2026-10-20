"""Run the ball detector over the whole tactical video and write a second ball track.

    python vision/infer.py                     # -> data/cache/ball_detector.npz
    python vision/infer.py --start 11185 --end 18684   # a stretch, for checking

Per frame: u, v (pixels of the heatmap peak) and score (peak probability). Pitch coordinates
come from the inverse homography of that frame, in the same normalised 0-1 units as the
tracking, so possession/features.py can use the detector wherever the tracker lost the ball.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession.games import DATA_DIR, asset_paths, require, source_frames  # noqa: E402
from vision.model import BallNet, peaks  # noqa: E402
from vision.train import WEIGHTS  # noqa: E402

OUT = os.path.join(DATA_DIR, "cache", "ball_detector.npz")


def decode(video, start, end):
    """(frame_index, BGR ndarray) for frames start..end, decoded in-process with PyAV."""
    import av
    c = av.open(video)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    if start > 0:
        # seek to the keyframe before `start`, then decode forward to it
        c.seek(int(start / float(s.average_rate) / float(s.time_base)), stream=s, backward=True, any_frame=False)
    i = None
    for fr in c.decode(s):
        i = int(round(float(fr.pts * s.time_base) * float(s.average_rate))) if i is None else i + 1
        if i < start:
            continue
        if i > end:
            break
        yield i, fr.to_ndarray(format="bgr24")
    c.close()


def unproject(H, u, v):
    """Pixels -> normalised pitch coordinates with the inverse homography."""
    Hi = np.linalg.inv(H)
    den = Hi[2, 0] * u + Hi[2, 1] * v + Hi[2, 2]
    return (Hi[0, 0] * u + Hi[0, 1] * v + Hi[0, 2]) / den, (Hi[1, 0] * u + Hi[1, 1] * v + Hi[1, 2]) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=source_frames("tactical") - 1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BallNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    table = pd.read_csv(require(["homography"])[0])
    Hs = table[["h{}{}".format(r, c) for r in range(3) for c in range(3)]].values.reshape(-1, 3, 3)
    n = source_frames("tactical")
    u = np.full(n, np.nan)
    v = np.full(n, np.nan)
    score = np.zeros(n, dtype=np.float32)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    if os.path.isfile(args.out):  # a partial run can be extended
        d = np.load(args.out)
        u, v, score, x, y = d["u"], d["v"], d["score"], d["x"], d["y"]

    t0 = time.time()
    batch, idx = [], []
    done = 0

    def flush():
        nonlocal batch, idx
        if not batch:
            return
        with torch.no_grad():
            t = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device, non_blocking=True)
            pu, pv, ps = peaks(model(t))
        for k, i in enumerate(idx):
            u[i], v[i], score[i] = pu[k].item(), pv[k].item(), ps[k].item()
            x[i], y[i] = unproject(Hs[i], u[i], v[i])
        batch, idx = [], []

    for i, img in decode(asset_paths()["video"], args.start, args.end):
        batch.append(img)
        idx.append(i)
        if len(batch) == args.batch:
            flush()
        done += 1
        if done % 5000 == 0:
            print("  {} frames, {:.0f} fps".format(done, done / (time.time() - t0)), flush=True)
            np.savez_compressed(args.out, u=u, v=v, score=score, x=x, y=y)
    flush()
    np.savez_compressed(args.out, u=u, v=v, score=score, x=x, y=y)
    sl = slice(args.start, args.end + 1)
    print("wrote {}: {} frames in {:.0f}s; score pct(10,50,90) = {}".format(
        args.out, done, time.time() - t0, np.percentile(score[sl], [10, 50, 90]).round(2)))


if __name__ == "__main__":
    main()
