"""Auto-labels for a ball detector: where the tracked ball is in each video frame, in pixels.

The tactical tracking has the ball on 57% of frames and the homography is valid on every frame,
so projecting the tracked ball through the homography gives a pixel position for free on about
113,000 frames. Those are the training labels; the detector is then applied to the frames where
the tracker lost the ball.

    python vision/autolabels.py        # -> data/cache/ball_pixels.npz

Arrays, all length n_frames: u, v (pixels, NaN where no ball), and the reprojection error of the
frame's homography. The projection assumes the ball is on the ground; aerial balls land a few
pixels off, which vision/check_labels.py measures and vision/dataset.py corrects.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession.features import load  # noqa: E402
from possession.games import DATA_DIR, require  # noqa: E402

OUT = os.path.join(DATA_DIR, "cache", "ball_pixels.npz")
WIDTH, HEIGHT = 1920, 1080


def build():
    f = load()
    path = require(["homography"])[0]
    table = pd.read_csv(path)
    H = table[["h{}{}".format(r, c) for r in range(3) for c in range(3)]].values.reshape(-1, 3, 3)
    assert len(H) == len(f["ball_x"])
    x, y = f["ball_x"], f["ball_y"]
    den = H[:, 2, 0] * x + H[:, 2, 1] * y + H[:, 2, 2]
    u = (H[:, 0, 0] * x + H[:, 0, 1] * y + H[:, 0, 2]) / den
    v = (H[:, 1, 0] * x + H[:, 1, 1] * y + H[:, 1, 2]) / den
    inside = (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT)
    u[~inside] = np.nan
    v[~inside] = np.nan
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, u=u, v=v, reproj=table["reprojection_error"].values)
    return u, v


def load_labels():
    if not os.path.isfile(OUT):
        build()
    d = np.load(OUT)
    return d["u"], d["v"], d["reproj"]


if __name__ == "__main__":
    u, v = build()
    ok = ~np.isnan(u)
    print("wrote", OUT)
    print("ball pixel labels on {} frames ({:.1f}%)".format(ok.sum(), 100 * ok.mean()))
    print("u pct(5,50,95) {}  v pct(5,50,95) {}".format(
        np.nanpercentile(u, [5, 50, 95]).round(0), np.nanpercentile(v, [5, 50, 95]).round(0)))
