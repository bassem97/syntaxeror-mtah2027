"""Are the auto-labels on the ball? Checked numerically, without anyone looking at frames.

The ball is the brightest small thing on the grass. For a sample of labelled frames, look for the
brightest 5x5 patch within a search radius of the projected point and report how far it is. If the
labels are right, most samples have a bright patch within a few pixels and the median offset is
small and consistent (a constant offset is a homography bias, random large ones are aerial balls or
tracking errors).

    python vision/check_labels.py --n 300
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession.games import asset_paths  # noqa: E402
from vision.autolabels import load_labels  # noqa: E402


def blob_response(gray, small=2.5, large=6.0):
    """Difference of Gaussians: positive for bright blobs a few pixels across (the ball), near
    zero for large bright areas (white shirts, sky) and weak for thin lines."""
    g = gray.astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), small) - cv2.GaussianBlur(g, (0, 0), large)


def brightest_patch(gray, u, v, radius, patch=5):
    """Strongest ball-like blob within `radius` of (u, v): x, y, response, local median response."""
    h, w = gray.shape
    pad = 3 * 6
    x0, x1 = max(0, int(u) - radius - pad), min(w, int(u) + radius + pad + 1)
    y0, y1 = max(0, int(v) - radius - pad), min(h, int(v) + radius + pad + 1)
    if x1 - x0 < 2 * pad + 1 or y1 - y0 < 2 * pad + 1:
        return None
    dog = blob_response(gray[y0:y1, x0:x1])
    inner = dog[pad:-pad, pad:-pad]
    j, i = np.unravel_index(np.argmax(inner), inner.shape)
    return x0 + pad + i, y0 + pad + j, float(inner[j, i]), float(np.median(inner))


def sample(n, radius, seed=0, isolated=False):
    u, v, _ = load_labels()
    ok = ~np.isnan(u)
    if isolated:
        # ball at least 3 m from every player and rolling slowly: the only blob near the label
        # can be the ball, so this measures the projection itself rather than the clutter
        from possession.features import load
        f = load()
        dmin = np.fmin(f["d_home"], f["d_away"])
        speed = np.nan_to_num(f["ball_speed"], nan=99.0)
        ok &= (dmin > 3.0) & (speed < 8.0)
    ok = np.flatnonzero(ok)
    rng = np.random.default_rng(seed)
    frames = np.sort(rng.choice(ok, size=min(n, len(ok)), replace=False))
    cap = cv2.VideoCapture(asset_paths()["video"])
    rows = []
    for fr in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ret, img = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hit = brightest_patch(gray, u[fr], v[fr], radius)
        if hit is None:
            continue
        bx, by, bright, bg = hit
        rows.append((fr, u[fr], v[fr], bx - u[fr], by - v[fr], bright, bg))
    cap.release()
    return np.array(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--radius", type=int, default=20)
    ap.add_argument("--isolated", action="store_true", help="only frames with the ball far from every player")
    args = ap.parse_args()
    r = sample(args.n, args.radius, isolated=args.isolated)
    du, dv, bright, bg = r[:, 3], r[:, 4], r[:, 5], r[:, 6]
    contrast = bright - bg
    dist = np.hypot(du, dv)
    print("{} frames checked, search radius {} px".format(len(r), args.radius))
    print("blob response at the best spot: pct(10,50,90) = {}".format(np.percentile(bright, [10, 50, 90]).round(1)))
    for thr in (8, 12, 20):
        good = bright > thr
        print("response > {:>2}: {:.2f} of frames; offset median du {:+.1f} dv {:+.1f}; |d| <= 4 px {:.2f}, <= 8 px {:.2f}, <= 12 px {:.2f}".format(
            thr, good.mean(), np.median(du[good]), np.median(dv[good]),
            (dist[good] <= 4).mean(), (dist[good] <= 8).mean(), (dist[good] <= 12).mean()))
    print("|d| pct(50,75,90) over all: {}".format(np.percentile(dist, [50, 75, 90]).round(1)))
