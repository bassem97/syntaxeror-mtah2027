"""How good is the detector's ball track on real frames? Compare it with the tracker where the
tracker has the ball, and look at its confidence where the tracker does not.

    python vision/eval_track.py                 # whole file from vision/infer.py
    python vision/eval_track.py --start 11185 --end 18684

On frames with a tracked ball: distance in pixels between the detector peak and the projected
tracked ball (both are noisy, see vision/check_labels.py, so a few pixels is as good as it gets),
split by whether the ball was isolated. On frames without: the score distribution, which is what
decides how many of the 41% ball-less graded frames the detector can fill.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession.features import load  # noqa: E402
from possession.games import half_video_blocks, GAME  # noqa: E402
from vision.autolabels import load_labels  # noqa: E402
from vision.infer import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--track", default=OUT)
    args = ap.parse_args()
    d = np.load(args.track)
    u, v, _ = load_labels()
    f = load()
    n = len(u)
    end = args.end if args.end is not None else n - 1
    sel = np.zeros(n, bool)
    sel[args.start:end + 1] = True
    sel &= ~np.isnan(d["u"])  # frames the detector ran on
    graded = np.zeros(n, bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    sel &= graded
    tracked = sel & ~np.isnan(u)
    missing = sel & np.isnan(u)
    dmin = np.nan_to_num(np.fmin(f["d_home"], f["d_away"]), nan=0.0)
    dist = np.hypot(d["u"] - u, d["v"] - v)
    print("frames considered: {} graded ({} with tracked ball, {} without)".format(sel.sum(), tracked.sum(), missing.sum()))
    for name, m in (("tracked, isolated (>2 m)", tracked & (dmin > 2)), ("tracked, near a player", tracked & (dmin <= 2))):
        if not m.any():
            continue
        print("  {:<26} n={:>6}  score pct(10,50,90)={}  |peak - tracked| px pct(50,75,90)={}  within 8 px {:.2f}, 16 px {:.2f}".format(
            name, m.sum(), np.percentile(d["score"][m], [10, 50, 90]).round(2), np.percentile(dist[m], [50, 75, 90]).round(1),
            (dist[m] <= 8).mean(), (dist[m] <= 16).mean()))
    for thr in (0.3, 0.5, 0.7):
        print("  score >= {:.1f}: tracked frames {:.2f} (of which within 16 px {:.2f}), ball-less frames {:.2f}".format(
            thr, (d["score"][tracked] >= thr).mean(),
            (dist[tracked & (d["score"] >= thr)] <= 16).mean() if (tracked & (d["score"] >= thr)).any() else 0,
            (d["score"][missing] >= thr).mean() if missing.any() else 0))


if __name__ == "__main__":
    main()
