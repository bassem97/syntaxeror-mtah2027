"""Test the detector where it matters: on frames a human clicked the ball in (labeling/data/clicks.csv),
which should mostly be frames the tracker lost.

    python vision/eval_clicks.py                       # uses cache/ball_detector.npz
    python vision/eval_clicks.py --track other.npz

Reports, for frames the detector ran on: how far its top peak and its best of five peaks are from
the click, and how many clicked frames the gap filler actually filled and how far off.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from vision.autolabels import load_labels  # noqa: E402
from vision.infer import OUT  # noqa: E402

CLICKS = os.path.join(ROOT, "labeling", "data", "clicks.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default=OUT)
    ap.add_argument("--clicks", default=CLICKS)
    args = ap.parse_args()
    with open(args.clicks, newline="") as fh:
        rows = {int(r["frame"]): (float(r["u"]), float(r["v"])) for r in csv.DictReader(fh)}  # last click per frame wins
    d = np.load(args.track)
    u_tr, _, _ = load_labels()
    frames = np.array(sorted(rows))
    cu = np.array([rows[f][0] for f in frames])
    cv = np.array([rows[f][1] for f in frames])
    ran = ~np.isnan(d["ku"][frames, 0])
    lost = np.isnan(u_tr[frames])
    print("{} clicked frames; tracker had no ball on {}; detector ran on {}".format(len(frames), lost.sum(), ran.sum()))
    m = ran
    if not m.any():
        return
    top = np.hypot(d["u"][frames][m] - cu[m], d["v"][frames][m] - cv[m])
    best5 = np.min(np.hypot(d["ku"][frames][m] - cu[m, None], d["kv"][frames][m] - cv[m, None]), axis=1)
    print("top peak:     within 8 px {:.2f}, 16 px {:.2f}, 32 px {:.2f}; median {:.1f} px".format(
        (top <= 8).mean(), (top <= 16).mean(), (top <= 32).mean(), np.median(top)))
    print("best of five: within 8 px {:.2f}, 16 px {:.2f}, 32 px {:.2f}; median {:.1f} px".format(
        (best5 <= 8).mean(), (best5 <= 16).mean(), (best5 <= 32).mean(), np.median(best5)))
    filled = ~np.isnan(d["x"][frames]) & lost
    if filled.any():
        fu = d["u"][frames][filled]
        fv = d["v"][frames][filled]
        err = np.hypot(fu - cu[filled], fv - cv[filled])
        print("gap filler: filled {} of {} tracker-lost clicked frames; of those within 16 px {:.2f}, 32 px {:.2f}; median {:.1f} px".format(
            filled.sum(), lost.sum(), (err <= 16).mean(), (err <= 32).mean(), np.median(err)))
    else:
        print("gap filler filled none of the clicked frames")


if __name__ == "__main__":
    main()
