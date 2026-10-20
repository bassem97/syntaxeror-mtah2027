"""Fill the frames where the tracker lost the ball, using the detector's peaks near where the ball
must be. The tracker knows the ball before and after every gap, so the search is local: a window
around the straight line between the two ends, widening with distance from them. That removes
most of the detector's false peaks (socks, boots, line crossings) without retraining it.

    python vision/fill_gaps.py                 # -> data/cache/ball_detector.npz gains x, y, score for filled frames
    python vision/fill_gaps.py --simulate 300  # hide 300 tracked stretches, measure recovery

Only frames inside a gap that the detector ran on are filled; gaps longer than --max-gap-s are
left alone (the straight-line prior is meaningless across a stoppage).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession.games import require  # noqa: E402
from vision.autolabels import load_labels  # noqa: E402
from vision.infer import OUT, unproject  # noqa: E402

FPS = 25


def gaps(known, lo, hi):
    """[(a, b)] maximal runs of unknown frames within [lo, hi], inclusive."""
    out = []
    i = lo
    while i <= hi:
        if known[i]:
            i += 1
            continue
        j = i
        while j + 1 <= hi and not known[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def fill(u, v, ku, kv, ks, lo, hi, base_px, grow_px, cap_frames, min_score, max_gap):
    """Fill NaNs of u, v in place from the top-k peaks; returns a bool mask of filled frames."""
    known = ~np.isnan(u)
    filled = np.zeros(len(u), bool)
    for a, b in gaps(known, lo, hi):
        if b - a + 1 > max_gap:
            continue
        p0 = (u[a - 1], v[a - 1]) if a - 1 >= 0 and known[a - 1] else None
        p1 = (u[b + 1], v[b + 1]) if b + 1 < len(u) and known[b + 1] else None
        if p0 is None and p1 is None:
            continue
        for t in range(a, b + 1):
            if np.isnan(ku[t, 0]):
                continue  # detector did not run here
            if p0 is not None and p1 is not None:
                w = (t - (a - 1)) / float(b + 2 - a)
                iu, iv = p0[0] + w * (p1[0] - p0[0]), p0[1] + w * (p1[1] - p0[1])
                edge = min(t - a + 1, b - t + 1)
            else:
                iu, iv = p0 if p0 is not None else p1
                edge = (t - a + 1) if p0 is not None else (b - t + 1)
            radius = base_px + grow_px * min(edge, cap_frames)
            d = np.hypot(ku[t] - iu, kv[t] - iv)
            ok = (d <= radius) & (ks[t] >= min_score)
            if not ok.any():
                continue
            # best score among candidates, mildly preferring the nearer one
            j = np.argmax(np.where(ok, ks[t] - 0.1 * d / radius, -np.inf))
            u[t], v[t] = ku[t, j], kv[t, j]
            filled[t] = True
    return filled


def simulate(n_gaps, length_frames, p, seed=0):
    u, v, _ = load_labels()
    d = np.load(OUT)
    ran = ~np.isnan(d["u"])
    rng = np.random.default_rng(seed)
    known = ~np.isnan(u)
    # tracked stretches that the detector ran on, long enough to hide `length` frames inside
    cand = np.flatnonzero(known & ran)
    hidden = []
    tries = 0
    while len(hidden) < n_gaps and tries < 20 * n_gaps:
        tries += 1
        s = int(rng.choice(cand))
        L = int(rng.integers(length_frames[0], length_frames[1] + 1))
        if s + L + 1 >= len(u) or not (known[s - 1:s + L + 1].all() and ran[s:s + L].all()):
            continue
        if any(abs(s - h[0]) < 2 * length_frames[1] for h in hidden):
            continue
        hidden.append((s, s + L - 1))
    uu, vv = u.copy(), v.copy()
    for a, b in hidden:
        uu[a:b + 1] = np.nan
        vv[a:b + 1] = np.nan
    filled = fill(uu, vv, d["ku"], d["kv"], d["ks"], 0, len(u) - 1, **p)
    errs, miss = [], 0
    for a, b in hidden:
        for t in range(a, b + 1):
            if filled[t]:
                errs.append(np.hypot(uu[t] - u[t], vv[t] - v[t]))
            else:
                miss += 1
    errs = np.array(errs)
    tot = sum(b - a + 1 for a, b in hidden)
    print("simulated {} gaps of {}-{} frames ({} frames hidden): filled {:.2f}; of filled, within 8 px {:.2f}, 16 px {:.2f}, 32 px {:.2f}; |err| pct(50,90) {}".format(
        len(hidden), length_frames[0], length_frames[1], tot, 1 - miss / tot,
        (errs <= 8).mean() if len(errs) else 0, (errs <= 16).mean() if len(errs) else 0, (errs <= 32).mean() if len(errs) else 0,
        np.percentile(errs, [50, 90]).round(1) if len(errs) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", type=int, default=0, help="number of hidden stretches to test on")
    ap.add_argument("--gap-frames", default="25-75", help="hidden stretch length range for --simulate")
    ap.add_argument("--base-px", type=float, default=40)
    ap.add_argument("--grow-px", type=float, default=4)
    ap.add_argument("--cap-frames", type=int, default=40)
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--max-gap-s", type=float, default=8)
    args = ap.parse_args()
    p = dict(base_px=args.base_px, grow_px=args.grow_px, cap_frames=args.cap_frames,
             min_score=args.min_score, max_gap=int(args.max_gap_s * FPS))
    if args.simulate:
        lo, hi = (int(x) for x in args.gap_frames.split("-"))
        simulate(args.simulate, (lo, hi), p)
        return
    u, v, _ = load_labels()
    d = dict(np.load(OUT))
    filled = fill(u, v, d["ku"], d["kv"], d["ks"], 0, len(u) - 1, **p)
    table = pd.read_csv(require(["homography"])[0])
    Hs = table[["h{}{}".format(r, c) for r in range(3) for c in range(3)]].values.reshape(-1, 3, 3)
    x, y = np.full(len(u), np.nan), np.full(len(u), np.nan)
    for t in np.flatnonzero(filled):
        x[t], y[t] = unproject(Hs[t], u[t], v[t])
    d["x"], d["y"] = x, y
    d["score"] = np.where(filled, d["score"], 0.0).astype(np.float32)
    np.savez_compressed(OUT, **d)
    print("filled {} frames; wrote {}".format(filled.sum(), OUT))


if __name__ == "__main__":
    main()
