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


def track(u, v, ku, kv, ks, lo, hi, min_score, max_step, miss_cost, max_gap_frames=None):
    """Follow the ball through each gap with dynamic programming over the top-k peaks.

    States per frame: the k peaks plus "lost". Moving between peaks costs the distance in pixels
    (a ball moves a bounded number of pixels per frame); a peak costs (1 - score); "lost" costs
    miss_cost per frame. Anchored on the tracked ball at both ends when available. Fills u, v in
    place where the path goes through a real peak; returns the mask of filled frames.
    """
    known = ~np.isnan(u)
    filled = np.zeros(len(u), bool)
    K = ku.shape[1]
    for a, b in gaps(known, lo, hi):
        if max_gap_frames and b - a + 1 > max_gap_frames:
            continue
        if np.isnan(ku[a:b + 1, 0]).any():
            continue  # detector did not run on the whole gap
        p0 = (u[a - 1], v[a - 1]) if a - 1 >= 0 and known[a - 1] else None
        p1 = (u[b + 1], v[b + 1]) if b + 1 < len(u) and known[b + 1] else None
        n = b - a + 1
        INF = 1e18
        cost = np.full((n, K + 1), INF)
        back = np.zeros((n, K + 1), dtype=np.int16)
        px = np.where(ks[a:b + 1] >= min_score, ku[a:b + 1], np.nan)
        py = np.where(ks[a:b + 1] >= min_score, kv[a:b + 1], np.nan)
        emit = np.where(ks[a:b + 1] >= min_score, 1.0 - ks[a:b + 1], INF)
        # first frame: distance from the anchor before the gap
        for j in range(K):
            if emit[0, j] < INF:
                dist = np.hypot(px[0, j] - p0[0], py[0, j] - p0[1]) if p0 else 0.0
                cost[0, j] = emit[0, j] + (dist / max_step if dist <= 2 * max_step else INF)
        cost[0, K] = miss_cost
        for t in range(1, n):
            for j in range(K + 1):
                if j < K and emit[t, j] >= INF:
                    continue
                best, arg = INF, K
                for i in range(K + 1):
                    if cost[t - 1, i] >= INF:
                        continue
                    if j < K and i < K:
                        dist = np.hypot(px[t, j] - px[t - 1, i], py[t, j] - py[t - 1, i])
                        c = cost[t - 1, i] + (dist / max_step if dist <= 2 * max_step else INF)
                    elif j < K:  # coming back from lost: no distance known
                        c = cost[t - 1, i] + 1.0
                    else:
                        c = cost[t - 1, i] + miss_cost
                    if c < best:
                        best, arg = c, i
                cost[t, j] = best + (emit[t, j] if j < K else 0.0)
                back[t, j] = arg
        # last frame: distance to the anchor after the gap
        final = cost[n - 1].copy()
        if p1:
            for j in range(K):
                if final[j] < INF:
                    dist = np.hypot(px[n - 1, j] - p1[0], py[n - 1, j] - p1[1])
                    final[j] += dist / max_step if dist <= 2 * max_step else INF
        j = int(np.argmin(final))
        if final[j] >= INF:
            continue
        for t in range(n - 1, -1, -1):
            if j < K:
                u[a + t], v[a + t] = px[t, j], py[t, j]
                filled[a + t] = True
            j = int(back[t, j]) if t > 0 else j
    return filled


def run(u, v, d, lo, hi, p):
    if p["method"] == "track":
        return track(u, v, d["ku"], d["kv"], d["ks"], lo, hi, p["min_score"], p["max_step"], p["miss_cost"], p["max_gap"])
    return fill(u, v, d["ku"], d["kv"], d["ks"], lo, hi, p["base_px"], p["grow_px"], p["cap_frames"], p["min_score"], p["max_gap"])


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
    filled = run(uu, vv, d, 0, len(u) - 1, p)
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
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", type=int, default=0, help="number of hidden stretches to test on")
    ap.add_argument("--gap-frames", default="25-75", help="hidden stretch length range for --simulate")
    ap.add_argument("--base-px", type=float, default=40)
    ap.add_argument("--grow-px", type=float, default=4)
    ap.add_argument("--cap-frames", type=int, default=40)
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--max-gap-s", type=float, default=8)
    ap.add_argument("--method", default="track", choices=["track", "interp"])
    ap.add_argument("--max-step", type=float, default=30, help="track: typical ball movement per frame, px")
    ap.add_argument("--miss-cost", type=float, default=1.5, help="track: cost of a frame with no peak")
    ap.add_argument("--track", default=OUT, help="detector track file to read and fill")
    args = ap.parse_args()
    OUT = args.track
    p = dict(base_px=args.base_px, grow_px=args.grow_px, cap_frames=args.cap_frames, min_score=args.min_score,
             max_gap=int(args.max_gap_s * FPS), method=args.method, max_step=args.max_step, miss_cost=args.miss_cost)
    if args.simulate:
        lo, hi = (int(x) for x in args.gap_frames.split("-"))
        simulate(args.simulate, (lo, hi), p)
        return
    u, v, _ = load_labels()
    d = dict(np.load(OUT))
    filled = run(u, v, d, 0, len(u) - 1, p)
    table = pd.read_csv(require(["homography"])[0])
    Hs = table[["h{}{}".format(r, c) for r in range(3) for c in range(3)]].values.reshape(-1, 3, 3)
    x, y = np.full(len(u), np.nan), np.full(len(u), np.nan)
    for t in np.flatnonzero(filled):
        x[t], y[t] = unproject(Hs[t], u[t], v[t])
    d["x"], d["y"] = x, y
    # confidence of the peak actually chosen (the tracker may take any of the k), 0 elsewhere
    score = np.zeros(len(u), dtype=np.float32)
    for t in np.flatnonzero(filled):
        j = int(np.argmin(np.abs(d["ku"][t] - u[t]) + np.abs(d["kv"][t] - v[t])))
        score[t] = d["ks"][t, j]
    d["score"] = score
    np.savez_compressed(OUT, **d)
    print("filled {} frames; wrote {}".format(filled.sum(), OUT))


if __name__ == "__main__":
    main()
