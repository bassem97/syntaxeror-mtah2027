"""Where does the missing spellF1 go? Classify every true and predicted possession.

    python analysis/error_budget.py out/SyntaxEror.csv labeling/data/gt_internal.csv

A true possession is "found" when a predicted possession of the same team overlaps it with
IoU >= 0.5 (the scorer's middle threshold). The rest are classified by what the prediction did
instead; predicted possessions that match nothing are classified the same way. Boundary offsets
of the found ones are summarised by transition type. Aggregates only, nothing frame-level.
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd

FPS = 25
SHORT_S = 3.0


def runs(lab, graded):
    """[(start, end, label)] over graded frames, one per constant-label run."""
    out, s = [], None
    for i in range(len(lab)):
        if not graded[i]:
            if s is not None:
                out.append((s, i - 1, lab[s])); s = None
            continue
        if s is None or lab[i] != lab[s]:
            if s is not None:
                out.append((s, i - 1, lab[s]))
            s = i
    if s is not None:
        out.append((s, len(lab) - 1, lab[s]))
    return out


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union


def classify(seg, other, cover):
    """Why a possession has no IoU>=0.5 partner: what the other track does over its span."""
    a, b, team = seg
    n = b - a + 1
    if n < SHORT_S * FPS:
        return "short (<{:.0f} s)".format(SHORT_S)
    same = [o for o in other if o[2] == team and iou(seg, o) > 0]
    if same and max(iou(seg, o) for o in same) >= 0.3:
        return "boundary (same team, IoU 0.3-0.5)"
    if len(same) >= 2 and sum(min(o[1], b) - max(o[0], a) + 1 for o in same) >= 0.5 * n:
        return "split into several"
    c = Counter(cover[a:b + 1])
    top = c.most_common(1)[0][0]
    if top == "D":
        return "covered by dead ball"
    if top != team:
        return "covered by the other team"
    return "boundary (same team, IoU < 0.3)"


def main():
    pred = pd.read_csv(sys.argv[1])["label"].values
    gt = pd.read_csv(sys.argv[2])["label"].values
    graded = gt != "X"
    g_runs = [r for r in runs(gt, graded) if r[2] in "HA"]
    p_runs = [r for r in runs(pred, graded) if r[2] in "HA"]

    found, offsets = 0, []
    miss = Counter()
    for seg in g_runs:
        best = max(((iou(seg, o), o) for o in p_runs if o[2] == seg[2]), default=(0, None))
        if best[0] >= 0.5:
            found += 1
            o = best[1]
            before = gt[seg[0] - 1] if seg[0] > 0 and graded[seg[0] - 1] else "?"
            after = gt[seg[1] + 1] if seg[1] + 1 < len(gt) and graded[seg[1] + 1] else "?"
            offsets.append(("start from " + ("dead" if before == "D" else "team"), (o[0] - seg[0]) / FPS))
            offsets.append(("end into " + ("dead" if after == "D" else "team"), (o[1] - seg[1]) / FPS))
        else:
            miss[classify(seg, p_runs, pred)] += 1
    spur = Counter()
    for seg in p_runs:
        best = max((iou(seg, o) for o in g_runs if o[2] == seg[2]), default=0)
        if best < 0.5:
            spur[classify(seg, g_runs, gt)] += 1

    print("true possessions {}  found at IoU>=0.5 {}  missed {}".format(len(g_runs), found, len(g_runs) - found))
    for k, v in miss.most_common():
        print("  missed: {:<40} {:>3}".format(k, v))
    print("predicted possessions {}  spurious {}".format(len(p_runs), sum(spur.values())))
    for k, v in spur.most_common():
        print("  spurious: {:<38} {:>3}".format(k, v))
    print("\nboundary offsets of the found possessions, seconds (+ = prediction late):")
    df = pd.DataFrame(offsets, columns=["kind", "s"])
    for k, d in df.groupby("kind"):
        print("  {:<18} n={:>3}  median {:+5.2f}  mean {:+5.2f}  |off|>2s {:>3}  |off|>4s {:>3}".format(
            k, len(d), d.s.median(), d.s.mean(), int((d.s.abs() > 2).sum()), int((d.s.abs() > 4).sum())))
    # what the lost IoU is made of: how much of the found possessions' union is off at the ends
    print("\ndurations: true possessions median {:.1f} s, missed ones median {:.1f} s".format(
        np.median([(b - a + 1) / FPS for a, b, _ in g_runs]),
        np.median([(b - a + 1) / FPS for seg in g_runs for a, b, _ in [seg]
                   if max((iou(seg, o) for o in p_runs if o[2] == seg[2]), default=0) < 0.5] or [0])))


if __name__ == "__main__":
    main()
