"""Fit the possession model's parameters on the labelled windows, leave-one-window-out.

    python analysis/fit.py --gt labeling/data/gt_internal.csv --windows 11185-18684,18685-23921,50000-57499

For each candidate setting the model is decoded once (a few seconds) and scored per window. A
setting is chosen on the windows it was NOT scored on, so the report says how each parameter
generalises across windows rather than how well it fits them. With three windows this is a
sanity check; with ten it becomes a real fit.

Scores the windows with the organisers' metric code (../mtah/evaluation/possession_metrics.py).
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "mtah", "evaluation"))
from possession import model  # noqa: E402

GRID = {
    "team_cost": [30, 45, 60],
    "dead_cost": [20, 30, 40],
    "ctrl_m": [1.2, 1.5, 2.0],
    "carry_w": [0.15, 0.25, 0.4],
    "touch_w": [0, 3, 6],
}


def spell_f1_window(pred, gt, lo, hi):
    """spellF1 of one window with the organisers' code; X outside the window."""
    import possession_metrics as pm
    g = np.full(len(gt), "X", dtype="<U1")
    g[lo:hi + 1] = gt[lo:hi + 1]
    r = pm.evaluate(pm.label_codes(g), pm.label_codes(pred), bootstrap_n=0)
    return float(r["spell_f1_mean"]), float(r["macro_f1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--windows", required=True, help="comma-separated first-last frames")
    ap.add_argument("--ball", default="tracker")
    ap.add_argument("--max", type=int, default=0, help="limit the number of settings (0 = all)")
    ap.add_argument("--grid", help='JSON overriding GRID entries, e.g. {"ctrl_m": [0.9, 1.2]}')
    ap.add_argument("--tag", default="", help="suffix for the output csv")
    args = ap.parse_args()
    if args.grid:
        GRID.update(json.loads(args.grid))
    gt = pd.read_csv(args.gt)["label"].values
    windows = [tuple(int(x) for x in w.split("-")) for w in args.windows.split(",")]
    base = {a.dest: a.default for a in model.argparser()._actions if a.dest != "help"}
    base["ball"] = args.ball
    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    if args.max:
        combos = combos[:args.max]
    combos = [tuple(base[k] for k in keys)] + combos  # row 0 is the model's actual defaults
    rows = []
    for c in combos:
        p = dict(base)
        p.update(dict(zip(keys, c)))
        labels = model.build(p)
        scores = [spell_f1_window(labels, gt, lo, hi) for lo, hi in windows]
        rows.append(dict(zip(keys, c), **{"w{}_spell".format(i): s[0] for i, s in enumerate(scores)},
                         **{"w{}_macro".format(i): s[1] for i, s in enumerate(scores)}))
        print(" ".join("{}={}".format(k, v) for k, v in zip(keys, c)), " spell per window:", [round(s[0], 3) for s in scores], flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "labeling", "data", "fit_{}{}.csv".format(args.ball, args.tag))
    df.to_csv(out, index=False)
    spell_cols = [c for c in df.columns if c.endswith("_spell")]
    df["mean_spell"] = df[spell_cols].mean(axis=1)
    print("\nleave-one-window-out: for each held-out window, the setting chosen on the others, and its score there")
    held = []
    for i, col in enumerate(spell_cols):
        others = [c for c in spell_cols if c != col]
        best = df.loc[df[others].mean(axis=1).idxmax()]
        held.append(best[col])
        print("  held-out window {}: chosen {} -> {:.3f} (default setting scores {:.3f})".format(
            i, {k: best[k] for k in keys}, best[col], df.loc[0, col]))
    print("mean held-out spellF1 of the fitted settings: {:.3f}; of the defaults: {:.3f}".format(
        np.mean(held), df.loc[0, spell_cols].mean()))
    print("wrote", out)


if __name__ == "__main__":
    main()
