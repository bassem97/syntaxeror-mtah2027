"""Fit the decoder's parameters on the external games (possession/external.py), not on any label
of the challenge match.

    python analysis/fit_external.py --external ../external/cache [--model models/learned] [--grid '{...}']

Decodes every external game for each setting, scores it against the event-derived labels with
the organisers' metric code, and reports the mean per game. Row 0 is the current defaults.
The classifier's probabilities for the external games are the models' own predictions on their
training matches when --model is given, so the classifier weight is NOT fit here (it would be
in-sample); only the hand-evidence and decoder parameters are.
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
from possession import model as M  # noqa: E402
from possession import learned as L  # noqa: E402

GRID = {
    "team_cost": [30, 45, 60],
    "dead_cost": [25, 40, 55],
    "ctrl_m": [1.2, 1.5, 2.0],
    "carry_w": [0.15, 0.25],
    "tempo_floor": [0, 1.0],
    "switch_delay": [0, 20],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True)
    ap.add_argument("--games", default="metrica_g1,metrica_g2,metrica_g3")
    ap.add_argument("--grid", help="JSON overriding GRID entries")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.grid:
        GRID.update(json.loads(args.grid))
    import possession_metrics as pm
    base = {a.dest: a.default for a in M.argparser()._actions if a.dest != "help"}
    base["events"] = []
    games = []
    for name in args.games.split(","):
        f = dict(np.load(os.path.join(args.external, name + ".npz")))
        lab = f.pop("label")
        games.append((name, f, lab))
    keys = list(GRID)
    combos = [tuple(base[k] for k in keys)] + list(itertools.product(*(GRID[k] for k in keys)))
    rows = []
    for c in combos:
        p = dict(base)
        p.update(dict(zip(keys, c)))
        scores = []
        for name, f, lab in games:
            dec = L.decode_periods(M.emissions(f, p), f["periods"], p)
            r = pm.evaluate(pm.label_codes(lab), pm.label_codes(dec), bootstrap_n=0)
            scores.append((r["spell_f1_mean"], r["macro_f1"]))
        row = dict(zip(keys, c))
        row.update({"{}_spell".format(n): s[0] for (n, _, _), s in zip(games, scores)})
        row.update({"{}_macro".format(n): s[1] for (n, _, _), s in zip(games, scores)})
        row["mean_spell"] = float(np.mean([s[0] for s in scores]))
        row["mean_macro"] = float(np.mean([s[1] for s in scores]))
        rows.append(row)
        print(" ".join("{}={}".format(k, v) for k, v in zip(keys, c)),
              " spell per game:", [round(s[0], 3) for s in scores], " mean {:.3f} macro {:.3f}".format(row["mean_spell"], row["mean_macro"]), flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "labeling", "data", "fit_external{}.csv".format(args.tag))
    df.to_csv(out, index=False)
    print("\ndefaults: mean spell {:.3f} macro {:.3f}".format(df.loc[0, "mean_spell"], df.loc[0, "mean_macro"]))
    best = df.iloc[1:].sort_values("mean_spell", ascending=False).head(8)
    print("best settings on the external games:")
    print(best[keys + ["mean_spell", "mean_macro"]].round(3).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
