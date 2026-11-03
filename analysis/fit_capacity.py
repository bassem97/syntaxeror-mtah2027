"""Choose the classifier's size on external data only: for each (trees, leaves) train the two
SkillCorner folds and score hand evidence + classifier (weight 0.35) out of sample on the other
fold and on the Metrica games, with the decoder at its current defaults.

    python analysis/fit_capacity.py --external ../external/cache
"""
import argparse
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

SIZES = [(300, 15), (600, 31), (1000, 63)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True)
    ap.add_argument("--w", type=float, default=0.35)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    import possession_metrics as pm
    p = {a.dest: a.default for a in M.argparser()._actions if a.dest != "help"}
    p["events"] = []
    games = L.external_tables(args.external, p, ["skillcorner_*", "metrica_g1", "metrica_g2"])
    sc = [g for g in games if g[0].startswith("skillcorner")]
    me = [g for g in games if g[0].startswith("metrica")]
    folds = [sc[: len(sc) // 2], sc[len(sc) // 2:]]
    hand = {name: M.emissions(f, p) for name, f, _, _ in games}
    truth = {name: np.where(y >= 0, np.array(list("HAD"))[np.maximum(y, 0)], "X") for name, _, _, y in games}
    results = {}
    for rounds, leaves in SIZES:
        proba = {}
        for k, train_games in enumerate(folds):
            X = pd.concat([t for _, _, t, _ in train_games], ignore_index=True)
            Y = np.concatenate([y for _, _, _, y in train_games])
            mask = np.concatenate([L.train_mask_external(y) for _, _, _, y in train_games])
            boosters = [L.train(X, Y, mask, seed=s, rounds=rounds, leaves=leaves) for s in range(args.seeds)]
            for name, _, t, _ in folds[1 - k] + me:
                proba.setdefault(name, []).append(np.mean([b.predict(t.values) for b in boosters], axis=0))
        by_src = {}
        for name, f, _, _ in games:
            sc_ = hand[name] + L.emissions_from(np.mean(proba[name], axis=0), args.w)
            dec = L.decode_periods(sc_, f["periods"], p)
            r = pm.evaluate(pm.label_codes(truth[name]), pm.label_codes(dec), bootstrap_n=0)
            by_src.setdefault(name.split("_")[0], []).append(r["spell_f1_mean"])
        per = {k: round(float(np.mean(v)), 3) for k, v in by_src.items()}
        results["{}x{}".format(rounds, leaves)] = dict(per, mean=round(float(np.mean(list(per.values()))), 3))
        print("{} trees x {} leaves: {}".format(rounds, leaves, results["{}x{}".format(rounds, leaves)]), flush=True)
    out = os.path.join(HERE, "fit_capacity_result.json")
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
