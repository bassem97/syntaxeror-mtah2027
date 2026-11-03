"""Two label-free choices tested on external data only: standardising the feature table per
match before the classifier, and training on Metrica's games next to SkillCorner's. Scored out
of sample: SkillCorner in two folds, each Metrica game predicted by the fold that did not train
on it (or by both folds' mean when Metrica is not trained on).

    python analysis/fit_features.py --external ../external/cache
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
    hand = {name: M.emissions(f, p) for name, f, _, _ in games}
    truth = {name: np.where(y >= 0, np.array(list("HAD"))[np.maximum(y, 0)], "X") for name, _, _, y in games}
    periods = {name: f["periods"] for name, f, _, _ in games}
    results = {}
    for zscore in (False, True):
        tables = {name: (L.standardise(t) if zscore else t) for name, _, t, _ in games}
        ys = {name: y for name, _, _, y in games}
        sc = sorted(n for n in tables if n.startswith("skillcorner"))
        folds = [sc[: len(sc) // 2], sc[len(sc) // 2:]]
        for with_metrica in (False, True):
            proba = {}
            for k in range(2):
                train_names = folds[k] + (["metrica_g{}".format(k + 1)] if with_metrica else [])
                test_names = folds[1 - k] + (["metrica_g{}".format(2 - k)] if with_metrica else ["metrica_g1", "metrica_g2"])
                X = pd.concat([tables[n] for n in train_names], ignore_index=True)
                Y = np.concatenate([ys[n] for n in train_names])
                mask = np.concatenate([L.train_mask_external(ys[n]) for n in train_names])
                boosters = [L.train(X, Y, mask, seed=s) for s in range(args.seeds)]
                for n in test_names:
                    proba.setdefault(n, []).append(np.mean([b.predict(tables[n].values) for b in boosters], axis=0))
            by_src = {}
            for name in tables:
                sc_ = hand[name] + L.emissions_from(np.mean(proba[name], axis=0), args.w)
                dec = L.decode_periods(sc_, periods[name], p)
                r = pm.evaluate(pm.label_codes(truth[name]), pm.label_codes(dec), bootstrap_n=0)
                by_src.setdefault(name.split("_")[0], []).append(r["spell_f1_mean"])
            per = {k: round(float(np.mean(v)), 3) for k, v in by_src.items()}
            key = "{}{}".format("zscore" if zscore else "raw", "+metrica" if with_metrica else "")
            results[key] = dict(per, mean=round(float(np.mean(list(per.values()))), 3))
            print(key, results[key], flush=True)
    out = os.path.join(HERE, "fit_features_result.json")
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
