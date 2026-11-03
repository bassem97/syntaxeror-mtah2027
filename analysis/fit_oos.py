"""Choose every tunable value on annotated matches from other sources, never on the challenge
match: the decoder's costs and cue weights, and the weight of the classifier's evidence.

    python analysis/fit_oos.py --external ../external/cache --proba ../external/oos

Stage 1 gives every external game an out-of-sample classifier: SkillCorner's twenty matches are
split in two folds (train on ten, predict the other ten) and the Metrica games get the mean of
both folds. Stage 2 is a coordinate descent from the values set by hand before any labelling:
each parameter in turn takes the value of its grid that maximises the mean spellF1 over the
external games (SkillCorner and Metrica weighted equally as sources), two passes. The result is
written to analysis/fit_oos_result.json and copied into possession/model.py's defaults by hand.
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

# Round 1 (23 Oct): the nine values that had been chosen on our own labels, from the values set
# by hand before any labelling. Round 2 (3 Nov, organisers' clarification: every tunable value
# must be tuned on external data): every remaining cue weight too, from round 1's result.
START1 = dict(team_cost=45, dead_cost=30, ctrl_m=1.5, carry_w=0.25, match_w=0.5, touch_w=3.0,
              tempo_floor=0.0, switch_delay=0, w=0.0)
GRID1 = {
    "w": [0.0, 0.25, 0.35, 0.45],
    "team_cost": [30, 45, 60],
    "dead_cost": [25, 30, 40, 55],
    "ctrl_m": [1.2, 1.5, 2.0],
    "carry_w": [0.0, 0.15, 0.25],
    "match_w": [0.0, 0.5],
    "touch_w": [0.0, 3.0],
    "tempo_floor": [0.0, 1.0],
    "switch_delay": [0, 10, 20],
}
START2 = dict(team_cost=30, dead_cost=25, ctrl_m=1.2, carry_w=0.25, match_w=0.5, touch_w=3.0,
              tempo_floor=1.0, switch_delay=10, w=0.35,
              shape_w=0.15, restart_w=0.5, moving_w=1.5, kick_w=1.0, out_hold_w=2.0, out_hold_s=3.0,
              min_dead=25, speed_ratio=0.8, far_m=13.0, invis_w=0.3)
GRID2 = dict(GRID1, **{
    "shape_w": [0.0, 0.15, 0.3],
    "restart_w": [0.0, 0.5, 1.0],
    "moving_w": [0.75, 1.5, 3.0],
    "kick_w": [0.0, 1.0, 2.0],
    "out_hold_w": [1.0, 2.0, 4.0],
    "out_hold_s": [1.5, 3.0, 6.0],
    "min_dead": [12, 25, 50],
    "speed_ratio": [0.6, 0.8, 1.0],
    "far_m": [10.0, 13.0, 16.0],
    "invis_w": [0.0, 0.3, 0.6],
})
START, GRID = START2, GRID2


def oos_probabilities(games, proba_dir, models=None):
    """Out-of-sample probabilities per game, computed once and cached as .npy. With no
    SkillCorner game in the list the committed models (trained on all of SkillCorner) are
    out of sample for every other source and are used as they are."""
    missing = [g for g in games if not os.path.exists(os.path.join(proba_dir, g[0] + ".npy"))]
    if not missing:
        return
    sc = [g for g in games if g[0].startswith("skillcorner")]
    rest = [g for g in games if not g[0].startswith("skillcorner")]
    if not sc:
        import glob
        import lightgbm as lgb
        boosters = [lgb.Booster(model_str=open(m, newline="").read().replace("\r\n", "\n")) for m in sorted(glob.glob(models + "_*.txt"))]
        os.makedirs(proba_dir, exist_ok=True)
        for name, _, t, _ in missing:
            np.save(os.path.join(proba_dir, name + ".npy"), np.mean([b.predict(t.values) for b in boosters], axis=0))
        return
    folds = [sc[: len(sc) // 2], sc[len(sc) // 2:]]
    out = {}
    for k, train_games in enumerate(folds):
        X = pd.concat([t for _, _, t, _ in train_games], ignore_index=True)
        Y = np.concatenate([y for _, _, _, y in train_games])
        mask = np.concatenate([L.train_mask_external(y) for _, _, _, y in train_games])
        boosters = [L.train(X, Y, mask, seed=s) for s in range(3)]
        print("fold {}: trained on {} frames".format(k, int(mask.sum())), flush=True)
        for name, _, t, _ in folds[1 - k] + rest:
            out.setdefault(name, []).append(np.mean([b.predict(t.values) for b in boosters], axis=0))
    os.makedirs(proba_dir, exist_ok=True)
    for name, prs in out.items():
        np.save(os.path.join(proba_dir, name + ".npy"), np.mean(prs, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True)
    ap.add_argument("--proba", required=True, help="folder for the out-of-sample probabilities")
    ap.add_argument("--games", default="skillcorner_*,metrica_g1,metrica_g2")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--models", default=os.path.join(ROOT, "models", "learned"), help="committed classifier prefix, used when no SkillCorner game is fitted on")
    ap.add_argument("--tag", default="", help="suffix of the result file")
    ap.add_argument("--round", type=int, default=2, help="1: the nine values of 23 Oct from the hand-set start; 2: every tunable value from round 1's result")
    ap.add_argument("--labels", help="frame,label CSV of our own annotation of the (single) fitted match, replacing its event-derived label")
    args = ap.parse_args()
    import possession_metrics as pm
    global START, GRID
    START, GRID = (START1, GRID1) if args.round == 1 else (START2, GRID2)
    base = {a.dest: a.default for a in M.argparser()._actions if a.dest != "help"}
    base["events"] = []
    games = L.external_tables(args.external, base, args.games.split(","))
    oos_probabilities(games, args.proba, args.models)
    data = []
    for name, f, _, y in games:
        lab = np.where(y >= 0, np.array(list("HAD"))[np.maximum(y, 0)], "X")
        if args.labels:
            assert len(games) == 1, "--labels is for one annotated match"
            lab = pd.read_csv(args.labels)["label"].values.astype("<U1")
            assert len(lab) == len(y), "annotation has {} frames, features {}".format(len(lab), len(y))
            print("hand labels: {} graded frames ({:.1f} min)".format(int((lab != "X").sum()), (lab != "X").sum() / 25 / 60), flush=True)
        data.append((name, f, lab, np.load(os.path.join(args.proba, name + ".npy"))))
    del games
    cache = {}

    def score(p):
        key = tuple(sorted((k, p[k]) for k in GRID))
        if key in cache:
            return cache[key]
        q = dict(base)
        q.update({k: v for k, v in p.items() if k != "w"})
        by_src = {}
        for name, f, lab, proba in data:
            sc = M.emissions(f, q) + L.emissions_from(proba, p["w"])
            dec = L.decode_periods(sc, f["periods"], q)
            r = pm.evaluate(pm.label_codes(lab), pm.label_codes(dec), bootstrap_n=0)
            by_src.setdefault(name.split("_")[0], []).append(r["spell_f1_mean"])
        val = float(np.mean([np.mean(v) for v in by_src.values()]))
        cache[key] = (val, {k: round(float(np.mean(v)), 3) for k, v in by_src.items()})
        return cache[key]

    cur = dict(START)
    best, per = score(cur)
    print("start {} -> {:.3f} {}".format(cur, best, per), flush=True)
    history = [dict(cur, score=best)]
    for n in range(args.passes):
        for k in GRID:
            trial = {v: score(dict(cur, **{k: v})) for v in GRID[k]}
            v_best = max(trial, key=lambda v: trial[v][0])
            if trial[v_best][0] > best + 1e-9:
                cur[k], (best, per) = v_best, trial[v_best]
            print("pass {} {}: {} -> {} {:.3f} {}".format(n, k, {v: round(t[0], 3) for v, t in trial.items()}, cur[k], best, per), flush=True)
        history.append(dict(cur, score=best))
    out = os.path.join(HERE, "fit_oos_result{}.json".format(args.tag))
    json.dump({"start": START, "grid": GRID, "best": cur, "score": best, "per_source": per, "history": history}, open(out, "w"), indent=1)
    print("best", cur, "mean spellF1 {:.3f}".format(best), "wrote", out)


if __name__ == "__main__":
    main()
