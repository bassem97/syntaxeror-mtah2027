"""Learned per-frame emissions: a gradient-boosted classifier over the tracking features and the
hand-written evidence terms, trained on the hand-labelled windows, decoded with the same Viterbi.

    python -m possession.learned --gt labeling/data/gt_internal.csv --windows a-b,c-d,...   # leave-one-window-out report
    python -m possession.learned --gt ... --windows ... --fit models/learned                 # train on all windows, save
    python -m possession.learned --model models/learned --name SyntaxEror --out out          # decode the whole match

The decoded evidence is the hand-written evidence of model.py plus the classifier's
log-probabilities at weight --w (0.5): the hand control cue is sharper at quick exchanges, the
classifier is better at dead balls and at who holds the ball while it is lost. Three seeds are
trained and their probabilities averaged; the saved models make decoding deterministic.

Frames within 10 frames of a hand-labelled transition are left out of training (the labels are
only good to about a third of a second there), and every window is also fed mirrored (home and
away swapped) so the classifier cannot learn a team bias.
"""
import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import model as M
from .features import load

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FPS = 25
LAB = {"H": 0, "A": 1, "D": 2}


def context(x, secs):
    out = {}
    for s in secs:
        w = int(s * FPS)
        out["m{}".format(s)] = pd.Series(x).rolling(2 * w + 1, center=True, min_periods=1).mean().values
    return out


def frame_table(f, p, secs=(0.5, 1, 2, 5)):
    """Per-frame feature table; columns that swap under a home/away mirror are named *_h / *_a."""
    parts = M.evidence_parts(f, p)
    cols = {}
    vis = (~np.isnan(f["ball_x"])).astype(float)
    cols["vis"] = vis
    cols["ball_x"] = np.nan_to_num(f["ball_x"], nan=0.5)
    cols["ball_y"] = np.nan_to_num(f["ball_y"], nan=0.5)
    cols["ball_speed"] = np.nan_to_num(f["ball_speed"], nan=-1)
    cols["d_h"] = np.nan_to_num(f["d_home"], nan=50)
    cols["d_a"] = np.nan_to_num(f["d_away"], nan=50)
    cols["n3_h"] = np.nan_to_num(f["n_home3"], nan=0)
    cols["n3_a"] = np.nan_to_num(f["n_away3"], nan=0)
    cols["mean_speed"] = np.nan_to_num(f["mean_speed"], nan=0)
    cols["n_players"] = np.nan_to_num(f["n_players"], nan=0)
    cols["cx_h"] = np.nan_to_num(f["home_cx"], nan=50)
    cols["cx_a"] = np.nan_to_num(f["away_cx"], nan=50)
    cols["spread_h"] = np.nan_to_num(f["home_spread"], nan=0)
    cols["spread_a"] = np.nan_to_num(f["away_spread"], nan=0)
    for k, v in parts.items():
        cols["ev_" + k] = v          # 'team' is signed: + home, - away
    # time since the ball was last seen, and since it was last seen out
    n = len(vis)
    last = np.full(n, -1.0)
    t = -1e9
    for i in range(n):
        if vis[i]:
            t = i
        last[i] = min((i - t) / FPS, 60)
    cols["since_vis"] = last
    base = dict(cols)
    for k, v in base.items():
        if k in ("vis", "since_vis"):
            continue
        for ck, cv in context(v, secs).items():
            cols[k + "_" + ck] = cv
    df = pd.DataFrame(cols)
    return df


def mirror(df):
    """Swap home and away: *_h <-> *_a columns, signed team evidence negated, pitch x flipped."""
    m = df.copy()
    for c in df.columns:
        if "_h" in c and c.replace("_h", "_a", 1) in df.columns:
            m[c] = df[c.replace("_h", "_a", 1)]
            m[c.replace("_h", "_a", 1)] = df[c]
    for c in df.columns:
        if c.startswith("ev_team"):
            m[c] = -df[c]
        if c.startswith("ball_x"):
            m[c] = 1 - df[c]
    return m


def train(df, y, mask, seed=0):
    X = pd.concat([df[mask], mirror(df[mask])], ignore_index=True)
    ym = y[mask].copy()
    ym_m = ym.copy()
    ym_m[ym == 0] = 1
    ym_m[ym == 1] = 0
    Y = np.concatenate([ym, ym_m])
    params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=15, min_data_in_leaf=200,
                  bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, lambda_l2=5.0, seed=seed,
                  num_threads=8, verbose=-1)
    return lgb.train(params, lgb.Dataset(X.values, Y), num_boost_round=300)


def emissions_from(proba, w):
    """log-probabilities scaled to the hand evidence's units, same alive/dead decoupling."""
    lp = np.log(np.clip(proba, 1e-4, 1))
    team = (lp[:, 0] - lp[:, 1]) / 2
    alive = -np.abs(team)
    e_d = (lp[:, 2] - np.maximum(lp[:, 0], lp[:, 1]))
    return w * np.stack([alive + team, alive - team, e_d], axis=1)


def decode(score, p):
    n = len(score)
    labels = np.full(n, "X", dtype="<U1")
    for half, first, last in M.half_video_blocks(M.GAME):
        lo = M.KICKOFF_2ND if half == 2 else first
        labels[first:lo] = "D"
        hl = M.LABELS[M.viterbi(score[lo:last + 1], p["team_cost"], p["dead_cost"], p["dead_exit"])]
        labels[lo:last + 1] = M.merge_short_dead(hl, p["min_dead"])
    return M.delay_switches(labels, int(p["switch_delay"]))


def train_mask(gt, windows, margin=10):
    """Graded frames of the windows, minus a margin around every hand-labelled transition."""
    n = len(gt)
    mask = np.zeros(n, bool)
    for a, b in windows:
        mask[a:b + 1] = True
    mask &= gt != "X"
    ch = np.where(gt[1:] != gt[:-1])[0] + 1
    for i in ch:
        mask[max(0, i - margin):i + margin] = False
    return mask


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt")
    ap.add_argument("--windows", help="comma-separated first-last frames")
    ap.add_argument("--fit", help="train on all windows and save the models under this prefix")
    ap.add_argument("--model", help="prefix of the saved models to decode the whole match with")
    ap.add_argument("--w", type=float, default=0.5, help="scale of the log-probabilities as evidence")
    ap.add_argument("--hand-w", type=float, default=1.0, help="scale of the hand-written evidence (0 = classifier only)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--name", default="SyntaxEror_learned")
    ap.add_argument("--out", default=os.path.join(ROOT, "out"))
    ap.add_argument("--ball", default="tracker")
    args, rest = ap.parse_known_args(argv)
    p = {a.dest: a.default for a in M.argparser()._actions if a.dest != "help"}
    p.update(vars(M.argparser().parse_args(rest)))
    f = load(args.ball)
    df = frame_table(f, p)

    if args.model:
        import glob
        files = sorted(glob.glob(args.model + "_*.txt"))
        # read as text and normalise newlines: a CRLF checkout breaks LightGBM's own file parser
        boosters = [lgb.Booster(model_str=open(m, newline="").read().replace("\r\n", "\n")) for m in files]
        proba = np.mean([b.predict(df.values) for b in boosters], axis=0)
        labels = decode(args.hand_w * M.emissions(f, p) + emissions_from(proba, args.w), p)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, args.name + ".csv")
        M.write(labels, path)
        print("wrote", path)
        return

    sys.path.insert(0, os.path.join(ROOT, "..", "mtah", "evaluation"))
    import possession_metrics as pm
    gt = pd.read_csv(args.gt)["label"].values
    windows = [tuple(int(x) for x in w.split("-")) for w in args.windows.split(",")]
    y = np.array([LAB.get(l, -1) for l in gt])

    if args.fit:
        mask = train_mask(gt, windows)
        os.makedirs(os.path.dirname(args.fit) or ".", exist_ok=True)
        for k in range(args.seeds):
            train(df, y, mask, seed=k).save_model("{}_{}.txt".format(args.fit, k))
        print("saved", args.seeds, "models under", args.fit, "trained on", int(mask.sum()), "frames x2")
        return

    # leave-one-window-out: emissions for every window come from a model that never saw it
    proba = np.zeros((len(df), 3))
    for k, (a, b) in enumerate(windows):
        others = [w for w in windows if w != (a, b)]
        mask = train_mask(gt, others)
        clf = train(df, y, mask)
        proba[a:b + 1] = clf.predict(df.iloc[a:b + 1].values)
        acc = (proba[a:b + 1].argmax(1) == y[a:b + 1])[y[a:b + 1] >= 0].mean()
        print("window {} {}-{}: frame accuracy {:.3f}".format(k, a, b, acc), flush=True)
    hand = M.emissions(f, p)
    for w in [0.5, 1.0, 2.0, 4.0]:
        for tc, dc in [(45, 40), (30, 25), (60, 50), (20, 15)]:
            q = dict(p, team_cost=tc, dead_cost=dc)
            labels = decode(emissions_from(proba, w), q)
            per = []
            for a, b in windows:
                g = np.full(len(gt), "X", dtype="<U1")
                g[a:b + 1] = gt[a:b + 1]
                per.append(pm.evaluate(pm.label_codes(g), pm.label_codes(labels), bootstrap_n=0)["spell_f1_mean"])
            r = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels), bootstrap_n=0)
            print("w={} costs {}/{}: per window {}  mean {:.3f} pooled spell {:.3f} macro {:.3f}".format(
                w, tc, dc, [round(x, 3) for x in per], np.mean(per), r["spell_f1_mean"], r["macro_f1"]), flush=True)
    labels = decode(hand, p)
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels), bootstrap_n=0)
    print("hand evidence, same decoder: pooled spell {:.3f} macro {:.3f}".format(r["spell_f1_mean"], r["macro_f1"]))


if __name__ == "__main__":
    main()
