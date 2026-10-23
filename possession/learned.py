"""Learned per-frame emissions: a gradient-boosted classifier over the tracking features and the
hand-written evidence terms, trained on OTHER matches (SkillCorner's open data through
possession/skillcorner.py; Metrica's sample games through possession/external.py are the
alternative), decoded with the same Viterbi. Nothing from the challenge match is used in training.

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


def decode_periods(score, periods, p):
    """Decode a match given as (period, first_frame, last_frame) rows, 1-based frames."""
    labels = np.full(len(score), "X", dtype="<U1")
    score = np.nan_to_num(score, nan=0.0)   # frames without tracking carry no evidence
    for _, a, b in periods:
        a, b = int(a) - 1, int(b) - 1
        hl = M.LABELS[M.viterbi(score[a:b + 1], p["team_cost"], p["dead_cost"], p["dead_exit"])]
        labels[a:b + 1] = M.merge_short_dead(hl, p["min_dead"])
    return M.delay_switches(labels, int(p["switch_delay"]))


def external_tables(folder, p, patterns=("*",)):
    """[(name, features dict, table, label codes)] for every external game in the folder whose
    name matches one of the patterns."""
    import fnmatch
    import glob
    q = dict(p, events=[])
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        if not any(fnmatch.fnmatch(os.path.basename(path)[:-4], pat) for pat in patterns):
            continue
        f = dict(np.load(path))
        lab = f.pop("label")
        out.append((os.path.basename(path)[:-4], f, frame_table(f, q), np.array([LAB.get(l, -1) for l in lab])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", help="folder of external feature files with labels (possession/external.py)")
    ap.add_argument("--fit", help="train on the external games and save the models under this prefix")
    ap.add_argument("--model", help="prefix of the saved models to decode the whole match with")
    ap.add_argument("--w", type=float, default=0.35, help="scale of the log-probabilities as evidence")
    ap.add_argument("--hand-w", type=float, default=1.0, help="scale of the hand-written evidence (0 = classifier only)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--games", default="skillcorner_*", help="external games to train on: names or prefixes with *")
    ap.add_argument("--stride", type=int, default=2, help="keep every n-th training frame (neighbours are near-duplicates)")
    ap.add_argument("--margin", type=int, default=25, help="frames around an event-derived transition left out of training")
    ap.add_argument("--gt", help="internal ground truth to report on (validation only)")
    ap.add_argument("--windows", help="comma-separated first-last frames of the validation windows")
    ap.add_argument("--name", default="SyntaxEror_learned")
    ap.add_argument("--out", default=os.path.join(ROOT, "out"))
    args, rest = ap.parse_known_args(argv)
    p = {a.dest: a.default for a in M.argparser()._actions if a.dest != "help"}
    p.update(vars(M.argparser().parse_args(rest)))
    f = load("tracker")
    df = frame_table(f, p)
    hand = M.emissions(f, p)

    if args.model:
        import glob
        files = sorted(glob.glob(args.model + "_*.txt"))
        boosters = [lgb.Booster(model_str=open(m, newline="").read().replace("\r\n", "\n")) for m in files]
        proba = np.mean([b.predict(df.values) for b in boosters], axis=0)
        labels = decode(args.hand_w * hand + emissions_from(proba, args.w), p)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, args.name + ".csv")
        M.write(labels, path)
        print("wrote", path)
        return

    sys.path.insert(0, os.path.join(ROOT, "..", "mtah", "evaluation"))
    import possession_metrics as pm
    games = external_tables(args.external, p, args.games.split(","))
    # 1. are the derived labels consistent with the challenge's convention? score the hand model on them
    for name, fe, _, y in games:
        lab = decode_periods(M.emissions(fe, dict(p, events=[])), fe["periods"], p)
        g = np.where(y >= 0, np.array(list("HAD"))[np.maximum(y, 0)], "X")
        r = pm.evaluate(pm.label_codes(g), pm.label_codes(lab), bootstrap_n=0)
        print("hand model on {}: spellF1 {:.3f} macroF1 {:.3f} ({} true possessions)".format(name, r["spell_f1_mean"], r["macro_f1"], r["n_gt_spells"]), flush=True)
    # 2. train on all external games
    X = pd.concat([t for _, _, t, _ in games], ignore_index=True)
    Y = np.concatenate([y for _, _, _, y in games])
    mask = np.zeros(len(Y), bool)
    off = 0
    for _, _, t, y in games:
        m = y >= 0
        ch = np.where(y[1:] != y[:-1])[0] + 1
        for i in ch:
            m[max(0, i - args.margin):i + args.margin] = False
        if args.stride > 1:
            m[np.arange(len(m)) % args.stride != 0] = False
        mask[off:off + len(y)] = m
        off += len(y)
    print("training on {} external frames x2 (mirrored)".format(int(mask.sum())), flush=True)
    boosters = [train(X, Y, mask, seed=k) for k in range(args.seeds)]
    if args.fit:
        os.makedirs(os.path.dirname(args.fit) or ".", exist_ok=True)
        for k, b in enumerate(boosters):
            b.save_model("{}_{}.txt".format(args.fit, k))
        print("saved", len(boosters), "models under", args.fit)
    # 3. validation on the challenge match: public window and the hand-labelled windows
    proba = np.mean([b.predict(df.values) for b in boosters], axis=0)
    pub = pd.read_csv(os.path.join(ROOT, "..", "mtah", "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].values
    rows = [("hand", hand)] + [("hand + learned x{}".format(w), hand + emissions_from(proba, w)) for w in (0.25, 0.35, 0.45)] + [("learned alone x0.5", emissions_from(proba, 0.5))]
    gt = pd.read_csv(args.gt)["label"].values if args.gt else None
    windows = [tuple(int(x) for x in w.split("-")) for w in args.windows.split(",")] if args.windows else []
    for name, sc in rows:
        lab = decode(sc, p)
        q = pm.evaluate(pm.label_codes(pub), pm.label_codes(lab), bootstrap_n=0)
        line = "{:<22} public spell {:.3f} macro {:.3f}".format(name, q["spell_f1_mean"], q["macro_f1"])
        if gt is not None:
            per = []
            for a, b in windows:
                g = np.full(len(gt), "X", dtype="<U1")
                g[a:b + 1] = gt[a:b + 1]
                per.append(pm.evaluate(pm.label_codes(g), pm.label_codes(lab), bootstrap_n=0)["spell_f1_mean"])
            r = pm.evaluate(pm.label_codes(gt), pm.label_codes(lab), bootstrap_n=500)
            line += " | windows pooled spell {:.3f} [{:.2f}-{:.2f}] macro {:.3f} [{}]".format(r["spell_f1_mean"], r["spell_bootstrap_low"], r["spell_bootstrap_high"], r["macro_f1"], " ".join("%.2f" % x for x in per))
        print(line, flush=True)


if __name__ == "__main__":
    main()
