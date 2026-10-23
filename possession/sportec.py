"""The DFL / Sportec open dataset (Bassek et al. 2025, CC BY 4.0; seven Bundesliga matches of
optical tracking at 25 fps with the provider's per-frame ball status and possession team) as
more training data.

    python -m possession.sportec --out ../external/cache [--matches J03WPY,...]

kloppy fetches the files from the public mirror (pysport/idsse-data) and caches them. Each match
becomes a feature file in the layout of possession/features.py plus a per-frame label: D when
the ball status is dead, otherwise H or A from the possession team. The tracking always has the
ball, so it is blanked in runs drawn from the challenge tracking's own gap lengths
(labeling/data/ball_gap_stats.npz) to look like the tracking the model will meet.
"""
import argparse
import os

import numpy as np

from .features import derive
from .soccertrack import degrade_ball

FPS = 25
LEN, WID = 105.0, 68.0
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


def _speed(x, y):
    s = np.full_like(x, np.nan)
    s[2:-2] = np.sqrt((x[4:] - x[:-4]) ** 2 + (y[4:] - y[:-4]) ** 2) * FPS / 4
    return s


def convert_match(match, out, stats_path):
    from kloppy import sportec
    ds = sportec.load_open_tracking_data(match_id=match, coordinates="metrica")
    df = ds.to_df()
    home = [t for t in ds.metadata.teams if str(t.ground) == "home"][0]
    away = [t for t in ds.metadata.teams if str(t.ground) == "away"][0]
    cols = {c[:-2] for c in df.columns if c.endswith("_x") and not c.startswith("ball")}
    home_ids = [p.player_id for p in home.players if p.player_id in cols]
    away_ids = [p.player_id for p in away.players if p.player_id in cols]
    parts, balls, labs, periods, off = [], [], [], [], 0
    for per, g in df.groupby("period_id", sort=True):
        n = len(g)
        players = np.full((n, len(home_ids) + len(away_ids), 3), np.nan)
        for k, pid in enumerate(home_ids + away_ids):
            x, y = g[pid + "_x"].values.astype(float), g[pid + "_y"].values.astype(float)
            players[:, k, 0], players[:, k, 1] = x, y
            players[:, k, 2] = _speed(x * LEN, y * WID)
        ball = np.stack([g.ball_x.values.astype(float), g.ball_y.values.astype(float)], axis=1)
        ball = degrade_ball(ball, stats_path, seed=abs(hash(match)) % 1000 + int(per))
        lab = np.full(n, "X", dtype="<U1")
        state = g.ball_state.astype(str).values
        owner = g.ball_owning_team_id.astype(str).values
        lab[state == "dead"] = "D"
        alive = state == "alive"
        lab[alive & (owner == home.team_id)] = "H"
        lab[alive & (owner == away.team_id)] = "A"
        parts.append(players)
        balls.append(ball)
        labs.append(lab)
        periods.append([int(per), off + 1, off + n])
        off += n
    players = np.concatenate(parts)
    ball = np.concatenate(balls)
    lab = np.concatenate(labs)
    is_home = np.array([True] * len(home_ids) + [False] * len(away_ids), dtype=bool)
    arrays = derive(players, ball, is_home, LEN, WID, FPS)
    arrays["ball_src"] = np.where(np.isnan(ball[:, 0]), 0, 1).astype(np.int8)
    arrays["label"] = lab
    arrays["periods"] = np.array(periods)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "sportec_{}.npz".format(match)), **arrays)
    graded = lab != "X"
    runs = int(((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"])).sum())
    print("sportec_{}: {} frames, graded {:.0f} min, H {:.2f} A {:.2f} D {:.2f}, {} possessions, ball seen {:.0%}, players {}+{}".format(
        match, len(lab), graded.sum() / FPS / 60, *((lab == l).sum() / max(1, graded.sum()) for l in "HAD"), runs,
        1 - np.isnan(ball[:, 0]).mean(), len(home_ids), len(away_ids)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--matches", default=",".join(MATCHES))
    ap.add_argument("--gap-stats", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "labeling", "data", "ball_gap_stats.npz"))
    args = ap.parse_args()
    import warnings
    warnings.simplefilter("ignore")
    for m in args.matches.split(","):
        if os.path.exists(os.path.join(args.out, "sportec_{}.npz".format(m))):
            print(m, "already converted", flush=True)
            continue
        convert_match(m, args.out, args.gap_stats)
