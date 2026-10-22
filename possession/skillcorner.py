"""SkillCorner open data (https://github.com/SkillCorner/opendata, MIT) as more training data:
broadcast tracking of A-League matches at 10 fps with a per-frame possession field.

    python -m possession.skillcorner --root ../external/skillcorner-tracking --out ../external/cache

Each match becomes a feature file in the layout of possession/features.py, resampled to 25 fps
by linear interpolation inside each period, plus a per-frame label: H or A from the possession
field (home team / away team); a frame without a possession is D when at least `min_detected`
players are seen in the broadcast image (the ball is out of play) and X otherwise (the broadcast
is away from the pitch, a replay or a close-up, and the positions are extrapolations). Only a
ball marked as detected counts as seen; players keep their extrapolated positions, as the
challenge tracking always has every player.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .features import derive

FPS_IN, FPS = 10, 25
LEN, WID = 105.0, 68.0


def upsample(a, factor=FPS / FPS_IN):
    """[n10, ...] -> [n25, ...] by linear interpolation along time; NaN where the nearest 10 fps
    sample was NaN, so gaps stay gaps."""
    n10 = a.shape[0]
    n25 = int(round((n10 - 1) * factor)) + 1
    t25 = np.arange(n25) / factor
    flat = a.reshape(n10, -1)
    out = np.full((n25, flat.shape[1]), np.nan)
    near = np.clip(np.rint(t25).astype(int), 0, n10 - 1)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        ok = ~np.isnan(col)
        if ok.sum() >= 2:
            out[:, c] = np.interp(t25, np.where(ok)[0], col[ok])
        out[np.isnan(col[near]), c] = np.nan
    return out.reshape((n25,) + a.shape[1:]), near


def convert_match(folder, match, out, min_detected=6):
    meta = json.load(open(os.path.join(folder, "{}_match.json".format(match)), encoding="utf-8"))
    home_id = meta["home_team"]["id"]
    tid = lambda p: p.get("id")                                  # the id the tracking file uses
    pids_home = sorted({tid(p) for p in meta["players"] if p["team_id"] == home_id})
    pids_away = sorted({tid(p) for p in meta["players"] if p["team_id"] != home_id})
    # tracking ids: collect every player_id seen, then split by the lineup when possible
    frames = []
    with open(os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(match)), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("period") not in (1, 2):
                continue
            frames.append(d)
    seen = sorted({p["player_id"] for d in frames for p in d["player_data"]})
    by_team = {pid: (pid in pids_home) for pid in seen if pid in pids_home or pid in pids_away}
    unknown = [pid for pid in seen if pid not in by_team]
    if unknown:
        # fall back on the lineup's team_player_id vs player_id mismatch: keep known ones only
        pass
    slots_home = [pid for pid in seen if by_team.get(pid) is True]
    slots_away = [pid for pid in seen if by_team.get(pid) is False]
    slot_of = {pid: k for k, pid in enumerate(slots_home + slots_away)}
    is_home = np.array([True] * len(slots_home) + [False] * len(slots_away), dtype=bool)
    assert len(slots_home) >= 11 and len(slots_away) >= 11, "team mapping failed: {} home, {} away, {} unknown".format(len(slots_home), len(slots_away), len(unknown))
    parts, balls, labs, periods, off = [], [], [], [], 0
    for per in (1, 2):
        fr = [d for d in frames if d["period"] == per]
        if not fr:
            continue
        f0 = fr[0]["frame"]
        n10 = fr[-1]["frame"] - f0 + 1
        players = np.full((n10, len(slot_of), 2), np.nan)
        ball = np.full((n10, 2), np.nan)
        lab10 = np.full(n10, "X", dtype="<U1")
        for d in fr:
            i = d["frame"] - f0
            for p in d["player_data"]:
                k = slot_of.get(p["player_id"])
                if k is not None and p.get("x") is not None:
                    players[i, k] = (p["x"], p["y"])
            b = d.get("ball_data") or {}
            if b.get("x") is not None and b.get("is_detected"):
                ball[i] = (b["x"], b["y"])
            g = (d.get("possession") or {}).get("group")
            if g == "home team":
                lab10[i] = "H"
            elif g == "away team":
                lab10[i] = "A"
            else:
                n_det = sum(1 for p in d["player_data"] if p.get("is_detected"))
                lab10[i] = "D" if n_det >= min_detected else "X"
        pl25, near = upsample(players)
        bl25, _ = upsample(ball)
        lab25 = lab10[near]
        # metres from the centre -> 0..1, and speed from +/- 2 frames at 25 fps
        pl = np.full((pl25.shape[0], pl25.shape[1], 3), np.nan)
        pl[:, :, 0] = (pl25[:, :, 0] + LEN / 2) / LEN
        pl[:, :, 1] = (pl25[:, :, 1] + WID / 2) / WID
        sp = np.full(pl25.shape[:2], np.nan)
        sp[2:-2] = np.sqrt((pl25[4:, :, 0] - pl25[:-4, :, 0]) ** 2 + (pl25[4:, :, 1] - pl25[:-4, :, 1]) ** 2) * FPS / 4
        pl[:, :, 2] = sp
        bl = np.stack([(bl25[:, 0] + LEN / 2) / LEN, (bl25[:, 1] + WID / 2) / WID], axis=1)
        parts.append(pl)
        balls.append(bl)
        labs.append(lab25)
        periods.append([per, off + 1, off + pl.shape[0]])
        off += pl.shape[0]
    players = np.concatenate(parts)
    ball = np.concatenate(balls)
    lab = np.concatenate(labs)
    arrays = derive(players, ball, is_home, LEN, WID, FPS)
    arrays["ball_src"] = np.where(np.isnan(ball[:, 0]), 0, 1).astype(np.int8)
    arrays["label"] = lab
    arrays["periods"] = np.array(periods)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "skillcorner_{}.npz".format(match)), **arrays)
    graded = lab != "X"
    runs = int(((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"])).sum())
    print("skillcorner_{}: {} frames, graded {:.0f} min, H {:.2f} A {:.2f} D {:.2f}, X {:.0%}, {} possessions, ball seen {:.0%}, players {}+{}".format(
        match, len(lab), graded.sum() / FPS / 60, *((lab == l).sum() / max(1, graded.sum()) for l in "HAD"), (~graded).mean(), runs,
        1 - np.isnan(ball[:, 0]).mean(), len(slots_home), len(slots_away)), flush=True)


def self_check():
    a = np.array([[0.0], [1.0], [np.nan], [3.0]])
    u, near = upsample(a, 2.0)
    assert u.shape == (7, 1) and abs(u[1, 0] - 0.5) < 1e-9 and np.isnan(u[4, 0]) and abs(u[6, 0] - 3.0) < 1e-9, u.ravel()


if __name__ == "__main__":
    self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with one sub-folder per match")
    ap.add_argument("--out", required=True)
    ap.add_argument("--matches", default="")
    args = ap.parse_args()
    import warnings
    warnings.simplefilter("ignore")
    matches = args.matches.split(",") if args.matches else sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d))
    for m in matches:
        if not os.path.exists(os.path.join(args.root, m, "{}_tracking_extrapolated.jsonl".format(m))):
            print(m, "has no tracking file yet", flush=True)
            continue
        convert_match(os.path.join(args.root, m), m, args.out)
