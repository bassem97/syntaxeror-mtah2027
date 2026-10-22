"""SoccerTrack v2 (https://huggingface.co/datasets/atomscott/soccertrack-v2, CC BY 4.0) as more
training data: ten university matches filmed by fixed panoramic cameras, with per-frame pitch
coordinates of every player (gsr/), team-labelled ball actions (bas/) and a ball track
reconstructed between those actions (ball/).

    python -m possession.soccertrack --root ../external/soccertrack-v2 --out ../external/cache

Each half's game-state JSON is 2.7 GB, so it is parsed as a stream, reduced to arrays, and may be
deleted afterwards. A match becomes one feature file in the layout of possession/features.py
plus a per-frame label (H the team that defends the left goal in the first half, A the other,
D dead, X not graded) derived from the ball actions:

  * Pass, Drive, Header, High Pass, Cross, Throw In, Shot, Free Kick, Ball Player Block and
    Player Successful Tackle put the ball with the acting team from that frame;
  * Out and Goal make the ball dead from their frame until the next action of any kind;
  * a Free Kick or Throw In that comes more than 3 s after the previous action is preceded by a
    dead ball from 1 s after that previous action (the foul or the ball leaving play was not
    tagged at its own moment);
  * before the first action of a half and after its last, frames are X.

The reconstructed ball is interpolated between actions, which makes it cleaner than any real
tracking. To look like the challenge's tracking it is blanked in runs drawn from the challenge
tracking's own gap-length distribution (labeling/data/ball_gap_stats.npz) until 43% is missing.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from .features import derive

FPS = 25
LEN, WID = 105.0, 68.0
POSSESSION = {"Pass", "Drive", "Header", "High Pass", "Cross", "Throw In", "Shot", "Free Kick",
              "Ball Player Block", "Player Successful Tackle"}
DEAD = {"Out", "Goal"}
RESTARTS = {"Free Kick", "Throw In"}


def stream_half(path):
    """-> dict track_id -> {"team": "left"/"right", "player_id", "role", "frames": [], "x": [], "y": []}."""
    import ijson
    tracks = {}
    with open(path, "rb") as fh:
        for a in ijson.items(fh, "annotations.item"):
            if a.get("supercategory", "object") != "object":
                continue
            at = a.get("attributes") or {}
            if at.get("role") not in ("player", "goalkeeper"):
                continue
            bp = a.get("bbox_pitch") or {}
            x, y = bp.get("x_bottom_middle"), bp.get("y_bottom_middle")
            if x is None or y is None:
                continue
            frame = int(str(a["image_id"])[-6:])          # "3000001" -> 1 (1-based)
            t = tracks.setdefault(a["track_id"], {"team": at.get("team"), "player_id": at.get("player_id"),
                                                  "role": at.get("role"), "frames": [], "x": [], "y": []})
            t["frames"].append(frame)
            t["x"].append(float(x))
            t["y"].append(float(y))
    return tracks


def half_arrays(tracks, n):
    """Per-team slot arrays for one half: players[n, slots, 3] normalised like the ATD, by team side."""
    sides = {"left": [], "right": []}
    for tid, t in tracks.items():
        if t["team"] in sides:
            sides[t["team"]].append(tid)
    out = {}
    for side, tids in sides.items():
        arr = np.full((n, len(tids), 3), np.nan)
        for k, tid in enumerate(tids):
            t = tracks[tid]
            fr = np.array(t["frames"]) - 1
            ok = (fr >= 0) & (fr < n)
            arr[fr[ok], k, 0] = (np.array(t["x"])[ok] + LEN / 2) / LEN     # metres from centre -> 0..1
            arr[fr[ok], k, 1] = (np.array(t["y"])[ok] + WID / 2) / WID
        xm, ym = arr[:, :, 0] * LEN, arr[:, :, 1] * WID
        sp = np.full_like(xm, np.nan)
        sp[2:-2] = np.sqrt((xm[4:] - xm[:-4]) ** 2 + (ym[4:] - ym[:-4]) ** 2) * FPS / 4
        arr[:, :, 2] = sp
        out[side] = (arr, {tracks[tid]["player_id"] for tid in tids})
    return out


def labels_from_events(events, n, side_is_home):
    """events: list of (frame, label, side) for one half, 1-based frames; side_is_home: {"left": bool}."""
    lab = np.full(n, "X", dtype="<U1")
    ev = sorted(events)
    state, cursor, prev_frame = None, 0, None
    for frame, label, side in ev:
        t = frame - 1
        team = "H" if side_is_home.get(side) else "A"
        if state is not None:
            lab[cursor:max(cursor, t)] = state
        if label in RESTARTS and prev_frame is not None and t - prev_frame > 3 * FPS and state != "D":
            lab[prev_frame + FPS:t] = "D"
        if label in DEAD:
            state, cursor = "D", t
        elif label in POSSESSION:
            state, cursor = team, t
        prev_frame = t
    if state is not None and ev:
        lab[cursor:ev[-1][0]] = state   # hold until the last action; after it nothing is graded
    return lab


def degrade_ball(ball, stats_path, missing=0.43, seed=0):
    """Blank the reconstructed ball in runs drawn from the challenge tracking's gap lengths."""
    rng = np.random.default_rng(seed)
    st = np.load(stats_path)
    gaps, seen = st["gaps"], st["seen"]
    b = ball.copy()
    n = len(b)
    i = int(rng.choice(seen))
    while i < n:
        g = int(rng.choice(gaps))
        b[i:i + g] = np.nan
        i += g + int(rng.choice(seen))
    return b


def load_ball(path, n):
    z = np.load(path)
    keys = list(z.keys())
    if "x" in z and "y" in z:
        x, y = z["x"], z["y"]
    else:
        a = z[keys[0]]
        x, y = a[:, 0], a[:, 1]
    ball = np.full((n, 2), np.nan)
    m = min(n, len(x))
    ball[:m, 0] = (x[:m] + LEN / 2) / LEN
    ball[:m, 1] = (y[:m] + WID / 2) / WID
    return ball, keys


def convert_match(root, match, out, stats_path, keep_json=False):
    bas = json.load(open(os.path.join(root, "bas", match, "{}_12_class_events.json".format(match)), encoding="utf-8"))
    halves = []
    home_ids = None
    for h, tag in ((1, "1st"), (2, "2nd")):
        gpath = os.path.join(root, "gsr", match, "{}_{}.json".format(match, tag))
        tracks = stream_half(gpath)
        n = max(f for t in tracks.values() for f in t["frames"])
        sides = half_arrays(tracks, n)
        if home_ids is None:                      # home = the side defending the left goal in half 1
            home_ids = sides["left"][1]
            side_is_home = {"left": True, "right": False}
        else:                                     # sides can swap: follow the player ids
            overlap = {s: len(ids & home_ids) for s, (_, ids) in sides.items()}
            left_home = overlap["left"] >= overlap["right"]
            side_is_home = {"left": left_home, "right": not left_home}
        home_side = "left" if side_is_home["left"] else "right"
        away_side = "right" if home_side == "left" else "left"
        players = np.concatenate([sides[home_side][0], sides[away_side][0]], axis=1)
        is_home = np.array([True] * sides[home_side][0].shape[1] + [False] * sides[away_side][0].shape[1])
        events = []
        for e in bas["annotations"]:
            half = int(str(e["gameTime"]).split(" - ")[0])
            if half != h:
                continue
            frame = int(e["frame"]) if "frame" in e else int(round(int(e["position"]) / 40))
            events.append((frame, e["label"], e["team"]))
        lab = labels_from_events(events, n, side_is_home)
        ball, keys = load_ball(os.path.join(root, "ball", "{}_{}_ball.npz".format(match, tag)), n)
        ball = degrade_ball(ball, stats_path, seed=int(match) + h)
        halves.append((players, is_home, ball, lab))
        if not keep_json:
            os.remove(gpath)
        print("  half {}: {} frames, {} tracks, {} events, ball keys {}".format(h, n, len(tracks), len(events), keys), flush=True)
    # stitch halves into one match with a frame gap of nothing: periods array keeps them apart
    slots = max(p.shape[1] for p, _, _, _ in halves)
    parts, labs, balls, periods, off = [], [], [], [], 0
    is_home_all = None
    for k, (players, is_home, ball, lab) in enumerate(halves):
        pad = np.full((players.shape[0], slots - players.shape[1], 3), np.nan)
        parts.append(np.concatenate([players, pad], axis=1))
        if is_home_all is None:
            is_home_all = np.concatenate([is_home, np.array([False] * (slots - len(is_home)))])
        labs.append(lab)
        balls.append(ball)
        periods.append([k + 1, off + 1, off + players.shape[0]])
        off += players.shape[0]
    players = np.concatenate(parts)
    ball = np.concatenate(balls)
    lab = np.concatenate(labs)
    arrays = derive(players, ball, is_home_all, LEN, WID, FPS)
    arrays["ball_src"] = np.where(np.isnan(ball[:, 0]), 0, 1).astype(np.int8)
    arrays["label"] = lab
    arrays["periods"] = np.array(periods)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "soccertrack_{}.npz".format(match)), **arrays)
    graded = lab != "X"
    runs = int(((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"])).sum())
    print("soccertrack_{}: {} frames, graded {:.0f} min, H {:.2f} A {:.2f} D {:.2f}, {} possessions, ball visible {:.0%}".format(
        match, len(lab), graded.sum() / FPS / 60, *((lab == l).sum() / max(1, graded.sum()) for l in "HAD"), runs,
        1 - np.isnan(ball[:, 0]).mean()), flush=True)


def self_check():
    ev = [(100, "Pass", "left"), (200, "Out", "right"), (300, "Throw In", "left"), (500, "Pass", "right"),
          (700, "Free Kick", "left"), (800, "Goal", "left"), (900, "Pass", "right")]
    lab = labels_from_events(ev, 1000, {"left": True, "right": False})
    assert lab[50] == "X" and lab[100] == "H" and lab[199] == "D" and lab[298] == "D" and lab[299] == "H"
    assert lab[500] == "A" and lab[523] == "A" and lab[524] == "D" and lab[698] == "D" and lab[700] == "H"
    assert lab[799] == "D" and lab[899] == "A" and lab[950] == "X"
    st = {"gaps": np.array([10, 20]), "seen": np.array([30, 40])}
    np.savez("_gapcheck.npz", **st)
    b = degrade_ball(np.zeros((1000, 2)), "_gapcheck.npz")
    os.remove("_gapcheck.npz")
    assert 0.1 < np.isnan(b[:, 0]).mean() < 0.5


if __name__ == "__main__":
    self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder holding gsr/, bas/ and ball/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--matches", default="", help="comma-separated match ids (default: all with gsr files)")
    ap.add_argument("--keep-json", action="store_true")
    ap.add_argument("--gap-stats", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "labeling", "data", "ball_gap_stats.npz"))
    args = ap.parse_args()
    matches = args.matches.split(",") if args.matches else sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.root, "gsr", "*")))
    for m in matches:
        print("match", m, flush=True)
        convert_match(args.root, m, args.out, args.gap_stats, args.keep_json)
