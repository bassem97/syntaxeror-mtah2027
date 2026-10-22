"""Other matches as training data: Metrica Sports' three public sample games.

    python -m possession.external --data ../external/metrica-sample-data/data --out ../external/cache

Source: https://github.com/metrica-sports/sample-data (tracking at 25 fps and events; games 1 and 2
in Metrica's CSV format, game 3 in the FIFA EPTS format; "please acknowledge the source"). Every
game becomes a feature file in the layout of possession/features.py plus a per-frame label
(H home, A away, D dead, X not graded) derived from the events:

  * a possession event (pass, carry, recovery, shot, set piece, ball lost) puts the ball with its
    team from its start frame;
  * the ball going out, a goal, a foul, a card, an offside or the end of a half make it dead from
    that event's end frame;
  * the next set piece ends the dead ball and gives possession to the team taking it;
  * before the first kick-off of a period, and after the half ends, frames are X.

The labels are only as good as the events (about one event every two seconds), so the frames
around a transition are uncertain by a second or so; the training mask drops them.
"""
import argparse
import os

import numpy as np
import pandas as pd

from .features import derive

FPS = 25
LEN, WID = 105.0, 68.0


def _speed(x, y, fps=FPS):
    """Speed in m/s from positions in metres over +/- 2 frames, like the ATD's own speed field."""
    s = np.full_like(x, np.nan)
    s[2:-2] = np.sqrt((x[4:] - x[:-4]) ** 2 + (y[4:] - y[:-4]) ** 2) * fps / 4
    return s


def tracking_arrays(df, tracks_home, tracks_away):
    """kloppy dataframe -> (players[n, slots, 3], ball[n, 2], is_home[slots]) indexed by frame."""
    n = int(df.frame_id.max())
    players = np.full((n, len(tracks_home) + len(tracks_away), 3), np.nan)
    ball = np.full((n, 2), np.nan)
    idx = df.frame_id.values.astype(int) - 1
    ball[idx, 0] = df.ball_x.values
    ball[idx, 1] = df.ball_y.values
    for k, t in enumerate(tracks_home + tracks_away):
        x, y = df[t + "_x"].values, df[t + "_y"].values
        players[idx, k, 0], players[idx, k, 1] = x, y
        players[idx, k, 2] = _speed(x * LEN, y * WID)
    is_home = np.array([True] * len(tracks_home) + [False] * len(tracks_away))
    return players, ball, is_home


DEAD_TYPES_CSV = {"BALL OUT", "FAULT RECEIVED", "CARD"}
POSS_TYPES_CSV = {"PASS", "RECOVERY", "SHOT", "BALL LOST", "SET PIECE"}


def labels_csv(events, n, period_frames):
    """Per-frame H/A/D/X from a Metrica CSV events table (games 1 and 2)."""
    lab = np.full(n, "X", dtype="<U1")
    ev = events.sort_values("Start Frame")
    state = None      # "H", "A", or "D"
    cursor = 0
    ended = set()
    for _, e in ev.iterrows():
        t, typ, sub = int(e["Start Frame"]) - 1, e["Type"], str(e["Subtype"]) if pd.notna(e["Subtype"]) else ""
        end = int(e["End Frame"]) - 1 if pd.notna(e["End Frame"]) and e["End Frame"] >= e["Start Frame"] else t
        team = "H" if e["Team"] == "Home" else "A"
        if state is not None and e["Period"] not in ended:
            lab[cursor:max(cursor, t)] = state
        if typ == "SET PIECE" or "GOAL KICK" in sub:   # goal kicks are tagged as passes, not set pieces
            state = team
            cursor = t
        elif typ in POSS_TYPES_CSV and state != "D":
            state = team
            cursor = t
        if "END HALF" in sub:
            lab[cursor:end + 1] = state if state else "X"
            state, cursor = None, end + 1
            ended.add(e["Period"])
            continue
        if typ in DEAD_TYPES_CSV or "GOAL" in sub and typ == "SHOT" or "-OUT" in sub or "OFFSIDE" in sub \
                or "HAND BALL" in sub or (typ == "CHALLENGE" and "FAULT" in sub):
            if state is not None:
                lab[cursor:end + 1] = state
            state, cursor = "D", end + 1
    if state is not None:
        lab[cursor:n] = state
    # nothing is graded before each period's first kick-off
    for p, (first, _) in period_frames.items():
        ko = ev[(ev["Type"] == "SET PIECE") & (ev["Subtype"].astype(str).str.contains("KICK OFF")) & (ev["Period"] == p)]
        if len(ko):
            lab[first - 1:int(ko["Start Frame"].iloc[0]) - 1] = "X"
    return lab


def labels_epts(events, n, period_start_frame):
    """Per-frame H/A/D/X from kloppy's event dataframe of game 3."""
    lab = np.full(n, "X", dtype="<U1")
    ev = events.copy()
    ev["frame"] = [period_start_frame[p] + int(round(ts.total_seconds() * FPS)) - 1 for p, ts in zip(ev.period_id, ev.timestamp)]
    ev["end_frame"] = [period_start_frame[p] + int(round(ts.total_seconds() * FPS)) - 1 if pd.notna(ts) else f
                       for p, ts, f in zip(ev.period_id, ev.end_timestamp, ev.frame)]
    ev = ev.sort_values("frame")
    state, cursor = None, 0
    for _, e in ev.iterrows():
        t, end = int(e.frame), max(int(e.frame), int(e.end_frame))
        team = "H" if e.team_id == "FIFATMA" else "A"
        typ = e.event_type
        if state is not None:
            lab[cursor:max(cursor, t)] = state
        if pd.notna(e.get("set_piece_type")):
            state, cursor = team, t
        elif typ in ("PASS", "CARRY", "RECOVERY", "SHOT") and state != "D":
            state, cursor = team, t
        if typ == "BALL_OUT" or typ == "FOUL_COMMITTED" or (typ == "SHOT" and str(e.result) == "GOAL") or typ.endswith("CARD"):
            if state is not None:
                lab[cursor:end + 1] = state
            state, cursor = "D", end + 1
    if state is not None:
        lab[cursor:n] = state
    return lab


def build_csv(data, game, out):
    from kloppy import metrica
    d = os.path.join(data, "Sample_Game_{}".format(game))
    ds = metrica.load_tracking_csv(home_data=os.path.join(d, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(game)),
                                   away_data=os.path.join(d, "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(game)),
                                   coordinates="metrica")
    df = ds.to_df()
    home = sorted({c[:-2] for c in df.columns if c.startswith("home_") and c.endswith("_x")})
    away = sorted({c[:-2] for c in df.columns if c.startswith("away_") and c.endswith("_x")})
    players, ball, is_home = tracking_arrays(df, home, away)
    events = pd.read_csv(os.path.join(d, "Sample_Game_{}_RawEventsData.csv".format(game)))
    pf = {p: (int(g.frame_id.min()), int(g.frame_id.max())) for p, g in df.groupby("period_id")}
    lab = labels_csv(events, len(ball), pf)
    save(out, "metrica_g{}".format(game), players, ball, is_home, lab, pf)


def build_epts(data, out):
    from kloppy import metrica
    d = os.path.join(data, "Sample_Game_3")
    meta = os.path.join(d, "Sample_Game_3_metadata.xml")
    ds = metrica.load_tracking_epts(meta_data=meta, raw_data=os.path.join(d, "Sample_Game_3_tracking.txt"), coordinates="metrica")
    df = ds.to_df()
    home = [p.player_id for t in ds.metadata.teams if str(t.ground) == "home" for p in t.players]
    away = [p.player_id for t in ds.metadata.teams if str(t.ground) == "away" for p in t.players]
    home = [p for p in home if p + "_x" in df.columns]
    away = [p for p in away if p + "_x" in df.columns]
    players, ball, is_home = tracking_arrays(df, home, away)
    ev = metrica.load_event(event_data=os.path.join(d, "Sample_Game_3_events.json"), meta_data=meta, coordinates="metrica").to_df()
    pf = {p: (int(g.frame_id.min()), int(g.frame_id.max())) for p, g in df.groupby("period_id")}
    lab = labels_epts(ev, len(ball), {p: a for p, (a, _) in pf.items()})
    save(out, "metrica_g3", players, ball, is_home, lab, pf)


def save(out, name, players, ball, is_home, lab, period_frames):
    os.makedirs(out, exist_ok=True)
    arrays = derive(players, ball, is_home, LEN, WID, FPS)
    arrays["ball_src"] = np.where(np.isnan(ball[:, 0]), 0, 1).astype(np.int8)
    arrays["label"] = lab
    arrays["periods"] = np.array([[p, a, b] for p, (a, b) in sorted(period_frames.items())])
    np.savez_compressed(os.path.join(out, name + ".npz"), **arrays)
    graded = lab != "X"
    runs = int(((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"])).sum())
    print("{}: {} frames, graded {:.0f} min, H {:.2f} A {:.2f} D {:.2f}, {} possessions, ball visible {:.0%}".format(
        name, len(lab), graded.sum() / FPS / 60, *((lab == l).sum() / max(1, graded.sum()) for l in "HAD"), runs,
        1 - np.isnan(ball[:, 0]).mean()))


def load(path):
    f = dict(np.load(path))
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder holding Sample_Game_1..3")
    ap.add_argument("--out", required=True, help="where the feature files go")
    args = ap.parse_args()
    import warnings
    warnings.simplefilter("ignore")
    ap_games = [1, 2, 3]
    for g in ap_games:
        build_csv(args.data, g, args.out) if g < 3 else build_epts(args.data, args.out)
