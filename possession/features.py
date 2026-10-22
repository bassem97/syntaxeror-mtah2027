"""Per-frame features from the tactical ATD, cached as one .npz next to the data.

Run once (about a minute), then everything else loads the cache in milliseconds:

    python -m possession.features

Arrays, all length n_frames (video frame index, 0-based):
  ball_x, ball_y      normalised 0-1, NaN when the ball is not tracked
  d_home, d_away      distance in metres from the ball to the nearest home / away player (NaN if no ball)
  n_home3, n_away3    players of each team within 3 m of the ball
  mean_speed          mean speed (m/s) of all tracked players
  n_players           tracked players on the frame
  ball_speed          ball speed in m/s from consecutive visible frames (NaN otherwise)
  match_home, match_away   best agreement (0-1) between a nearby player's velocity and the ball's, per team
  home_cx, away_cx    team centroid along the pitch, metres
  home_spread, away_spread   mean distance of a team's players from their centroid, metres
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

from .games import DATA_DIR, GAME, GAMES, require, source_frames

CACHE = os.path.join(DATA_DIR, "cache", "tactical_features.npz")
NEAR_M = 3.0


def cache_path(variant):
    return CACHE if variant == "tracker" else CACHE.replace(".npz", "_{}.npz".format(variant))


def team_of_each_track(meta_path):
    root = ET.parse(meta_path).getroot()
    home_id = [t.get("id") for t in root.iter("Team")][0]
    return np.array([p.get("teamId") == home_id for p in root.iter("Player")])  # True = home


def parse_atd(raw_path, n_slots):
    """(frames, players[n, slots, 3], ball[n, 3]) straight from the raw file, vectorised."""
    with open(raw_path) as f:
        text = f.read()
    flat = np.fromstring(text.replace(":", ",").replace(";", ",").replace("\n", ","),
                         sep=",", dtype=np.float64)
    width = 1 + n_slots * 3 + 3
    assert flat.size % width == 0, "unexpected field count per line"
    table = flat.reshape(-1, width)
    frames = table[:, 0].astype(int) - 1  # frame_id counts from 1
    players = table[:, 1:1 + n_slots * 3].reshape(-1, n_slots, 3)
    ball = table[:, 1 + n_slots * 3:]
    return frames, players, ball


def build(variant="tracker"):
    """The ball as the ATD tracks it; other variants are for experiments and share the cache naming."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n = source_frames("tactical")
    length, width = GAMES[GAME]["pitch"]
    is_home = team_of_each_track(meta_path)

    frames, players, ball = parse_atd(raw_path, len(is_home))
    assert len(frames) == n and frames[0] == 0 and frames[-1] == n - 1
    ball = ball.copy()
    ball_src = np.where(np.isnan(ball[:, 0]), 0, 1).astype(np.int8)  # 0 none, 1 tracker

    px = players[:, :, 0] * length
    py = players[:, :, 1] * width
    ps = players[:, :, 2]
    bx = ball[:, 0] * length
    by = ball[:, 1] * width

    dist = np.sqrt((px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2)  # NaN where either missing
    d_home = np.nanmin(np.where(is_home[None, :], dist, np.nan), axis=1, initial=np.inf)
    d_away = np.nanmin(np.where(~is_home[None, :], dist, np.nan), axis=1, initial=np.inf)
    d_home[np.isinf(d_home)] = np.nan
    d_away[np.isinf(d_away)] = np.nan
    near = dist < NEAR_M
    n_home3 = (near & is_home[None, :]).sum(1)
    n_away3 = (near & ~is_home[None, :]).sum(1)

    ball_speed = np.full(n, np.nan)
    step = np.sqrt(np.diff(bx) ** 2 + np.diff(by) ** 2) * GAMES[GAME]["fps"]
    ball_speed[1:] = step

    # Who moves with the ball: velocity of every player and of the ball over +/- 2 frames, and per
    # team the best agreement between a nearby player's velocity and the ball's. A player carrying
    # the ball shares its velocity; a defender next to it usually does not.
    fps = GAMES[GAME]["fps"]
    def velocity(a):
        v = np.full_like(a, np.nan)
        v[2:-2] = (a[4:] - a[:-4]) * fps / 4
        return v
    pvx, pvy = velocity(px), velocity(py)
    bvx, bvy = velocity(bx), velocity(by)
    dv = np.sqrt((pvx - bvx[:, None]) ** 2 + (pvy - bvy[:, None]) ** 2)
    agree = np.exp(-dv / 2.0) * np.exp(-dist / 2.0)      # 1 = same velocity at the ball, fades with 2 m/s or 2 m
    agree = np.where(np.isnan(agree), 0.0, agree)
    match_home = np.max(np.where(is_home[None, :], agree, 0.0), axis=1)
    match_away = np.max(np.where(~is_home[None, :], agree, 0.0), axis=1)

    # Team shape, independent of the ball: centroid along the pitch and spread (mean distance of
    # a team's players from their centroid). The team in possession tends to spread and push up.
    hx = np.where(is_home[None, :], px, np.nan)
    hy = np.where(is_home[None, :], py, np.nan)
    ax = np.where(~is_home[None, :], px, np.nan)
    ay = np.where(~is_home[None, :], py, np.nan)
    home_cx, away_cx = np.nanmean(hx, axis=1), np.nanmean(ax, axis=1)
    home_cy, away_cy = np.nanmean(hy, axis=1), np.nanmean(ay, axis=1)
    home_spread = np.nanmean(np.sqrt((hx - home_cx[:, None]) ** 2 + (hy - home_cy[:, None]) ** 2), axis=1)
    away_spread = np.nanmean(np.sqrt((ax - away_cx[:, None]) ** 2 + (ay - away_cy[:, None]) ** 2), axis=1)

    out = cache_path(variant)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out,
                        ball_x=ball[:, 0], ball_y=ball[:, 1], ball_src=ball_src,
                        d_home=d_home, d_away=d_away,
                        n_home3=n_home3, n_away3=n_away3,
                        mean_speed=np.nanmean(ps, axis=1),
                        n_players=np.sum(~np.isnan(px), axis=1),
                        ball_speed=ball_speed, match_home=match_home, match_away=match_away,
                        home_cx=home_cx, away_cx=away_cx, home_spread=home_spread, away_spread=away_spread)
    return out


def load(variant="tracker"):
    path = cache_path(variant)
    if not os.path.isfile(path):
        build(variant)
    return dict(np.load(path))


if __name__ == "__main__":
    import argparse
    import warnings
    warnings.simplefilter("ignore")  # nanmean over empty frames at halftime
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    path = build()
    f = dict(np.load(path))
    vis = ~np.isnan(f["ball_x"])
    print("cached", path)
    print("ball visible {:.1f}%  mean players {:.1f}".format(100 * vis.mean(), np.nanmean(f["n_players"])))
    print("median d_home {:.2f} m  d_away {:.2f} m".format(np.nanmedian(f["d_home"]), np.nanmedian(f["d_away"])))
