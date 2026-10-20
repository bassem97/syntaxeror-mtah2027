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
  home_cx, away_cx    team centroid along the pitch, metres
  home_spread, away_spread   mean distance of a team's players from their centroid, metres
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

from .games import DATA_DIR, GAME, GAMES, require, source_frames

CACHE = os.path.join(DATA_DIR, "cache", "tactical_features.npz")
NEAR_M = 3.0


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


def build():
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n = source_frames("tactical")
    length, width = GAMES[GAME]["pitch"]
    is_home = team_of_each_track(meta_path)

    frames, players, ball = parse_atd(raw_path, len(is_home))
    assert len(frames) == n and frames[0] == 0 and frames[-1] == n - 1

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

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE,
                        ball_x=ball[:, 0], ball_y=ball[:, 1],
                        d_home=d_home, d_away=d_away,
                        n_home3=n_home3, n_away3=n_away3,
                        mean_speed=np.nanmean(ps, axis=1),
                        n_players=np.sum(~np.isnan(px), axis=1),
                        ball_speed=ball_speed,
                        home_cx=home_cx, away_cx=away_cx, home_spread=home_spread, away_spread=away_spread)
    return CACHE


def load():
    if not os.path.isfile(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        build()
    return dict(np.load(CACHE))


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")  # nanmean over empty frames at halftime
    path = build()
    f = dict(np.load(path))
    vis = ~np.isnan(f["ball_x"])
    print("cached", path)
    print("ball visible {:.1f}%  mean players {:.1f}".format(100 * vis.mean(), np.nanmean(f["n_players"])))
    print("median d_home {:.2f} m  d_away {:.2f} m".format(np.nanmedian(f["d_home"]), np.nanmedian(f["d_away"])))
