"""Ball touches from the ball track: a sharp change of ball velocity, attributed to the nearest
player's team. The corrected tracking the ground truth comes from changes the owning team at a
touch, so touches are the right unit for team evidence, not per-frame proximity.

Returns one row per touch: frame, team (+1 home, -1 away), distance of the nearest player, and the
size of the velocity change in m/s.
"""
import numpy as np
import pandas as pd

from .games import GAME, GAMES

FPS = 25


def detect(f, dv_min=4.0, attach_m=3.5, refractory=6, smooth=2):
    length, width = GAMES[GAME]["pitch"]
    bx = f["ball_x"] * length
    by = f["ball_y"] * width
    n = len(bx)
    # velocity from a short centred window; NaN where the ball is missing anywhere in it
    k = 2 * smooth + 1
    sx = pd.Series(bx).rolling(k, center=True).mean().values
    sy = pd.Series(by).rolling(k, center=True).mean().values
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    vx[1:-1] = (sx[2:] - sx[:-2]) * FPS / 2
    vy[1:-1] = (sy[2:] - sy[:-2]) * FPS / 2
    # velocity change across +/- 3 frames
    lag = 3
    dv = np.full(n, np.nan)
    dv[lag:-lag] = np.hypot(vx[2 * lag:] - vx[:-2 * lag], vy[2 * lag:] - vy[:-2 * lag])

    dmin = np.fmin(f["d_home"], f["d_away"])
    cand = np.where((dv > dv_min) & (dmin < attach_m))[0]
    rows = []
    last = -10 ** 9
    for i in cand:
        # keep the local maximum of dv within the refractory window
        if i - last < refractory:
            if dv[i] > rows[-1][3]:
                rows[-1] = (i, 1 if f["d_home"][i] <= f["d_away"][i] else -1, dmin[i], dv[i])
            continue
        rows.append((i, 1 if f["d_home"][i] <= f["d_away"][i] else -1, dmin[i], dv[i]))
        last = i
    out = np.array(rows, dtype=float) if rows else np.zeros((0, 4))
    return out


def receptions(f, v_hi=4.0, v_lo=2.5, attach_m=2.5, refractory=10):
    """A fast ball that slows down onto a player: a controlled reception. Rarer than touches
    (about 4 per minute) but attributed to the right team 9 times in 10 on the public window."""
    length, width = GAMES[GAME]["pitch"]
    bx = f["ball_x"] * length
    by = f["ball_y"] * width
    n = len(bx)
    sx = pd.Series(bx).rolling(5, center=True).mean().values
    sy = pd.Series(by).rolling(5, center=True).mean().values
    v = np.full(n, np.nan)
    v[1:-1] = np.hypot(sx[2:] - sx[:-2], sy[2:] - sy[:-2]) * FPS / 2
    vmax_prev = pd.Series(v).rolling(10).max().shift(1).values
    vmean_next = pd.Series(v).rolling(10).mean().shift(-9).values
    dmin = np.fmin(f["d_home"], f["d_away"])
    cand = np.where((vmax_prev > v_hi) & (vmean_next < v_lo) & (dmin < attach_m))[0]
    rows = []
    last = -10 ** 9
    for i in cand:
        if i - last <= refractory:
            continue
        rows.append((i, 1 if f["d_home"][i] <= f["d_away"][i] else -1, dmin[i], 0.0))
        last = i
    return np.array(rows, dtype=float) if rows else np.zeros((0, 4))


def reappearances(f, attach_m=2.0, min_gap=5):
    """The ball reappears next to a player after being lost: treat as a touch by that team."""
    vis = ~np.isnan(f["ball_x"])
    dmin = np.fmin(f["d_home"], f["d_away"])
    rows = []
    gap = 0
    for i in range(len(vis)):
        if not vis[i]:
            gap += 1
            continue
        if gap >= min_gap and dmin[i] < attach_m:
            rows.append((i, 1 if f["d_home"][i] <= f["d_away"][i] else -1, dmin[i], 0.0))
        gap = 0
    return np.array(rows, dtype=float) if rows else np.zeros((0, 4))
