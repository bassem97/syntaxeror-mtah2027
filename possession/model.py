"""Segment-level possession model: per-frame evidence + 3-state Viterbi decoding.

    python make_submission.py                       # writes out/SyntaxEror.csv with the defaults
    python -m possession.model --name try2 --team-cost 40

States H, A, D. Each frame gets a score per state from the cached features (possession/features.py)
and the Smart Tagging; Viterbi then finds the labelling that maximises total score minus a cost for
every state change. The switching cost is what turns flickering per-frame evidence into whole
possessions: a change of possession has to be backed by roughly cost / typical-evidence frames.

Alive vs dead is decided by the dead-ball evidence alone (max(e_H, e_A) is always 0):
  + ball tracked outside the pitch, held for a few seconds afterwards (the stoppage instant)
  + mean player speed low compared with a rolling 3-minute median (the tempo drifts a lot)
  + one whole team far from the ball (set-piece geometry; capped, build-up looks the same)
  + ball lying still with nobody within 1.5 m (placed for a restart)
  + from the goal moment (ball over the line, or its last fast movement inside the GOALS tag)
    to the next kick-off tag
  - ball travelling fast, and a kick from standstill (restart or pass: either way in play)
Team evidence (H vs A only): nearest-player gap on control frames (somebody within ctrl_m of the
ball), a carry of the last controlling team through flight and loose balls until the other side
controls it, an impulse at every detected touch (possession/touches.py), team spread on frames
without a ball, and a prior for the team a tagged set piece names.

Costs are not fitted to the public window on purpose -- see EVALUATION.md -- they sit where the
full-match possession and stoppage counts (analysis/counts.py) look like a football match.
"""
import argparse
import os

import numpy as np
import pandas as pd

from . import smart_tagging as st
from . import touches
from .features import load
from .games import GAME, half_video_blocks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FPS = 25
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])
KICKOFF_2ND = 113742  # FAQ: second half graded from 110808, kicks off here; all D in between


def roll(a, w, how="mean"):
    s = pd.Series(a).rolling(2 * w + 1, center=True, min_periods=1)
    return (s.mean() if how == "mean" else s.median()).values


def emissions(f, p):
    parts = evidence_parts(f, p)
    e_d = sum(v for k, v in parts.items() if k != "team")
    team = parts["team"]
    # max(e_H, e_A) is always 0, so alive-vs-dead is decided by e_d alone and the team evidence
    # only decides H vs A. Without this a keeper standing on a dead ball outvotes the stoppage.
    alive = -np.abs(team)
    return np.stack([alive + team, alive - team, e_d], axis=1)


def evidence_parts(f, p):
    """Named per-frame evidence terms. All but 'team' are dead-ball evidence and sum to e_D."""
    n = len(f["ball_x"])
    vis = ~np.isnan(f["ball_x"])
    m = p["out_margin"]
    out = vis & ((f["ball_x"] < -m) | (f["ball_x"] > 1 + m) | (f["ball_y"] < -m) | (f["ball_y"] > 1 + m))
    # The ball seen out marks the stoppage instant; it is then dead for at least a few seconds
    # (nobody throws in or kicks a goal kick that fast), whatever the tracking shows meanwhile.
    s_out = p["out_w"] * out.astype(float)
    hold = int(p["out_hold_s"] * FPS)
    if hold > 0:
        for i in np.where(out)[0]:
            s_out[i + 1:i + 1 + hold] = np.maximum(s_out[i + 1:i + 1 + hold], p["out_hold_w"])

    spd = roll(f["mean_speed"], FPS)
    tempo = roll(np.nan_to_num(f["mean_speed"], nan=np.nanmedian(f["mean_speed"])), 90 * FPS, "median")
    ratio = spd / np.maximum(tempo, 0.3)
    s_speed = np.clip((p["speed_ratio"] - ratio) * 4, -2, 2)

    # One team far from the ball: set-piece geometry, but also a low block during build-up, so
    # this is capped low and the moving-ball cue below is what tells the two apart.
    far = roll(np.fmax(f["d_home"], f["d_away"]), FPS)
    s_far = np.clip((np.nan_to_num(far, nan=p["far_m"]) - p["far_m"]) / 4, -1, 1)

    dmin = np.fmin(f["d_home"], f["d_away"])
    bspd = roll(np.nan_to_num(f["ball_speed"], nan=5.0), 12)
    s_placed = np.where(vis & (bspd < 1.5) & (dmin > 1.5), 1.5, 0.0)
    # A ball travelling fast is in play; a dead ball is still, carried, or lost.
    s_moving = np.where(vis, -np.clip((bspd - 4.0) / 4.0, 0, 1) * p["moving_w"], 0.0)
    # A kick from standstill is a restart (or a pass, which is also play): strong alive evidence
    # for the next 2 s, so the decoder leaves D at the kick instead of 3-4 s later.
    before = pd.Series(bspd).shift(FPS).values
    kick = vis & (bspd > 8.0) & (before < 2.5)
    s_kick = np.zeros(n)
    for i in np.where(kick)[0]:
        s_kick[i:i + 2 * FPS] = -p["kick_w"]

    s_goal = np.zeros(n)
    ev = st.load()
    goals = sorted(e["start_frame"] for e in ev if e["code"] == "GOALS")
    kickoffs = sorted(e["start_frame"] for e in ev
                      if e["code"] == "SET PIECES" and "kick off" in (e["tags"].get("Type") or "").lower())
    # A GOALS tag starts well before the goal (it includes the build-up), so the dead window
    # starts at the goal moment: the ball seen beyond the line, or failing that the last fast
    # ball movement before it disappears into the net, searched within a minute of the tag.
    for g in goals:
        nxt = [k for k in kickoffs if k > g]
        if not nxt:
            continue
        win = slice(g, min(g + 60 * FPS, nxt[0]))
        o = np.where(out[win])[0]
        if len(o):
            moment = g + o[0]
        else:
            fast = np.where(vis[win] & (bspd[win] > 4.0))[0]
            moment = g + fast[-1] if len(fast) else g
        s_goal[moment:nxt[0]] = 3.0

    # Team evidence only where somebody controls the ball. In flight or loose, carry the last
    # controlling team weakly: possession stays with the last team that had it until the other
    # side gains control, which is the challenge's own definition of a change of possession.
    gap = np.nan_to_num(f["d_away"] - f["d_home"], nan=0.0)
    if p["ctrl_soft"] > 0:
        # Distance-weighted: full weight at the ball, fading with exp(-dmin / ctrl_soft), so a
        # tracked touch at 2 m still counts for something and there is no cliff at a radius.
        weight = np.where(vis, np.exp(-np.nan_to_num(dmin, nan=99.0) / p["ctrl_soft"]), 0.0)
        control = weight > 0.3
        team = np.clip(gap / 2, -2, 2) * weight
    else:
        control = vis & (dmin < p["ctrl_m"])
        team = np.where(control, np.clip(gap / 2, -2, 2), 0.0)
    last = pd.Series(np.where(control, np.sign(gap), np.nan)).ffill(limit=p["carry_s"] * FPS).fillna(0).values
    team = team + np.where(control, 0.0, p["carry_w"] * last)

    # Touches: a sharp change of ball velocity next to a player is the moment the corrected
    # tracking hands possession over. Each touch is an impulse for the toucher's team, spread over
    # a few frames so the decoder can place the boundary at it.
    if p["touch_w"] > 0:
        T = np.vstack([touches.detect(f, dv_min=p["touch_dv"]), touches.reappearances(f)])
        s_touch = np.zeros(n)
        span = 5
        for fr, side, _, _ in T:
            fr = int(fr)
            s_touch[fr:fr + span] += side * p["touch_w"] / span
        # Receptions: a fast ball slowing onto a player. Few, but attributed with high precision.
        if p["recv_w"] > 0:
            for fr, side, _, _ in touches.receptions(f, v_hi=p["recv_vhi"], v_lo=p["recv_vlo"], attach_m=p["recv_m"]):
                fr = int(fr)
                s_touch[fr:fr + span] += side * p["recv_w"] / span
        team = team + s_touch

    # Team shape, which needs no ball: the side in possession spreads out, the defending side
    # compacts. Weak per frame, but it is the only team evidence on the 41% of frames without a
    # ball, and it never flickers the way proximity does.
    if p["shape_w"] > 0:
        spread = roll(f["home_spread"] - f["away_spread"], FPS)
        s_shape = np.clip(np.nan_to_num(spread, nan=0.0) / 6.0, -1, 1) * p["shape_w"]
        team = team + (np.where(vis, 0.0, s_shape) if p["shape_invis_only"] else s_shape)

    # Restart prior: a tagged set piece names the team that restarts play. The tag starts up to
    # ~12 s before the actual restart, so the prior covers a window rather than an instant.
    home_name = "barcelona"
    team = team + restart_prior(ev, goals, kickoffs, home_name, n, p["restart_w"])
    return {"speed": s_speed, "far": s_far, "placed": s_placed, "moving": s_moving, "kick": s_kick, "goal": s_goal,
            "out": s_out, "invis": np.where(vis, 0.0, p["invis_w"]), "team": team}


def restart_prior(ev, goals, kickoffs, home_name, n, w):
    prior = np.zeros(n)
    goal_team = {}
    for e in ev:
        if e["code"] == "GOALS":
            goal_team[e["start_frame"]] = (e["tags"].get("Team") or "").lower()
    for e in ev:
        if e["code"] != "SET PIECES":
            continue
        t, typ, team = e["start_frame"], (e["tags"].get("Type") or "").lower(), (e["tags"].get("Team") or "").lower()
        if "after goal" in typ:
            prev = [g for g in goals if g < t]
            if not prev:
                continue
            scorer = goal_team.get(prev[-1], "")
            if not scorer or scorer == "n/a":
                continue
            team = "away" if scorer == home_name else "home"  # the team that conceded kicks off
        elif team in ("", "n/a"):
            continue
        sign = 1.0 if team in (home_name, "home") else -1.0
        prior[max(0, t - 2 * FPS):t + 15 * FPS] += sign * w
    return prior


def merge_short_dead(labels, min_frames):
    """A dead ball shorter than min_frames is a decoding shortcut, not a stoppage. Give it to a
    neighbour: the right one, since a short D between two teams is really the hand-over point."""
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] != "D":
            i += 1
            continue
        j = i
        while j < n and out[j] == "D":
            j += 1
        if j - i < min_frames:
            fill = out[j] if j < n and out[j] in "HA" else (out[i - 1] if i > 0 else "D")
            out[i:j] = fill
        i = j
    return out


def viterbi(score, team_cost, dead_cost, dead_exit=None):
    """Max-score path through n x 3 scores with a cost per state change.

    cost[i, j] is paid for moving from state i to state j. Entering D (a stoppage) and leaving it
    (a restart) can cost differently: a restart is a sharp event the evidence shows well, a
    spurious stoppage is the expensive mistake."""
    n = score.shape[0]
    dead_exit = dead_cost if dead_exit is None else dead_exit
    cost = np.array([[0, team_cost, dead_cost],
                     [team_cost, 0, dead_cost],
                     [dead_exit, dead_exit, 0]], dtype=float)
    back = np.zeros((n, 3), dtype=np.int8)
    cur = score[0].copy()
    for t in range(1, n):
        cand = cur[:, None] - cost  # cand[i, j] = best ending in i, then move to j
        back[t] = np.argmax(cand, axis=0)
        cur = cand[back[t], np.arange(3)] + score[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(cur))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def build(p):
    f = load(p.get("ball", "tracker"))
    n = len(f["ball_x"])
    score = emissions(f, p)
    labels = np.full(n, "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        lo = KICKOFF_2ND if half == 2 else first
        labels[first:lo] = "D"
        half_labels = LABELS[viterbi(score[lo:last + 1], p["team_cost"], p["dead_cost"], p["dead_exit"])]
        labels[lo:last + 1] = merge_short_dead(half_labels, p["min_dead"])
    return labels


def self_check():
    """Viterbi must hold a state through weak counter-evidence and switch on sustained evidence."""
    s = np.zeros((40, 3))
    s[:20, 0] = 1.0        # home for 20 frames
    s[20:22, 1] = 1.0      # a 2-frame blip of away evidence
    s[22:30, 0] = 1.0
    s[30:, 1] = 1.0        # then sustained away
    path = viterbi(s, team_cost=5, dead_cost=100)
    assert list(path[:30]) == [H] * 30, path
    assert list(path[30:]) == [A] * 10, path
    assert "".join(merge_short_dead(np.array(list("HHDDAA")), 3)) == "HHAAAA"
    assert "".join(merge_short_dead(np.array(list("HHDDDAA")), 3)) == "HHDDDAA"


def write(labels, path):
    with open(path, "w") as h:
        h.write("frame,label\n")
        h.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="SyntaxEror")
    ap.add_argument("--ball", default="tracker", choices=["tracker", "detector"], help="which ball track the features use")
    ap.add_argument("--out", default=os.path.join(ROOT, "out"), help="output folder")
    ap.add_argument("--team-cost", type=float, default=45)
    ap.add_argument("--dead-cost", type=float, default=30, help="keep >= team-cost / 2, or a team switch routes through a 1-frame D")
    ap.add_argument("--dead-exit", type=float, default=None, help="cost of leaving D; defaults to dead-cost")
    ap.add_argument("--kick-w", type=float, default=1.0)
    ap.add_argument("--min-dead", type=int, default=25, help="frames; shorter D runs are merged away")
    ap.add_argument("--speed-ratio", type=float, default=0.8)
    ap.add_argument("--far-m", type=float, default=13)
    ap.add_argument("--moving-w", type=float, default=1.5)
    ap.add_argument("--out-w", type=float, default=4)
    ap.add_argument("--invis-w", type=float, default=0.3)
    ap.add_argument("--out-margin", type=float, default=0.005)
    ap.add_argument("--out-hold-s", type=float, default=3.0, help="seconds of held dead evidence after the ball is seen out")
    ap.add_argument("--out-hold-w", type=float, default=2.0)
    ap.add_argument("--ctrl-soft", type=float, default=0.0, help="metres; >0 switches to distance-weighted control evidence")
    ap.add_argument("--shape-w", type=float, default=0.15, help="weight of the team-spread evidence")
    ap.add_argument("--shape-invis-only", type=int, default=1, help="1: shape evidence only on frames without a ball")
    ap.add_argument("--restart-w", type=float, default=0.5)
    ap.add_argument("--ctrl-m", type=float, default=1.5, help="a player this close has the ball")
    ap.add_argument("--carry-w", type=float, default=0.25, help="evidence for the last controlling team while nobody has it")
    ap.add_argument("--carry-s", type=int, default=60, help="seconds the carry lasts")
    ap.add_argument("--touch-w", type=float, default=3.0, help="evidence impulse per detected touch")
    ap.add_argument("--touch-dv", type=float, default=4.0, help="ball velocity change (m/s) that counts as a touch")
    ap.add_argument("--recv-w", type=float, default=0.0, help="evidence impulse per detected reception")
    ap.add_argument("--recv-vhi", type=float, default=4.0)
    ap.add_argument("--recv-vlo", type=float, default=2.5)
    ap.add_argument("--recv-m", type=float, default=2.5)
    return ap


def main(argv=None):
    args = argparser().parse_args(argv)
    p = vars(args)
    self_check()
    labels = build(p)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, args.name + ".csv")
    write(labels, out)
    graded = np.zeros(len(labels), bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    counts = {l: int(((labels == l) & graded).sum()) for l in "HADX"}
    print("wrote", os.path.relpath(out, ROOT), "graded counts", counts)


if __name__ == "__main__":
    main()
