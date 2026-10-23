"""PFF FC's 2022 World Cup data (https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset,
free on request) as more training data: broadcast tracking of the 64 matches at 29.97 fps, every
player placed (estimated when out of shot), the ball only when seen, and every on-the-ball event
with its team written into the frames it spans.

    python -m possession.pff --root ../external/pff --out ../external/cache [--matches 3812,...]

Each match becomes a feature file in the layout of possession/features.py, resampled to 25 fps,
plus a per-frame label: H or A for the team of the last on-the-ball event (the ball in flight
belongs to the team that played it, as in the organisers' convention), D from a ball-out or
period-end event until the next on-the-ball event (PFF writes a ball-out event at the whistle
after a foul too, so free-kick stoppages come out the same way). Frames the match's event file
marks as missing video, and dead frames with fewer than `min_visible` players in the broadcast
image (a replay or a close-up: the positions are estimates), are X, not graded.

PFF has no ball while it is out of play (seen on 1% of dead frames, against 29% in the
challenge's hand-labelled windows), which would let a classifier learn "ball seen = alive". So
on dead frames the ball is drawn on the straight line from where it was last seen to where it
reappears, the way the challenge tracker follows a ball carried to the restart, and blanked in
runs drawn from the challenge tracking's gap lengths (labeling/data/ball_gap_stats.npz) down to
about the challenge's rate.
"""
import argparse
import bz2
import glob
import json
import os

import numpy as np

from .features import derive

FPS = 25
LEN, WID = 105.0, 68.0
MAX_SLOTS = 20                                   # per team: starters, subs and a spare row or two
KICKOFFS = {"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF"}


def resample(a, t_src, t_dst):
    """[n_src, ...] sampled at t_src -> [n_dst, ...] at t_dst by linear interpolation; NaN where
    the nearest source sample was NaN, so gaps stay gaps. Also returns the nearest source index."""
    flat = a.reshape(len(t_src), -1)
    out = np.full((len(t_dst), flat.shape[1]), np.nan)
    near = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    near = np.where(t_dst - t_src[near - 1] < t_src[near] - t_dst, near - 1, near)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        ok = ~np.isnan(col)
        if ok.sum() >= 2:
            out[:, c] = np.interp(t_dst, t_src[ok], col[ok])
        out[np.isnan(col[near]), c] = np.nan
    return out.reshape((len(t_dst),) + a.shape[1:]), near


def read_frames(path):
    """One pass over the tracking file: per period, times, positions by jersey slot, the ball,
    the number of players seen in the image, and the game event covering each frame."""
    slots = {"home": {}, "away": {}}
    out = {}
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            per = d.get("period")
            if per not in (1, 2):
                continue
            g = out.setdefault(per, {"t": [], "vt": [], "pl": [], "ball": [], "ev": [], "nvis": []})
            g["t"].append(float(d["periodElapsedTime"]))
            g["vt"].append(float(d["videoTimeMs"]) / 1000)
            row = np.full((2 * MAX_SLOTS, 2), np.nan)
            nvis = 0
            for side, off in (("home", 0), ("away", MAX_SLOTS)):
                for p in d.get(side + "PlayersSmoothed") or []:
                    nvis += p.get("visibility") == "VISIBLE"
                    j = p.get("jerseyNum")
                    if j is None or p.get("x") is None:
                        continue
                    k = slots[side].setdefault(j, len(slots[side]))
                    if k < MAX_SLOTS and np.isnan(row[off + k, 0]):
                        row[off + k] = (p["x"], p["y"])
            g["pl"].append(row)
            g["nvis"].append(nvis)
            b = d.get("ballsSmoothed") or {}
            g["ball"].append((b["x"], b["y"]) if b.get("x") is not None else (np.nan, np.nan))
            e = d.get("game_event") or {}
            g["ev"].append((e.get("game_event_type"), e.get("home_ball")))
    return out, slots


def labels_from_events(ev, vid):
    """Per-frame labels at the source rate from the game event covering each frame."""
    n = len(ev)
    lab = np.full(n, "D", dtype="<U1")
    team, dead = None, True
    for i, (typ, home_ball) in enumerate(ev):
        if typ in ("OUT", "END"):
            dead = True
        elif typ in KICKOFFS or typ == "OTB":
            dead = False
            if home_ball is not None:
                team = "H" if home_ball else "A"
        lab[i] = "D" if dead or team is None else team
    for a, b in vid:
        lab[a:b] = "X"
    return lab


def dead_ball(ball, dead, stats_path, seed):
    """Ball on dead frames: the straight line between the last seen and the next seen position,
    kept in runs drawn from the challenge tracking's run lengths (about 30% of the frames)."""
    b = ball.copy()
    seen = ~np.isnan(b[:, 0])
    if seen.sum() < 2:
        return b
    idx = np.arange(len(b))
    fill = dead & ~seen
    for c in range(2):
        b[fill, c] = np.interp(idx[fill], idx[seen], b[seen, c])
    rng = np.random.default_rng(seed)
    st = np.load(stats_path)
    keep = np.zeros(len(b), bool)
    i = int(rng.choice(st["gaps"]))
    while i < len(b):
        k = int(rng.choice(st["seen"])) // 2
        keep[i:i + k] = True
        i += k + int(rng.choice(st["gaps"]))
    b[fill & ~keep] = np.nan
    return b


def convert_match(root, match, out, stats_path, min_visible=6):
    frames, slots = read_frames(os.path.join(root, "tracking", "{}.jsonl.bz2".format(match)))
    missing = video_missing(os.path.join(root, "events", "{}.json".format(match)))
    parts, balls, labs, periods, off = [], [], [], [], 0
    for per in sorted(frames):
        g = frames[per]
        t = np.array(g["t"])
        order = np.argsort(t, kind="stable")
        t = t[order]
        vt = np.array(g["vt"])[order]
        vid = [(int(np.searchsorted(vt, a)), int(np.searchsorted(vt, b))) for a, b in missing.get(per, [])]
        pl = np.stack(g["pl"])[order]
        ball = np.array(g["ball"], dtype=float)[order]
        lab_src = labels_from_events([g["ev"][i] for i in order], vid)
        lab_src[(lab_src == "D") & (np.array(g["nvis"])[order] < min_visible)] = "X"
        t25 = np.arange(0, t[-1], 1.0 / FPS)
        pl25, near = resample(pl, t, t25)
        bl25, _ = resample(ball, t, t25)
        lab25 = lab_src[near]
        bl25 = dead_ball(bl25, lab25 == "D", stats_path, seed=int(match) * 10 + per)
        pl = np.full((len(t25), 2 * MAX_SLOTS, 3), np.nan)
        pl[:, :, 0] = (pl25[:, :, 0] + LEN / 2) / LEN
        pl[:, :, 1] = (pl25[:, :, 1] + WID / 2) / WID
        sp = np.full(pl25.shape[:2], np.nan)
        sp[2:-2] = np.sqrt((pl25[4:, :, 0] - pl25[:-4, :, 0]) ** 2 + (pl25[4:, :, 1] - pl25[:-4, :, 1]) ** 2) * FPS / 4
        pl[:, :, 2] = sp
        parts.append(pl)
        balls.append(np.stack([(bl25[:, 0] + LEN / 2) / LEN, (bl25[:, 1] + WID / 2) / WID], axis=1))
        labs.append(lab25)
        periods.append([per, off + 1, off + len(t25)])
        off += len(t25)
    players = np.concatenate(parts)
    ball = np.concatenate(balls)
    lab = np.concatenate(labs)
    used = ~np.all(np.isnan(players[:, :, 0]), axis=0)          # drop empty slots
    is_home = np.array([True] * MAX_SLOTS + [False] * MAX_SLOTS)[used]
    players = players[:, used]
    arrays = derive(players, ball, is_home, LEN, WID, FPS)
    arrays["ball_src"] = np.where(np.isnan(ball[:, 0]), 0, 1).astype(np.int8)
    arrays["label"] = lab
    arrays["periods"] = np.array(periods)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "pff_{}.npz".format(match)), **arrays)
    graded = lab != "X"
    runs = int(((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"])).sum())
    print("pff_{}: {} frames, graded {:.0f} min, H {:.2f} A {:.2f} D {:.2f}, X {:.0%}, {} possessions, ball seen {:.0%}, players {}+{}".format(
        match, len(lab), graded.sum() / FPS / 60, *((lab == l).sum() / max(1, graded.sum()) for l in "HAD"), (~graded).mean(), runs,
        1 - np.isnan(ball[:, 0]).mean(), int(is_home.sum()), int((~is_home).sum())), flush=True)


def video_missing(path):
    """{period: [(start, end)]} video-time spans the event file marks as missing video: from such
    an event to the next event of the period. Without the file nothing is marked."""
    if not os.path.exists(path):
        return {}
    rows = []
    for e in json.load(open(path, encoding="utf-8")):
        g = e.get("gameEvents") or {}
        if g.get("period") is not None and e.get("startTime") is not None:
            rows.append((int(g["period"]), float(e["startTime"]), bool(g.get("videoMissing"))))
    rows.sort()
    out = {}
    for (per, t, miss), nxt in zip(rows, rows[1:] + [None]):
        if miss and nxt is not None and nxt[0] == per and nxt[1] > t:
            out.setdefault(per, []).append((t, nxt[1]))
    return out


def self_check():
    a = np.array([[0.0], [1.0], [np.nan], [3.0]])
    t = np.array([0.0, 1.0, 2.0, 3.0])
    u, near = resample(a, t, np.array([0.0, 0.5, 1.9, 3.0]))
    assert abs(u[1, 0] - 0.5) < 1e-9 and np.isnan(u[2, 0]) and abs(u[3, 0] - 3.0) < 1e-9 and list(near) == [0, 1, 2, 3], (u.ravel(), near)
    b = np.array([[0.0, 0.0], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [4.0, 4.0]])
    d = np.array([False, True, True, False, False])
    rng_stats = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "labeling", "data", "ball_gap_stats.npz")
    if os.path.exists(rng_stats):
        f = dead_ball(b, d, rng_stats, 0)
        assert np.isnan(f[3, 0]) and all(np.isnan(f[i, 0]) or abs(f[i, 0] - i) < 1e-9 for i in (1, 2)), f
    lab = labels_from_events([("FIRSTKICKOFF", True), (None, None), ("OTB", False), ("OUT", None), (None, None), ("OTB", True)], [(1, 2)])
    assert "".join(lab) == "HXADDH", lab


if __name__ == "__main__":
    self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with tracking/ and events/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--matches", default="")
    ap.add_argument("--min-visible", type=int, default=6, help="dead frames with fewer players seen in the image are not graded")
    ap.add_argument("--gap-stats", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "labeling", "data", "ball_gap_stats.npz"))
    args = ap.parse_args()
    import warnings
    warnings.simplefilter("ignore")
    matches = args.matches.split(",") if args.matches else sorted(os.path.basename(p)[:-10] for p in glob.glob(os.path.join(args.root, "tracking", "*.jsonl.bz2")))
    for m in matches:
        if os.path.exists(os.path.join(args.out, "pff_{}.npz".format(m))):
            print(m, "already converted", flush=True)
            continue
        convert_match(args.root, m, args.out, args.gap_stats, args.min_visible)
