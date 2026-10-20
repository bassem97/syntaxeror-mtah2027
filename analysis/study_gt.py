"""Aggregate view of the public 5-minute ground truth against the cached features and the tagging.

Prints one line per GT segment -- no raw tracking rows -- to understand what D looks like.

    python analysis/study_gt.py ../mtah/evaluation/gt/BAR-ATH_possession_gt_public.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import smart_tagging as st  # noqa: E402
from possession.features import load  # noqa: E402

gt = pd.read_csv(sys.argv[1])["label"].values
f = load()
events = st.load()
sp = [(e["start_frame"], e["tags"].get("Type"), e["tags"].get("Team")) for e in events if e["code"] == "SET PIECES"]
sp.sort()
sp_frames = np.array([s[0] for s in sp])
goals = [(e["start_frame"], e["end_frame"], e["tags"].get("Team")) for e in events if e["code"] == "GOALS"]
print("set pieces (start_frame, type, team) in window:",
      [s for s in sp if 11185 - 300 <= s[0] <= 18684 + 300])
print("goals:", goals)

# segments of the graded window
lab = gt.copy()
idx = np.where(lab != "X")[0]
start = idx[0]
segs = []
for i in idx[1:]:
    if lab[i] != lab[start]:
        segs.append((start, i - 1))
        start = i
segs.append((start, idx[-1]))

out_x = (f["ball_x"] < 0) | (f["ball_x"] > 1)
out_y = (f["ball_y"] < 0) | (f["ball_y"] > 1)
vis = ~np.isnan(f["ball_x"])
print("\n{:>6} {:>6} {:>6} {:>4} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}  {}".format(
    "start", "end", "len", "lab", "vis%", "out%", "dH", "dA", "spd", "bspd", "nearest set piece (frame, type, team, offset)"))
for s, e in segs:
    sl = slice(s, e + 1)
    j = np.argmin(np.abs(sp_frames - s))
    print("{:>6} {:>6} {:>6} {:>4} {:>5.0f} {:>5.0f} {:>5.1f} {:>5.1f} {:>5.2f} {:>5.1f}  {} off={:+d}".format(
        s, e, e - s + 1, lab[s], 100 * vis[sl].mean(), 100 * (out_x[sl] | out_y[sl]).mean(),
        np.nanmedian(f["d_home"][sl]), np.nanmedian(f["d_away"][sl]),
        np.nanmean(f["mean_speed"][sl]), np.nanmedian(f["ball_speed"][sl]),
        sp[j][1:], int(sp[j][0] - s)))

# how does the nearest-team signal relate to the GT team on alive frames?
alive = (gt == "H") | (gt == "A")
nearest = np.where(f["d_home"] < f["d_away"], "H", "A")
ok = alive & vis
print("\nnearest-team agreement on alive+visible frames: {:.3f}  (n={})".format(
    (nearest[ok] == gt[ok]).mean(), ok.sum()))
for r in (1.0, 1.5, 2.0, 3.0):
    close = ok & (np.minimum(f["d_home"], f["d_away"]) < r)
    print("  within {:.1f} m: agreement {:.3f}  frac of alive frames {:.2f}".format(
        r, (nearest[close] == gt[close]).mean(), close.sum() / alive.sum()))
