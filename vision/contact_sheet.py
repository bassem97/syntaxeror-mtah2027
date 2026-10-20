"""A contact sheet for a human to check the auto-labels: crops around the projected tracked ball
with the projection drawn as a circle and the blob-refined point as a cross.

    python vision/contact_sheet.py --n 48 --out labeling/data/labels_check.png

Isolated frames only (ball more than 3 m from every player), so what is near the marks is the
ball or nothing. Count how many circles and how many crosses sit on the ball.
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession.features import load  # noqa: E402
from possession.games import asset_paths  # noqa: E402
from vision.autolabels import load_labels  # noqa: E402
from vision.check_labels import brightest_patch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--half", type=int, default=48, help="half size of each crop in px")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "labeling", "data", "labels_check.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    u, v, _ = load_labels()
    f = load()
    dmin = np.nan_to_num(np.fmin(f["d_home"], f["d_away"]), nan=0.0)
    speed = np.nan_to_num(f["ball_speed"], nan=99.0)
    ok = np.flatnonzero(~np.isnan(u) & (dmin > 3.0) & (speed < 8.0))
    rng = np.random.default_rng(args.seed)
    frames = np.sort(rng.choice(ok, size=args.n, replace=False))
    cap = cv2.VideoCapture(asset_paths()["video"])
    tiles = []
    h = args.half
    for fr in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ret, img = cap.read()
        if not ret:
            continue
        cx, cy = int(round(u[fr])), int(round(v[fr]))
        x0, y0 = max(0, cx - h), max(0, cy - h)
        crop = img[y0:y0 + 2 * h, x0:x0 + 2 * h].copy()
        if crop.shape[0] < 2 * h or crop.shape[1] < 2 * h:
            continue
        crop = cv2.resize(crop, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
        s = args.scale
        cv2.circle(crop, ((cx - x0) * s, (cy - y0) * s), 7 * s, (0, 255, 255), 1)  # projection: yellow circle
        hit = brightest_patch(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), u[fr], v[fr], 20)
        if hit is not None and hit[2] >= 12:
            bx, by = int(hit[0]) - x0, int(hit[1]) - y0
            cv2.drawMarker(crop, (bx * s, by * s), (255, 0, 255), cv2.MARKER_CROSS, 10 * s, 1)  # refined: magenta cross
        cv2.putText(crop, str(fr), (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(crop)
    cap.release()
    cols = 8
    rows = (len(tiles) + cols - 1) // cols
    side = 2 * h * args.scale
    sheet = np.zeros((rows * side, cols * side, 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * side:(r + 1) * side, c * side:(c + 1) * side] = t
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, sheet)
    print("wrote", args.out, "with", len(tiles), "crops; yellow circle = projected tracked ball, magenta cross = blob-refined label")


if __name__ == "__main__":
    main()
