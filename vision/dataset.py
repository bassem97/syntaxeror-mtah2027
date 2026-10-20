"""Build the ball-detector training set: crops around the auto-labelled ball, plus crops without it.

Frames are decoded once, sequentially, in-process with PyAV (random seeks in a 5 GB file are slow).
Every `--every` labelled frame yields one positive crop (ball somewhere inside, position
jittered) and one negative crop (same frame, far from the ball). The label is refined from the
projected point to the strongest ball-like blob within a few pixels (vision/check_labels.py shows
the projection sits a few pixels off); frames with no such blob keep the projected point. A column says whether the ball was isolated
(no player within 2 m), where the refined label is reliable; validation reads those separately.

    python vision/dataset.py --every 8          # ~14k positives + 14k negatives, 256 px crops

Output: data/cache/crops/{pos,neg}/<frame>.jpg and data/cache/crops/labels.csv
(file, frame, u, v in crop pixels; u = v = -1 for negatives).
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession.games import DATA_DIR, asset_paths  # noqa: E402
from vision.autolabels import load_labels  # noqa: E402
from vision.check_labels import brightest_patch  # noqa: E402

WIDTH, HEIGHT = 1920, 1080
CROPS = os.path.join(DATA_DIR, "cache", "crops")


def frames(video, step=1, start=0, end=None):
    """(frame_index, BGR ndarray) for every `step`-th frame, decoded in-process with PyAV.

    In-process decoding runs at about 500 fps on this machine; piping raw frames out of ffmpeg
    managed 13, because a 1080p frame is 6 MB."""
    import av
    c = av.open(video)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    try:
        i = -1
        for fr in c.decode(s):
            i += 1
            if i < start or i % step:
                continue
            if end is not None and i > end:
                break
            yield i, fr.to_ndarray(format="bgr24")
    finally:
        c.close()


def crop_at(img, cx, cy, size):
    x0 = int(np.clip(cx - size // 2, 0, WIDTH - size))
    y0 = int(np.clip(cy - size // 2, 0, HEIGHT - size))
    return img[y0:y0 + size, x0:x0 + size], x0, y0


def build(every, size, refine_radius, min_response, seed=0, limit=None):
    u, v, _ = load_labels()
    from possession.features import load as load_features
    feat = load_features()
    dmin = np.nan_to_num(np.fmin(feat["d_home"], feat["d_away"]), nan=0.0)
    # every k-th video frame, kept when it has a label (ffmpeg only hands over those k-th frames)
    labelled = np.flatnonzero(~np.isnan(u))
    labelled = labelled[labelled % every == 0]
    if limit:
        labelled = labelled[:limit]
    wanted = set(int(x) for x in labelled)
    rng = np.random.default_rng(seed)
    for sub in ("pos", "neg"):
        os.makedirs(os.path.join(CROPS, sub), exist_ok=True)
    rows, kept, skipped = [], 0, 0
    last = max(wanted) if wanted else -1
    gen = frames(asset_paths()["video"], step=every)
    for i, img in gen:
        if i > last:
            break
        if i not in wanted:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # refine to the ball-like blob when there is a clear one, else trust the projection
        hit = brightest_patch(gray, u[i], v[i], refine_radius)
        if hit is not None and hit[2] >= min_response:
            bu, bv = hit[0], hit[1]
        else:
            bu, bv = u[i], v[i]
            skipped += 1
        iso = int(dmin[i] > 2.0)
        # positive: ball anywhere in the crop, not always centred
        jx, jy = rng.integers(-size // 3, size // 3, size=2)
        crop, x0, y0 = crop_at(img, bu + jx, bv + jy, size)
        name = "pos/{:06d}.jpg".format(i)
        cv2.imwrite(os.path.join(CROPS, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append((name, i, bu - x0, bv - y0, iso))
        # negative: a crop that cannot contain the ball
        for _ in range(20):
            cx, cy = rng.integers(size // 2, WIDTH - size // 2), rng.integers(size // 2, HEIGHT - size // 2)
            if abs(cx - bu) > size or abs(cy - bv) > size:
                break
        crop, x0, y0 = crop_at(img, cx, cy, size)
        name = "neg/{:06d}.jpg".format(i)
        cv2.imwrite(os.path.join(CROPS, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append((name, i, -1, -1, iso))
        kept += 1
        if kept % 1000 == 0:
            print("  {} frames kept, {} skipped, at frame {}".format(kept, skipped, i), flush=True)
    gen.close()
    with open(os.path.join(CROPS, "labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "frame", "u", "v", "iso"])
        w.writerows(rows)
    return kept, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=8, help="use every k-th labelled frame")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--refine-radius", type=int, default=20)
    ap.add_argument("--min-response", type=float, default=12.0)
    ap.add_argument("--limit", type=int, help="stop after this many candidate frames (smoke test)")
    args = ap.parse_args()
    kept, skipped = build(args.every, args.size, args.refine_radius, args.min_response, limit=args.limit)
    print("done: {} positive + {} negative crops in {}; {} kept the projected point (no clear blob)".format(kept, kept, CROPS, skipped))
