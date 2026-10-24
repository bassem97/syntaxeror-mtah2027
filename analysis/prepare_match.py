"""Prepare a SoccerTrack v2 match for hand labelling with labeling/: one proxy video of both
halves back to back at 25 fps, the match's features in the same frame layout, and the model's
own decode as the reference track the labelling page shows.

    python analysis/prepare_match.py --root ../external/soccertrack --match 117093 --out ../external/cache

Writes <root>/<match>_proxy.mp4, <out>/soccertrack_<match>.npz (via possession/soccertrack.py,
halves stitched exactly as the proxy: half 1 frames first, then half 2) and
out/soccertrack_<match>_model.csv, and prints the serve.py command to run.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def probe(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                                   "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
                                   "-of", "json", path])
    st = json.loads(out)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return int(st["width"]), int(st["height"]), float(num) / float(den), int(st["nb_read_packets"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--match", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=2560, help="proxy width (the panorama is 4096 wide)")
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()
    m = args.match
    halves = [os.path.join(args.root, "videos", m, "{}_panorama_{}_half.mp4".format(m, h)) for h in ("1st", "2nd")]
    counts = []
    for path in halves:
        w, h, fps, n = probe(path)
        print("{}: {}x{} {:.3f} fps {} frames".format(os.path.basename(path), w, h, fps, n), flush=True)
        counts.append(n)
    proxy = os.path.join(args.root, "{}_proxy.mp4".format(m))
    if not args.skip_video and not os.path.exists(proxy):
        # each half to its own constant-25-fps proxy (keyframe every 12 frames), then joined
        # without re-encoding: frame n of half 2 is counts[0] + n
        parts = []
        for path, tag in zip(halves, ("1st", "2nd")):
            part = os.path.join(args.root, "{}_proxy_{}.mp4".format(m, tag))
            if not os.path.exists(part):
                # every source frame exactly once, at n/25 s: a constant-rate conversion duplicated
                # frames on timestamp jitter and put the proxy 24 frames out of step with the positions
                subprocess.check_call(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf", "scale={}:-2,setpts=N/(25*TB)".format(args.width),
                                       "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                                       "-g", "12", "-keyint_min", "12", "-sc_threshold", "0", "-an", "-movflags", "+faststart", part])
            parts.append(part)
        lst = os.path.join(args.root, "{}_concat.txt".format(m))
        with open(lst, "w") as fh:
            for part in parts:
                fh.write("file '{}'\n".format(os.path.abspath(part).replace("\\", "/")))
        subprocess.check_call(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                               "-movflags", "+faststart", proxy])
    w, h, fps, n = probe(proxy)
    print("proxy: {}x{} {:.3f} fps {} frames (halves {} + {} = {})".format(w, h, fps, n, counts[0], counts[1], sum(counts)), flush=True)

    from possession import soccertrack as S
    stats = os.path.join(ROOT, "labeling", "data", "ball_gap_stats.npz")
    npz = os.path.join(args.out, "soccertrack_{}.npz".format(m))
    if not os.path.exists(npz):
        S.convert_match(args.root, m, args.out, stats, keep_json=True, half_frames=counts)
    f = dict(np.load(npz))
    periods = f["periods"]
    print("features: {} frames, periods {}".format(len(f["label"]), periods.tolist()), flush=True)

    # the model's decode as a reference track for the labeller (hand evidence + committed classifier)
    import glob
    import lightgbm as lgb
    from possession import learned as L
    from possession import model as M
    p = {a.dest: a.default for a in M.argparser()._actions if a.dest != "help"}
    p["events"] = []
    lab = f.pop("label")
    t = L.frame_table(f, p)
    boosters = [lgb.Booster(model_str=open(path, newline="").read().replace(chr(13) + chr(10), chr(10))) for path in sorted(glob.glob(os.path.join(ROOT, "models", "learned_*.txt")))]
    proba = np.mean([b.predict(t.values) for b in boosters], axis=0)
    dec = L.decode_periods(M.emissions(f, p) + L.emissions_from(proba, 0.35), periods, p)
    ref = os.path.join(ROOT, "out", "soccertrack_{}_model.csv".format(m))
    with open(ref, "w", newline="") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(dec))
    print("reference track:", ref)
    print("\nrun:\n  python labeling/serve.py --video \"{}\" --marks st{} --n {} --home Left --away Right --window 0-7499 --model-ref \"{}\"".format(
        proxy, m, sum(counts), ref))


if __name__ == "__main__":
    main()
