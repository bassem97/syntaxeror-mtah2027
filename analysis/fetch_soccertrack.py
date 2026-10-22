"""Fetch SoccerTrack v2 match by match, convert each to features and labels, delete the JSON.

    python analysis/fetch_soccertrack.py --root ../external/soccertrack-v2 --out ../external/cache

Needs a Hugging Face login with access to atomscott/soccertrack-v2 (the dataset is gated).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058", "132831", "132877"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--matches", default=",".join(MATCHES))
    ap.add_argument("--keep-json", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from possession import soccertrack as S
    stats = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "labeling", "data", "ball_gap_stats.npz")
    for m in args.matches.split(","):
        if os.path.exists(os.path.join(args.out, "soccertrack_{}.npz".format(m))):
            print("match", m, "already converted", flush=True)
            continue
        t = time.time()
        for tag in ("1st", "2nd"):
            hf_hub_download("atomscott/soccertrack-v2", "gsr/{}/{}_{}.json".format(m, m, tag), repo_type="dataset", local_dir=args.root)
        print("match {}: downloaded in {:.0f} s".format(m, time.time() - t), flush=True)
        S.convert_match(args.root, m, args.out, stats, keep_json=args.keep_json)
        print("match {}: done in {:.0f} s".format(m, time.time() - t), flush=True)


if __name__ == "__main__":
    main()
