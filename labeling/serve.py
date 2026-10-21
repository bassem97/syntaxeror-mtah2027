"""Serve the labelling page locally so it needs no file pickers and saves marks to disk itself.

    python labeling/serve.py                 # then open http://localhost:8765
    python labeling/serve.py --video path    # a different video file (e.g. the 540p proxy)

Routes:
  /                      the labelling page
  /video.mp4             the tactical video, with HTTP range support so seeking works
  /ref/public.csv        the public ground truth (from ../mtah, if present)
  /ref/model.csv         out/SyntaxEror.csv, if present
  /marks/<window>.csv    GET resumes, PUT saves -> labeling/data/marks_<window>.csv

Everything stays on this machine. Standard library only.
"""
import argparse
import os
import time
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession.games import asset_paths  # noqa: E402

DATA = os.path.join(HERE, "data")
PAGE = os.path.join(HERE, "label_tool.html")
REFS = {
    "public.csv": os.path.join(ROOT, "..", "mtah", "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"),
    "model.csv": os.path.join(ROOT, "out", "SyntaxEror.csv"),
}
VIDEO = None


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet, except errors
        if args and str(args[0]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            return self.send_file(PAGE, "text/html; charset=utf-8")
        if self.path == "/video.mp4":
            return self.send_file(VIDEO, "video/mp4")
        if self.path.startswith("/ref/"):
            name = self.path[5:]
            if name in REFS and os.path.isfile(REFS[name]):
                return self.send_file(REFS[name], "text/csv")
            return self.send_error(404)
        if self.path.startswith("/marks/"):
            path = self.marks_path()
            if path and os.path.isfile(path):
                return self.send_file(path, "text/csv")
            return self.send_error(404)
        return self.send_error(404)

    def do_POST(self):
        # one ball click: "frame,u,v" appended to labeling/data/clicks.csv
        if self.path != "/clicks":
            return self.send_error(404)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("ascii", "ignore").strip()
        parts = body.split(",")
        if len(parts) != 3 or not all(p.strip().lstrip("-").isdigit() for p in parts):
            return self.send_error(400)
        os.makedirs(DATA, exist_ok=True)
        path = os.path.join(DATA, "clicks.csv")
        new = not os.path.isfile(path)
        with open(path, "a", newline="") as fh:
            if new:
                fh.write("frame,u,v\n")
            fh.write(body + "\n")
        self.send_response(204)
        self.end_headers()

    def do_PUT(self):
        path = self.marks_path()
        if not path:
            return self.send_error(404)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        os.makedirs(DATA, exist_ok=True)
        # A save with fewer marks than the file has is the only way labels get lost: keep the
        # old file in marks_history/ so any wipe can be undone.
        if os.path.isfile(path):
            old = open(path, "rb").read()
            if old.count(b"\n") > body.count(b"\n"):
                hist = os.path.join(DATA, "marks_history")
                os.makedirs(hist, exist_ok=True)
                with open(os.path.join(hist, time.strftime("%Y%m%d-%H%M%S_") + os.path.basename(path)), "wb") as fh:
                    fh.write(old)
        with open(path, "wb") as fh:
            fh.write(body)
        self.send_response(204)
        self.end_headers()

    def marks_path(self):
        name = self.path[len("/marks/"):]
        if not name.endswith(".csv") or "/" in name or "\\" in name:
            return None
        safe = "".join(c for c in name[:-4] if c.isalnum() or c in "-_")
        return os.path.join(DATA, "marks_{}.csv".format(safe)) if safe else None

    def send_file(self, path, ctype):
        if not path or not os.path.isfile(path):
            return self.send_error(404)
        size = os.path.getsize(path)
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if (b and a) else size - 1
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionAbortedError, BrokenPipeError):
                    return
                left -= len(chunk)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="video file to serve (default: the tactical video from the data dir)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    VIDEO = args.video or asset_paths()["video"]
    # Prefer the seek-friendly proxy when it sits next to the original (same frame numbers):
    #   ffmpeg -i BAR-ATH_tactical.mp4 -vf scale=960:-2 -r 25 -vsync cfr -c:v libx264 -preset veryfast
    #          -crf 24 -g 12 -keyint_min 12 -sc_threshold 0 -an -movflags +faststart BAR-ATH_tactical_proxy540.mp4
    proxy = VIDEO[:-4] + "_proxy540.mp4"
    if not args.video and os.path.isfile(proxy):
        VIDEO = proxy
    if not os.path.isfile(VIDEO):
        sys.exit("video not found: {}".format(VIDEO))
    print("serving", VIDEO)
    print("open http://localhost:{}".format(args.port))
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
