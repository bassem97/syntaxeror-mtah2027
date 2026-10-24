# Hand labels

**Ruling (organisers, 23 Oct 2026): hand labels of the challenge match may not be used for any
purpose, not training, not validation, not choosing parameter values.** The windows of the
challenge match below were labelled before that answer; their marks stay in `labeling/data/`
(not committed) and are not used anywhere. What the organisers allow is to annotate *another*
match, choose the values on it, apply them unchanged, and share the video and annotations. That
is the section "Another match" at the end; the kit, keys and convention are the same.

## The challenge match (labelled before the ruling; not used)

## Windows

| window | frames | where in the match |
|---|---|---|
| calibration | 11185–18684 | the public ground truth: label it first, then score yourself against the truth |
| W1 | 50000–57499 | first half, about minute 26 |
| W2 | 115000–122499 | second half, about minute 4 |
| W3 | 166000–173499 | second half, about minute 37 |
| W4 | 68000–75499 | first half, about minute 38, stoppage-heavy |
| W5 | 143000–150499 | second half, about minute 20 |
| W6 | 30000–37499 | first half, about minute 13 |
| W7 | 128000–135499 | second half, about minute 10 |
| W8 | 180000–187499 | second half, about minute 44, the closing minutes |
| W9 | 38700–46199 | first half, about minute 18, open play |
| W10 | 76200–83699 | first half, about minute 43, open play |
| W11 | 151250–158749 | second half, about minute 25, open play |

Each labeller does the calibration window before any real window. Score your calibration marks
against the public truth with the organisers' scorer (below): `macroF1` above 0.95 and every true
possession matched at IoU 0.7 means your convention matches theirs. If not, compare your marks with
the reference track in the tool and adjust before continuing.

## Tool

Start the local server from the repository root, then open the page in Chrome or Edge:

```bash
python labeling/serve.py
```

```
http://localhost:8765
```

Nothing leaves the machine. The page loads the video, the public ground truth and the current
model output by itself, so a reload needs no file picking, and every key press saves your marks to
`labeling/data/marks_all.csv`, one file for everything you mark. The window field only tells the
page which stretch you are working on (marks outside it are dimmed in the list); the converter
picks each window out of the file.

The server prefers `BAR-ATH_tactical_proxy540.mp4` when it sits next to the original: a
half-resolution copy with the same 199,319 frames and a keyframe every half second, so stepping
backwards is instant. The ffmpeg command that makes it is in `serve.py`; pass `--video` to serve
a specific file instead.

Keys: `Enter` plays and pauses, arrows step 1 frame (Shift: 10), up/down jump 5 s, `,` `.` change speed.
`Q` Barça, `L` Athletic, `Space` dead ball (also `H` `A` `D` or `1` `2` `3`) mean "from this frame
on, this state", and every later mark inside the window is cleared, because going back means
re-watching from there. Marks in other windows are never touched, so set the window field first. `Shift` + key sets the state at the current frame but keeps the later
marks (insert a change without wiping what follows). `Backspace` deletes the mark at or before the
cursor. The export button is only needed when
the page is opened as a file without the server.

Check the clock once: at frame 11185 the first-half kick-off is being struck, and at 113742 the
second-half one. If either is off by more than a frame or two, stop and report it.

## Ball clicks (a test set for the ball detector)

The detector can only be trained where the tracker already saw the ball, so it needs a test set
on the frames the tracker lost. Press `B` in the tool to enter ball-click mode (the cursor becomes
a crosshair), then click the ball in the video; each click saves `frame,u,v` in 1920x1080 pixels
to `labeling/data/clicks.csv`. `B` again leaves the mode. Pick frames where the model output and
the public truth disagree, or stretches the tracker lost, and click the ball on perhaps 300
frames spread over the match; one click per frame is enough. Score the detector on them with:

```bash
python vision/eval_clicks.py
```

## Convention (match the public truth, not your intuition)

- `H` Barcelona, `A` Athletic, `D` dead ball. Nothing else on graded frames.
- Possession changes on the **first touch by the other team that is controlled**: a pass received,
  a dribble, a clearance, a header that goes where the player meant. A deflection, a touch in a
  duel that the same opponent immediately wins back, a rebound off a player who had no control, do
  not change it. When in doubt, look at what the ball does next: if the touching team keeps it for
  a second or makes a pass, it was control.
- The truth contains possessions under a second (a one-touch interception followed by losing it
  again). Mark them.
- The change is at the touch frame, not when the player approaches the ball. For a header, the
  frame the head meets the ball, even if the tracked ball looks far away.
- `D` starts the frame the ball crosses the line, enters the goal, or the referee's whistle takes
  effect (players visibly stop). `D` ends at the restart touch: the throw leaves the hands, the
  kick is struck. The team restarting is in possession from that frame.
- Between a goal and the kick-off: `D`. At half-time and before the first kick-off: nothing to mark,
  those frames are never scored.

## From marks to a scorable file

```bash
python labeling/marks_to_gt.py labeling/data/marks_all.csv --window 50000-57499 -o labeling/data/gt_w1.csv
```

Merge several windows, and the public truth, into the internal set:

```bash
python labeling/marks_to_gt.py labeling/data/marks_all.csv --window 50000-57499 --window 115000-122499 --window 166000-173499 --window 68000-75499 --window 143000-150499 --window 30000-37499 --window 128000-135499 --window 180000-187499 --window 38700-46199 --window 76200-83699 --window 151250-158749 --public ../mtah/evaluation/gt/BAR-ATH_possession_gt_public.csv -o labeling/data/gt_internal.csv
```

Score the model, or a labeller's calibration marks exported through the same converter, with the
organisers' own scorer:

```bash
python ../mtah/evaluation/run_evaluation.py --gt labeling/data/gt_internal.csv --results out --out labeling/data
```

Two labellers on the same window is the cheapest quality check: convert both and score one as a
"submission" against the other.

## Another match: SoccerTrack v2, match 117093

A university match from SoccerTrack v2 (CC BY 4.0, Hugging Face `atomscott/SoccerTrack-v2`),
full-pitch panoramic video of both halves, chosen because the labels we make on it follow the
organisers' convention and the organisers allow choosing parameter values on a match annotated
elsewhere. `analysis/prepare_match.py` builds one 25 fps proxy of both halves back to back
(first half frames 0 to 67624, second half 67625 to 138099), converts the positions to the same
frame layout and writes the model's own decode as the reference track under the timeline.

```bash
python labeling/serve.py --port 8766 --video ../external/soccertrack/117093_proxy.mp4 --marks st117093 --n 138100 --home Left --away Right --window 1250-8749 --model-ref out/soccertrack_117093_model.csv
```

`Q` is **Left**, the team defending the left goal at the first-half kick-off, and `L` is
**Right**, the other team. The teams swap ends at half-time: keep the keys with the teams, not
with the sides. Marks save to `labeling/data/marks_st117093.csv`, separate from the challenge
match's file.

| window | frames | where in the match |
|---|---|---|
| S1 | 1250–8749 | first half, minute 1 |
| S2 | 18750–26249 | first half, minute 12 |
| S3 | 36250–43749 | first half, minute 24 |
| S4 | 53750–61249 | first half, minute 36 |
| S5 | 68875–76374 | second half, minute 1 |
| S6 | 86375–93874 | second half, minute 12 |
| S7 | 103875–111374 | second half, minute 24 |
| S8 | 121375–128874 | second half, minute 36 |

Every window starts with a mark (set the state at its first frame before anything else). The
convention is the one above, word for word. From marks to a file the fit reads:

```bash
python labeling/marks_to_gt.py labeling/data/marks_st117093.csv --n-frames 138100 --window 1250-8749 --window 18750-26249 --window 36250-43749 --window 53750-61249 --window 68875-76374 --window 86375-93874 --window 103875-111374 --window 121375-128874 -o labeling/soccertrack_117093_labels.csv
```

That file is committed (it is the annotation the organisers asked us to share) and
`analysis/fit_oos.py --games soccertrack_117093 --labels labeling/soccertrack_117093_labels.csv`
chooses the parameter values on it.
