# SyntaxEror — More than a Hack 2027, Phase 1: Ball Possession

Our submission for the [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027)
initial challenge: one possession label (`H` / `A` / `D` / `X`) for every frame of the tactical
video of FC Barcelona vs Athletic Club, produced from the automatic tracking and the Smart Tagging.

The method is a per-frame evidence model, hand-written terms plus a small classifier trained
on SkillCorner's open tracking data (twenty other matches), decoded with a 3-state Viterbi, so
possessions come out as whole segments rather than as a majority vote over frames. It is described in
[docs/technical_document.md](docs/technical_document.md).

## Reproduce the submission

Requirements: Python 3.10 or newer, `numpy`, `pandas` and `lightgbm` (pinned in
`requirements.txt`). The file was generated with Python 3.14.6, numpy 2.4.6, pandas 3.0.6 and
lightgbm 4.7.0.

```bash
pip install -r requirements.txt
python make_submission.py
```

That writes `out/SyntaxEror.csv`, 199,319 rows, in about three minutes the first time (the tracking
file is parsed into `data/cache/`) and a few seconds afterwards. The run is deterministic: two runs
produce byte-identical files.

### Where the data goes

The match assets are gated and are **not** in this repository. Put the bundle in `./data` with the
layout the organisers' repository expects, or point `MTAH_DATA_DIR` at it:

```
data/
└─ tactical/
   ├─ BAR-ATH_tactical_atd.xml
   ├─ BAR-ATH_tactical_atd.txt
   └─ BAR-ATH_tactical_smart_tagging.xml
```

Only those three files are read. The videos, the homography and the broadcast recording are not
used.

### Check it

With the organisers' repository cloned next to this one:

```bash
python ../more-than-a-hack-2027/evaluation/validate_submission.py out/SyntaxEror.csv
```

## Layout

```
make_submission.py          the one command
possession/
  features.py               tracking file -> cached per-frame features (ball, distances, speeds)
  model.py                  evidence terms + Viterbi decoder; all parameters are CLI flags
  learned.py                classifier on top of the evidence terms, trained on other matches
  skillcorner.py            SkillCorner's open data -> features and labels for that training
  external.py               Metrica's public sample games -> the same, the first source we tried
models/                     the three trained classifiers the submission decodes with
  touches.py                ball touches from velocity changes, attributed to the nearest team
  smart_tagging.py          Smart Tagging XML reader (vendored from the organisers' repo, MIT)
  games.py                  match constants and asset paths (vendored, MIT; one path edit)
labeling/                   hand-labelling kit for the internal validation set (see its README)
analysis/
  counts.py                 full-match possession / stoppage counts of a submission
  diag.py                   per-segment view of a submission against the public ground truth
  study_gt.py               the aggregates that drove the design
docs/technical_document.md  the technical document
```

Every model parameter is a flag on `python -m possession.model --help` (hand evidence only) and
`python -m possession.learned --help`. The defaults are the submission.

## Licence

Our code is MIT. `possession/games.py` and `possession/smart_tagging.py` are adapted from
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027),
MIT, see `LICENSE-metrica-sports.txt`. The classifier was trained on
[SkillCorner's open data](https://github.com/SkillCorner/opendata) (MIT), which we credit with thanks;
[Metrica Sports' sample data](https://github.com/metrica-sports/sample-data) served as the first
source we tried. Neither dataset is redistributed here. The match data is not covered by any licence here and is not
redistributed.
