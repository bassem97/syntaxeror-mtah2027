"""Regenerate the submission file from the provided match data. Deterministic, numpy + pandas only.

    python make_submission.py          # -> out/SyntaxEror.csv

Needs the data bundle in ./data or at $MTAH_DATA_DIR (same layout the organisers' repo expects).
The first run parses the tracking file into data/cache/ (about three minutes); later runs take
seconds.
"""
import warnings

from possession.model import main

if __name__ == "__main__":
    warnings.simplefilter("ignore")  # nanmean over empty halftime frames
    main([])
