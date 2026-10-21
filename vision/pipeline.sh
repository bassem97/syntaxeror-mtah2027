#!/bin/sh
# The whole detector pipeline, unattended: hard negatives -> retrain -> whole-match inference ->
# gap filling -> possession features with the filled ball -> both submissions -> scores.
#
#   sh vision/pipeline.sh [gpu_share]          # log in $MTAH_DATA_DIR/cache/pipeline.log
#
# Needs the clean crops from vision/dataset.py and ballnet_clean.pt from vision/train.py.
set -e
SHARE=${1:-0.7}
cd "$(dirname "$0")/.."
C="$MTAH_DATA_DIR/cache"
LOG="$C/pipeline.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== pipeline start $(date) gpu share $SHARE ==="

echo "--- 1/8 hard negatives with the clean model ---"
python vision/mine_negatives.py --every 8 --per-frame 2 --weights "$C/ballnet_clean.pt" --gpu-share "$SHARE" 2>&1 | grep -v Warning

echo "--- 2/8 retrain on clean labels + hard negatives ---"
python vision/train.py --epochs 8 --batch 32 --out "$C/ballnet_final.pt" --gpu-share "$SHARE" 2>&1 | grep -v Warning
cp "$C/ballnet_final.pt" "$C/ballnet.pt"

echo "--- 3/8 detector over the whole match ---"
rm -f "$C/ball_detector.npz"
python vision/infer.py --batch 8 --gpu-share "$SHARE" 2>&1 | grep -v Warning

echo "--- 4/8 full-frame check on the public window ---"
python vision/eval_track.py --start 11185 --end 18684 2>&1 | grep -v Warning

echo "--- 5/8 gap filling (tracker) + simulated-gap check ---"
python vision/fill_gaps.py --simulate 150 --gap-frames 100-250 --method track --max-gap-s 30 2>&1 | grep -v Warning
python vision/fill_gaps.py --method track --max-gap-s 30 --min-score 0.4 2>&1 | grep -v Warning

echo "--- 6/8 possession features with the filled ball ---"
rm -f "$C/tactical_features_detector.npz"
python -m possession.features --variant detector --score-min 0.3 2>&1 | grep -v Warning

echo "--- 7/8 both submissions ---"
python make_submission.py 2>&1 | grep -v Warning
python -m possession.model --ball detector --name SyntaxEror_detector --out out 2>&1 | grep -v Warning
python ../mtah/evaluation/validate_submission.py out/SyntaxEror_detector.csv | head -1

echo "--- 8/8 scores ---"
R="$C/final_results"; rm -rf "$R"; mkdir -p "$R"
cp out/SyntaxEror.csv out/SyntaxEror_detector.csv "$R/"
echo "public window:"
python ../mtah/evaluation/run_evaluation.py --gt public --results "$R" --out "$C/final_out" --no-detail 2>&1 | sed -n '/^rank/,/^$/p'
echo "internal set:"
python ../mtah/evaluation/run_evaluation.py --gt labeling/data/gt_internal.csv --results "$R" --out "$C/final_out" --no-detail 2>&1 | sed -n '/^rank/,/^$/p'
python analysis/counts.py out/SyntaxEror.csv out/SyntaxEror_detector.csv
echo "=== pipeline done $(date) ==="
