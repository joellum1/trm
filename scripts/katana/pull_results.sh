#!/usr/bin/env bash
# Mac: copy experiment outputs (no weights) from Katana into results/.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=katana:/srv/scratch/z5481782/trm

for d in maze_hard_baseline first_run diagnostics; do
  rsync -az --exclude '*.pt' --exclude '*_all_preds.*' "$R/results/$d" results/ 2>/dev/null \
    && echo "pulled results/$d" || echo "not on Katana yet: results/$d"
done

# Per-eval verdict dirs are git-ignored; the last one is kept as final_verdicts.jsonl.
mkdir -p results/finetune
rsync -az --prune-empty-dirs \
  --include '*/' --include 'all_config.yaml' --include 'eval_metrics.jsonl' \
  --include 'training_metrics.csv' --include 'DONE' --include 'job.log' \
  --include 'evaluator_MazePath_step_*/verdicts.jsonl' \
  --exclude 'code/***' --exclude '*' \
  "$R/checkpoints/Maze-custom-finetune/" results/finetune/ 2>/dev/null || true
for run in results/finetune/*/; do
  run=${run%/}
  last=$(ls -d "$run"/evaluator_MazePath_step_* 2>/dev/null | sed 's/.*_step_//' | sort -n | tail -1)
  [ -n "$last" ] && cp "$run/evaluator_MazePath_step_$last/verdicts.jsonl" "$run/final_verdicts.jsonl"
  echo "pulled $run (last eval: step ${last:-none}$([ -f $run/DONE ] && echo ', finished'))"
done
