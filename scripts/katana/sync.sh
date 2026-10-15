#!/usr/bin/env bash
# Mac: rsync code (and with --data, data/maze-custom*) to Katana. Does not touch git.
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -az --delete \
  --exclude .git --exclude .venv --exclude data --exclude checkpoints --exclude results \
  --exclude wandb --exclude outputs --exclude '__pycache__' --exclude '.DS_Store' --exclude '*.egg-info' \
  ./ katana:thesis/trm/
if [[ "${1:-}" == "--data" ]]; then
  rsync -az data/maze-custom data/maze-custom-eval1k data/maze-custom-raw katana:/srv/scratch/z5481782/trm/data/
fi
echo "synced -> katana:~/thesis/trm"
