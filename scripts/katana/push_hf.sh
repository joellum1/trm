#!/bin/bash
# Katana: catch-up HF upload of a run's checkpoints + metrics (existing files are skipped).
#   bash scripts/katana/push_hf.sh <run_name> [repo]
set -euo pipefail
cd $HOME/thesis/trm
source scripts/katana/env.sh
RUN=$1
REPO=${2:-joellum1/trm-maze-finetune}
python - "$RUN" "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
run, repo = sys.argv[1:]
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path=f"checkpoints/Maze-custom-finetune/{run}", path_in_repo=run,
                  allow_patterns=["step_*", "all_config.yaml", "eval_metrics.jsonl", "training_metrics.csv"],
                  ignore_patterns=["*_all_preds.*"], commit_message=f"{run}: catch-up upload")
print(f"uploaded -> https://huggingface.co/{repo}/tree/main/{run}")
PY
