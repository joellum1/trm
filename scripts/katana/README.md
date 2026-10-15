# Running on Katana (UNSW, PBS Pro)

Code is edited on the Mac and pushed with rsync; jobs run on Katana GPUs.

| Step | Command (from the Mac unless noted) |
|---|---|
| Push code | `scripts/katana/sync.sh` (`--data` also pushes `data/maze-custom*`) |
| Check before a new kind of run | `ssh katana 'cd ~/thesis/trm && qsub scripts/katana/smoke.pbs'` (exits non-zero on failure) |
| Fine-tune one seed | `ssh katana 'cd ~/thesis/trm && qsub -v SEED=0 scripts/katana/finetune.pbs'` |
| Gate runs on the smoke test | `qsub -W depend=afterok:<smoke job id> -v SEED=1 scripts/katana/finetune.pbs` |
| Status + latest eval per run | `scripts/katana/status.sh` |
| Resume after a walltime kill | re-submit the same `qsub -v SEED=...` line; it continues from `last.pt` |
| Unit tests (CPU, login node) | `ssh katana 'cd ~/thesis/trm && source scripts/katana/env.sh && python -m pytest -q tests'` |
| Catch-up HF upload (on Katana) | `bash scripts/katana/push_hf.sh <run_name>` |
| Re-score Maze-Hard baseline (E1/E5) | `qsub scripts/katana/rescore_baseline.pbs` → `results/maze_hard_baseline/` |
| Vocabulary expansion check (E3) | `qsub scripts/katana/transplant_check.pbs` → `results/diagnostics/` |
| Re-score first run (E2) | `qsub scripts/katana/rescore_first_run.pbs` → `results/first_run/` |
| Pull results + rebuild thesis outputs | `scripts/katana/pull_results.sh && python scripts/make_report.py` (on the Mac) |

Layout: `~/thesis/trm` holds code; `data/`, `checkpoints/`, `results/` are
symlinks to `/srv/scratch/$USER/trm/` (home is 15 GB). Scratch is not backed
up, so HF is the off-cluster copy.

One-time setup for HF uploads (write token from huggingface.co/settings/tokens):

    ssh katana
    cd ~/thesis/trm && source scripts/katana/env.sh
    hf auth login        # paste the token; say no to git credential
    hf auth whoami

The token is stored in `$HF_HOME/token` on scratch (private to you). Running
jobs retry HF at every checkpoint, so they start uploading once you log in.

Developing while runs are going: each fine-tune run copies `src/ scripts/
config/` into `<run>/code` on first start and always runs that copy, so
`sync.sh` only affects runs that have not started yet. To change a running
experiment, submit a new run (new `RUN`/`SEED`), don't edit its frozen code.

Per run (`checkpoints/Maze-custom-finetune/<run>/`): `job.log`, `step_N`
(EMA weights, every 250 steps), `last.pt` (resume state), `eval_metrics.jsonl`
(every 100 steps on the 1,000-maze set), `evaluator_MazePath_step_N/verdicts.jsonl`
(per-maze checker verdicts with reasons). W&B logs offline under
`/srv/scratch/$USER/trm/wandb`; `wandb sync` them if wanted.
