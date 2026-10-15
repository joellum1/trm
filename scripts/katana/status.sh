#!/usr/bin/env bash
# Mac: queue state + latest eval of each fine-tune run on Katana.
ssh katana 'qstat -u $USER 2>/dev/null | tail -n +4
cd /srv/scratch/$USER/trm/checkpoints/Maze-custom-finetune 2>/dev/null || exit 0
for run in */; do
  run=${run%/}; f=$run/eval_metrics.jsonl
  [ -f $run/DONE ] && st=done || st=running
  [ -s $f ] || { echo "$run ($st): no evals yet"; continue; }
  tail -1 $f | python3 -c "import sys,json; j=json.load(sys.stdin); g=lambda k: j.get(k,float(\"nan\")); print(\"%s step %d: exact %.3f | shortest %.3f | longer %.3f | broken %.3f | all C reached %.3f\" % (sys.argv[1], j[\"step\"], g(\"all/exact_accuracy\"), g(\"path/verdict/valid_shortest\"), g(\"path/verdict/valid_longer\"), g(\"path/verdict/broken\"), g(\"path/reaches_all_checkpoints\")))" "$run ($st)"
print(f\"$run ($st) step {j[\"step\"]}: exact {g(\"all/exact_accuracy\"):.3f} | shortest {g(\"path/verdict/valid_shortest\"):.3f} longer {g(\"path/verdict/valid_longer\"):.3f} broken {g(\"path/verdict/broken\"):.3f}\")"
done'
