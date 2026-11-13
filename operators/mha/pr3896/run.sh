#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
: "${ASCEND_HOME_PATH:?Source CANN first}"
source operators/mha/env_recovered.sh
python operators/spec_v1/prepare_mha.py
MODE="${1:-smoke}"
case "$MODE" in smoke|all|tails-smoke|tails-all) ;; *) echo 'Usage: run.sh [smoke|all|tails-smoke|tails-all]'; exit 2;; esac
EXTRA=()
if [[ "$MODE" == tails-* ]]; then EXTRA=(--tails); fi
OUT="results/mha-pr3896/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
exec > >(tee "$OUT/runner.log") 2>&1
echo "Results: $OUT"
python - "$OUT" "$MODE" <<'PY'
import json, sys
from pathlib import Path
root, mode = Path(sys.argv[1]), sys.argv[2]
lengths = [1,63,64,65,127,129,193,255,3583,3585,4095,4097] if mode.startswith('tails-') else [4096,3584]
tags = ['smoke']
if mode.endswith('all'):
    tags += [f'check-{s}' for s in [42,43,44]]
    tags += [f'bench-{m}-rep{r}' for r in [1,2,3] for m in ['per_call']]
(root/'plan.json').write_text(json.dumps(dict(mode=mode, lengths=lengths, tags=tags), indent=2))
PY
git rev-parse HEAD > "$OUT/revision.txt"
python -m pip freeze > "$OUT/pip-freeze.txt"
npu-smi info > "$OUT/npu-smi.txt"
run_case() {
  local tag="$1"; shift
  python operators/mha/pr3896/bench_pr.py --suite mha_bf16 --output "$OUT/$tag" "${EXTRA[@]}" "$@"
}
run_case smoke --check-only
if [[ "$MODE" == all || "$MODE" == tails-all ]]; then
  for seed in 42 43 44; do
    run_case "check-$seed" --seed "$seed" --check-only
  done
  for rep in 1 2 3; do
    for mode in per_call; do
      run_case "bench-$mode-rep$rep" --ac-metadata "$mode" --warmup 100 --iters 100 --rounds 10
    done
  done
fi
python operators/mha/pr3896/summarize.py --input "$OUT" | tee "$OUT/summary.txt"
echo "Completed: $OUT/summary.txt"
