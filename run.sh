#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
case "${1:-smoke}" in
  smoke)
    bash operators/mha/pr3896/run.sh smoke
    bash operators/mha/pr3896/run.sh tails-smoke
    ;;
  all)
    bash operators/mha/pr3896/run.sh all
    bash operators/mha/pr3896/run.sh tails-all
    ;;
  *) echo 'Usage: bash run.sh [smoke|all]'; exit 2;;
esac
