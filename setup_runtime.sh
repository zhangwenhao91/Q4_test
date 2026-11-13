#!/usr/bin/env bash
# Reuse installed native runtime assets, never old Q4 benchmark code or results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OLD="${1:?Usage: bash setup_runtime.sh /absolute/path/to/existing/Q4}"
OLD="$(cd "$OLD" && pwd)"
mkdir -p third_party
for name in ops-transformer-q4fa q4fa-opp; do
  test -d "$OLD/third_party/$name" || { echo "Missing runtime: $name"; exit 1; }
  if [[ -e "third_party/$name" || -L "third_party/$name" ]]; then
    [[ "$(readlink -f "third_party/$name")" == "$(readlink -f "$OLD/third_party/$name")" ]] || {
      echo "Existing target differs: third_party/$name; preserve it before changing setup"; exit 1;
    }
  else
    ln -s "$OLD/third_party/$name" "third_party/$name"
  fi
done
python operators/spec_v1/prepare_mha.py
echo 'Runtime linked. Keep the existing Q4 runtime directories while using these links.'
