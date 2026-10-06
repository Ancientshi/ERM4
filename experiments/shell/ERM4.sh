#!/usr/bin/env bash
set -euo pipefail
ERM4_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON:-python3}" "$ERM4_ROOT/experiments/cached_qa.py" --root_path "$ERM4_ROOT" --dataset 2wikimqa --exp_name ERM4 "$@"
