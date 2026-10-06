#!/usr/bin/env bash
# Optional local Gemma service; main.py does not call it.
set -euo pipefail
ERM4_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ERM4_BASE_MODEL:?Set ERM4_BASE_MODEL to an existing local Gemma-2B directory}"
: "${ERM4_LORA_WEIGHTS:?Set ERM4_LORA_WEIGHTS to an existing local adapter directory}"
[[ -d "$ERM4_BASE_MODEL" && -d "$ERM4_LORA_WEIGHTS" ]] || { echo 'Local model/adapter paths do not exist.' >&2; exit 1; }
CUDA_VISIBLE_DEVICES="${ERM4_DEVICE:-0}" "${PYTHON:-python3}" "$ERM4_ROOT/experiments/infer_gemma_rewriter.py" \
    --base_model "$ERM4_BASE_MODEL" --lora_weights "$ERM4_LORA_WEIGHTS" \
    --cutoff_len 400 --seed 42 --max_new_tokens 100 --port "${ERM4_PORT:-8001}" "$@"
