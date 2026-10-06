#!/usr/bin/env bash
# Optional GPU training recipe; not part of the cached QA demo.
set -euo pipefail
ERM4_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ERM4_TRAIN_DATA:?Set ERM4_TRAIN_DATA to a local training JSON file}"
: "${ERM4_VAL_DATA:?Set ERM4_VAL_DATA to a local validation JSON file}"
: "${ERM4_BASE_MODEL:?Set ERM4_BASE_MODEL to an existing local Gemma-2B directory}"
: "${ERM4_OUTPUT_DIR:?Set ERM4_OUTPUT_DIR to a new checkpoint directory}"
[[ -f "$ERM4_TRAIN_DATA" && -f "$ERM4_VAL_DATA" && -d "$ERM4_BASE_MODEL" ]] || { echo 'Local data/model paths do not exist.' >&2; exit 1; }
[[ ! -e "$ERM4_OUTPUT_DIR" ]] || { echo 'Use a new output directory to preserve existing checkpoints.' >&2; exit 1; }
CUDA_VISIBLE_DEVICES="${ERM4_DEVICE:-0}" "${PYTHON:-python3}" -u "$ERM4_ROOT/experiments/finetune_gemma_rewriter.py" \
    --base_model "$ERM4_BASE_MODEL" --train_data_path "$ERM4_TRAIN_DATA" --val_data_path "$ERM4_VAL_DATA" \
    --output_dir "$ERM4_OUTPUT_DIR" --batch_size 8 --micro_batch_size 4 --num_epochs 6 \
    --learning_rate 1e-4 --cutoff_len 400 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
    --lora_target_modules 'q_proj,v_proj' --train_on_inputs True --group_by_length False \
    --sample -1 --seed 42 "$@"
