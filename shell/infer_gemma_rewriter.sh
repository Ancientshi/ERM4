
devices="0"
seed=42
base_model={path to gemma-2b}
model_path={path to peft model}

echo $model_path
CUDA_VISIBLE_DEVICES=$devices python infer_gemma_rewriter.py \
    --base_model $base_model \
    --lora_weights $model_path \
    --cutoff_len 400 \
    --seed $seed \
    --max_new_tokens 100 \
    --port 8001

