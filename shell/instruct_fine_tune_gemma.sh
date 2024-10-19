
devices="0"
seed=42

train_data={path to ERM4/Records/query_rewriter_plus_generated_data/train.json}
val_data={path to ERM4/Records/query_rewriter_plus_generated_data/eva.json}

base_model={path to gemma-2b}
output_dir={path to directory to save peft model}
# instruction_model={checkpoint if you want to continue fine-tuning}

for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in -1
        do
                mkdir -p $output_dir
                echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample"
                CUDA_VISIBLE_DEVICES=$devices python -u finetune_gemma_rewriter.py \
                    --base_model $base_model \
                    --train_data_path $train_data \
                    --val_data_path $val_data \
                    --output_dir ${output_dir}_${seed}_${sample} \
                    --batch_size 8 \
                    --micro_batch_size 4 \
                    --num_epochs 6 \
                    --learning_rate $lr \
                    --cutoff_len 400 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules 'q_proj,v_proj' \
                    --train_on_inputs True \
                    --group_by_length False \
                    --sample $sample \
                    --seed $seed \
                    # --resume_from_checkpoint $instruction_model \
        done
    done
done

