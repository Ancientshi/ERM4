import os
import sys
from typing import List

import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback
from safetensors.torch import save_file, load_file

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel,AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from sklearn.metrics import roc_auc_score

def generate_prompt(data_point):
    original_question = data_point["original_question"]
    rewritten_question_list = data_point["rewritten_question"]
    query_list = data_point["query"]
    rewritten_question_text='**'.join(rewritten_question_list)
    query_text='**'.join(query_list)
    answer=f'''Rewritten Question:\n{rewritten_question_text}\n\nQuery:\n{query_text}'''
    text=f"""Below is an instruction that describes a task, paired with further context. Write a response that appropriately completes the request.  # noqa: E501
        
### Instruction:
Your task is to transform Original Question, often colloquial, jargon-heavy, or ambiguous question into several semantically augmented, intent clear questions. Additionally, you need to generate a series of concise queries focusing on different semantic aspects that contributes to finding relevant informations through search engine for answering the question. 

### Original Question:
{original_question}

### Answer:
{answer}"""
    return text




    
def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_data_path: str = "",
    val_data_path: str = "",
    output_dir: str = "./lora-alpaca",
    sample: int = -1,
    seed: int = 0,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4090,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = 'q_proj,v_proj',
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter

):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    
    
    
    def compute_metrics(eval_preds):
        '''
        compute_metrics函数的参数eval_preds是一个包含两个元素的元组(tuple)，其中：

        第一个元素是模型的预测结果，这些预测结果可能是直接的输出，也可能是经过preprocess_logits_for_metrics处理后的结果，取决于是否定义了该处理函数。
        第二个元素是评估数据集（eval_dataset）的标签。
        '''
        
        pre, labels = eval_preds
        auc = roc_auc_score(pre[1], pre[0])
        return {'auc': auc}

        
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "answer": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt



    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size



    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    os.environ["WANDB_DISABLED"] = "true"


    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    # )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = prepare_model_for_kbit_training(model)


    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)



    tokenizer = AutoTokenizer.from_pretrained(base_model)
    

    if train_data_path.endswith(".json"):  # todo: support jsonl
        train_data = load_dataset("json", data_files=train_data_path)
    else:
        train_data = load_dataset(train_data_path)
    
    if val_data_path.endswith(".json"):  # todo: support jsonl
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)
        
        
    
    # train_data = train_data.shuffle(seed=42)[:sample] if sample > -1 else train_data
    # print(len(train_data))
    # if resume_from_checkpoint:
    #     # Check the available weights and load them
    #     checkpoint_name = os.path.join(
    #         resume_from_checkpoint, "pytorch_model.bin"
    #     )  # Full checkpoint
    #     if not os.path.exists(checkpoint_name):
    #         checkpoint_name = os.path.join(
    #             resume_from_checkpoint, "adapter_model.bin"
    #         )  # only LoRA model - LoRA config above has to fit
    #         resume_from_checkpoint = (
    #             False  # So the trainer won't try loading its state
    #         )
    #     # The two files above have a different name depending on how they were saved, but are actually the same.
    #     if os.path.exists(checkpoint_name):
    #         print(f"Restarting from {checkpoint_name}")
    #         adapters_weights = torch.load(checkpoint_name)
    #         model = set_peft_model_state_dict(model, adapters_weights)
    #     else:
    #         print(f"Checkpoint {checkpoint_name} not found")

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_model.safetensors"
        )  
            
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            
            adapters_weights = load_file(checkpoint_name, device='cuda')
            set_peft_model_state_dict(model, adapters_weights)

        else:
            print(f"Checkpoint {checkpoint_name} not found")
            

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.


    train_data["train"] = train_data["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data["train"].shuffle(seed=seed)
    train_data["train"] = train_data["train"].shuffle(seed=seed)
    train_data = (train_data["train"].map(generate_and_tokenize_prompt))
    val_data = (val_data["train"].map(generate_and_tokenize_prompt))

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True



    
    if sample > -1:
        if sample <= 128 :
            eval_step = 10
        else:
            eval_step = len(train_data) // batch_size // 2
    else:
        #every epoch
        eval_step = len(train_data) // batch_size // 2 // 2

    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_accumulation_steps=1,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_step,
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None
        ),
        # padding (bool, str or PaddingStrategy, optional, defaults to True) — Select a strategy to pad the returned sequences (according to the model’s padding side and padding index) among:
        # True or 'longest' (default): Pad to the longest sequence in the batch (or no padding if only a single sequence is provided).
        # 'max_length': Pad to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided.
        # False or 'do_not_pad': No padding (i.e., can output a batch with sequences of different lengths).
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)


    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


    model.save_pretrained(output_dir)

    print(
        "\n Finish Training! :)"
    )





if __name__ == "__main__":
    fire.Fire(train)
