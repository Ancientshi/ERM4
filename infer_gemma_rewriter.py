from flask import Flask, request, jsonify
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, HfArgumentParser
from peft import PeftModel
from datasets import load_dataset
import json
import numpy as np
from tqdm import tqdm

app = Flask(__name__)

@dataclass
class ScriptArguments:
    cutoff_len: Optional[int] = field(default=1024)
    base_model: Optional[str] = field(default="gpt2", metadata={"help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."})
    lora_weights: Optional[str] = field(default=None, metadata={"help": "The path to the LoRA weights file."})
    seed: Optional[int] = field(default=42, metadata={"help": "The seed to use for reproducibility."})
    max_new_tokens: Optional[int] = field(default=1000, metadata={"help": "The maximum number of tokens to generate."})
    port: Optional[int] = field(default=4545, metadata={"help": "The port to run the server on."})

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]


quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=False,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    args.base_model, 
    quantization_config=quantization_config, 
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation='flash_attention_2'
)

if args.lora_weights!=None:
    model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                torch_dtype=torch.float16,
                device_map="auto"
            )
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.base_model,model_max_length=args.cutoff_len)


def generate_prompt(data_point):
    original_question = data_point["original_question"]
    text=f"""Below is an instruction that describes a task, paired with further context. Write a response that appropriately completes the request.  # noqa: E501
        
### Instruction:
Your task is to transform Original Question, often colloquial, jargon-heavy, or ambiguous question into several semantically augmented, intent clear questions. Additionally, you need to generate a series of concise queries focusing on different semantic aspects that contributes to finding relevant informations through search engine for answering the question. 

### Original Question:
{original_question}

### Answer:
"""
    return text


def generate_answer(data_point):
    text=generate_prompt(data_point)
    temperature=data_point["temperature"]
    if temperature==0:
        temperature=0.01

    device = "cuda"
    top_p=0.95
    top_k=60
    repetition_penalty=1.0
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True
    )

    model_inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=True).to(device)
    model_outputs = model.generate(**model_inputs, generation_config=generation_config,max_new_tokens=args.max_new_tokens)
    output_text = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    output_text = output_text.split("### Answer:\n")[-1]
    return output_text

@app.route('/build', methods=['POST'])
def build():
    data = request.json
    output_text = generate_answer(data)
    return jsonify({'output': output_text})

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=args.port)
