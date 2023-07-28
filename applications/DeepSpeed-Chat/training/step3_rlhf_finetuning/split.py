"""
LoRA weights on top of a base model.
Usage:
python merge_lora.py
--base {Base model name or path}
--target {Output path}
--lora {LoRA path}
"""
import argparse
import math
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def split(base_model_path, target_model_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    base_tokenizer.pad_token = base_tokenizer.eos_token

    print("Applying the LoRA")
    print(f"Saving the target model to {target_model_path}")
    model.save_pretrained(target_model_path,max_shard_size="1024MB")
    base_tokenizer.save_pretrained(target_model_path)


#if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--base-model-path", type=str, required=True)
parser.add_argument("--target-model-path", type=str, required=True)

args = parser.parse_args()

split(args.base_model_path, args.target_model_path)
