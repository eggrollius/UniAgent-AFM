import argparse
import os
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import SFTTrainer, SFTConfig

# -----------------------
# Args
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="Use a small Qwen to mirror your future larger Qwen runs.")
parser.add_argument("--train_jsonl", type=str, required=True,
                    help="Path to prompt-completion JSONL file.")
parser.add_argument("--out_dir", type=str, default="./qwen-sft-out",
                    help="Where to save checkpoints and final model.")
parser.add_argument("--epochs", type=float, default=1.0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_length", type=int, default=512,
                    help="Also used as packing sequence length if packing=True.")
parser.add_argument("--packing", action="store_true",
                    help="Enable example packing. Safe to leave off for smoke tests.")
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()

# -----------------------
# Repro
# -----------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# -----------------------
# Load tokenizer
# -----------------------
tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)

#if tok.pad_token is None:
    #tok.pad_token = tok.eos_token

#eos_str = tok.eos_token if isinstance(tok.eos_token, str) else "<|im_end|>"

# -----------------------
# Load model on CPU
# -----------------------
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    device_map=None,
    torch_dtype=torch.float32,
    trust_remote_code=True
)

# -----------------------
# Load dataset from JSONL
# -----------------------
if not os.path.exists(args.train_jsonl):
    raise FileNotFoundError(f"Training file not found: {args.train_jsonl}")

ds = load_dataset("json", data_files={"train": args.train_jsonl})["train"]
ds = ds.shuffle(seed=args.seed).select(range(min(64, len(ds))))

# -----------------------
# Configure SFT
# -----------------------
cfg = SFTConfig(
    output_dir=args.out_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="no",
    report_to=None,

    dataset_text_field="text",  # ignored for prompt-completion
    max_length=args.max_length,
    packing=args.packing,
    completion_only_loss=False,

    #eos_token=eos_str,
    pad_token=tok.pad_token,
    no_cuda=True,
    use_cpu=True,
    bf16=False,
    fp16=False,
    tf32=None,
)

# -----------------------
# Train
# -----------------------
trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=ds,
    processing_class=tok,
)

trainer.train()
trainer.save_model(args.out_dir)

# -----------------------
# Quick generation sanity check
# -----------------------
pipe = pipeline("text-generation", model=args.out_dir, tokenizer=tok, device_map=None)

prompt_struct = [{"role": "user", "content": "Write a Python function `def add(a,b):` returning the sum. Only a single Python code block."}]
out = pipe(prompt_struct)
print("\n=== SAMPLE GENERATION ===")
print(out[0]["generated_text"])
