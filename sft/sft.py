import argparse
import os
import random
import json
import numpy as np
import torch

from datasets import load_dataset, Dataset
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
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

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
#   Your rows look like:
#   {"prompt":[...], "completion":[...]}
#   We adapt them to a single "text" field containing:
#     <user>...</user>\n<assistant_call>{"name":..., "arguments":{...}}</assistant_call>
#   (We only train the call here; final-answer training can be added similarly.)
# -----------------------
if not os.path.exists(args.train_jsonl):
    raise FileNotFoundError(f"Training file not found: {args.train_jsonl}")

raw = load_dataset("json", data_files={"train": args.train_jsonl})["train"]

def row_to_samples(row):
    prompt = row.get("prompt", [])
    completion = row.get("completion", [])

    # last user message content from prompt
    user_msg = next((m.get("content") for m in reversed(prompt) if m.get("role") == "user"), None)

    # try to pull a tool call (prefer completion.tool_calls, else assistant.tool_calls in prompt)
    assistant_call = None

    # 1) from completion
    if completion and isinstance(completion, list):
        tc_list = completion[0].get("tool_calls") or []
        if tc_list:
            fn = tc_list[0]["function"]["name"]
            args_ = tc_list[0]["function"]["arguments"]
            try:
                args_obj = json.loads(args_) if isinstance(args_, str) else args_
            except Exception:
                args_obj = args_
            assistant_call = json.dumps({"name": fn, "arguments": args_obj})

    # 2) fallback: from prompt assistant message
    if assistant_call is None:
        for m in prompt:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                fn = m["tool_calls"][0]["function"]["name"]
                args_ = m["tool_calls"][0]["function"]["arguments"]
                try:
                    args_obj = json.loads(args_) if isinstance(args_, str) else args_
                except Exception:
                    args_obj = args_
                assistant_call = json.dumps({"name": fn, "arguments": args_obj})
                break

    samples = []
    if user_msg and assistant_call:
        samples.append({
            "text": f"<user>{user_msg}</user>\n<assistant_call>{assistant_call}</assistant_call>"
        })
    return samples

# flat = []
# for r in raw:
#     flat.extend(row_to_samples(r))

# if not flat:
#     raise ValueError("No training samples were produced from the dataset. Check your schema mapping.")

ds = Dataset.from_list(raw)
ds = ds.shuffle(seed=args.seed).select(range(min(200, len(ds))))

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
    report_to=["tensorboard"],

    dataset_text_field="text",
    max_length=args.max_length,
    packing=args.packing,
    completion_only_loss=False,

    # for cpu args
    bf16=False,
    fp16=False,
    tf32=False,
)

# -----------------------
# Train
# -----------------------
trainer = SFTTrainer(
    model=model,
    processing_class=tok,
    args=cfg,
    train_dataset=ds,
)

trainer.train()
trainer.save_model(args.out_dir)

# -----------------------
# Quick generation sanity check
# -----------------------
pipe = pipeline("text-generation", model=args.out_dir, tokenizer=tok, device_map=None)

query = "I've uploaded a repo. Run a bash tool to list /testbed."
prompt_str = f"<user>{query}</user>\n<assistant_call>"
out = pipe(prompt_str, max_new_tokens=120, do_sample=False)[0]["generated_text"]
print("\n=== SAMPLE GENERATION ===")
print(out)
