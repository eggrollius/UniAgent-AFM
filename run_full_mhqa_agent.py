#!/usr/bin/env python3
"""
run_full_mhqa_agent.py - Run MHQA agent on ENTIRE HotpotQA dataset
"""
import subprocess
import json
import os
from pathlib import Path
from datasets import load_dataset

def download_hotpotqa():
    """Download the full HotpotQA dataset"""
    print("Downloading full HotpotQA dataset...")
    try:
        dataset = load_dataset("hotpot_qa", "distractor")
        print(f"Downloaded: {len(dataset['train'])} training questions")
        print(f"Downloaded: {len(dataset['validation'])} validation questions")
        return dataset
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def run_mhqa_agent(question, output_dir, question_id):
    """Run MHQA agent with a question"""
    output_file = output_dir / f"mhqa_{question_id:06d}.json"
    
    cmd = [
        "python", "-m", "agent_systems.MHQA_agent.main",
        "--question", question,
        "--topk_sparse", "5",
        "--topk_dense", "5",
        "-o", str(output_file)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error with question {question_id}: {e}")
        return False

def main():
    # Create output directory
    output_dir = Path("data/raw/full_mhqa_trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running MHQA Agent on FULL HotpotQA Dataset")
    print("=" * 60)
    
    # Download dataset
    dataset = download_hotpotqa()
    if not dataset:
        return
    
    # Process training set
    train_data = dataset['train']
    print(f"\nProcessing {len(train_data)} training questions...")
    
    success_count = 0
    for i, item in enumerate(train_data):
        question = item['question']
        
        if i % 100 == 0:
            print(f"Progress: {i}/{len(train_data)} ({i/len(train_data)*100:.1f}%)")
        
        if run_mhqa_agent(question, output_dir, i):
            success_count += 1
    
    print(f"\nMHQA Agent completed!")
    print(f"Generated {success_count}/{len(train_data)} trajectories")
    print(f"Saved in: {output_dir}")
    
    # Convert to training dataset
    print("\nConverting to training dataset...")
    try:
        subprocess.run([
            "python3", "sft/trajectory_to_dataset.py", 
            str(output_dir)
        ], check=True)
        print("Training dataset created!")
    except subprocess.CalledProcessError as e:
        print(f"Error converting: {e}")

if __name__ == "__main__":
    main()
