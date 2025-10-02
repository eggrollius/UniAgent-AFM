#!/usr/bin/env python3
"""
run_full_math_agent.py - Run Math agent on ENTIRE GSM8K dataset
"""
import subprocess
import json
from pathlib import Path
from datasets import load_dataset

def download_gsm8k():
    """Download the full GSM8K dataset"""
    print("📥 Downloading full GSM8K dataset...")
    try:
        dataset = load_dataset("gsm8k", "main")
        print(f"✅ Downloaded: {len(dataset['train'])} training problems")
        print(f"✅ Downloaded: {len(dataset['test'])} test problems")
        return dataset
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def run_math_agent(problem, output_dir, problem_id):
    """Run Math agent with a problem"""
    output_file = output_dir / f"math_{problem_id:06d}.json"
    
    # Extract the question part (before the answer)
    question = problem.split('\n\n')[0]
    
    cmd = [
        "python", "-m", "agent_systems.Math_agent.main",
        "--task", question,
        "-o", str(output_file)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error with problem {problem_id}: {e}")
        return False

def main():
    # Create output directory
    output_dir = Path("data/raw/full_math_trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Running Math Agent on FULL GSM8K Dataset")
    print("=" * 60)
    
    # Download dataset
    dataset = download_gsm8k()
    if not dataset:
        return
    
    # Process training set
    train_data = dataset['train']
    print(f"\n📊 Processing {len(train_data)} training problems...")
    
    success_count = 0
    for i, item in enumerate(train_data):
        problem = item['question']
        
        if i % 100 == 0:
            print(f"Progress: {i}/{len(train_data)} ({i/len(train_data)*100:.1f}%)")
        
        if run_math_agent(problem, output_dir, i):
            success_count += 1
    
    print(f"\n🎉 Math Agent completed!")
    print(f"📊 Generated {success_count}/{len(train_data)} trajectories")
    print(f"📁 Saved in: {output_dir}")
    
    # Convert to training dataset
    print("\n🔄 Converting to training dataset...")
    try:
        subprocess.run([
            "python3", "sft/trajectory_to_dataset.py", 
            str(output_dir)
        ], check=True)
        print("✅ Training dataset created!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting: {e}")

if __name__ == "__main__":
    main()
