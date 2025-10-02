#!/usr/bin/env python3
"""
run_full_swe_agent.py - Run SWE agent on ENTIRE SWE-bench dataset
"""
import subprocess
import json
from pathlib import Path
from datasets import load_dataset

def download_swe_bench():
    """Download the full SWE-bench dataset"""
    print("📥 Downloading full SWE-bench dataset...")
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite")
        print(f"✅ Downloaded: {len(dataset['test'])} test problems")
        return dataset
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def run_swe_agent(problem, output_dir, problem_id):
    """Run SWE agent with a problem"""
    output_file = output_dir / f"swe_{problem_id:06d}.json"
    
    # Extract the problem statement
    problem_text = problem['problem_statement'][:200]  # First 200 chars
    
    cmd = [
        "python", "-m", "agent_systems.SWE_agent.main",
        "--task", problem_text,
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
    output_dir = Path("data/raw/full_swe_trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Running SWE Agent on FULL SWE-bench Dataset")
    print("=" * 60)
    
    # Download dataset
    dataset = download_swe_bench()
    if not dataset:
        return
    
    # Process test set
    test_data = dataset['test']
    print(f"\n📊 Processing {len(test_data)} test problems...")
    
    success_count = 0
    for i, item in enumerate(test_data):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(test_data)} ({i/len(test_data)*100:.1f}%)")
        
        if run_swe_agent(item, output_dir, i):
            success_count += 1
    
    print(f"\n🎉 SWE Agent completed!")
    print(f"📊 Generated {success_count}/{len(test_data)} trajectories")
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
