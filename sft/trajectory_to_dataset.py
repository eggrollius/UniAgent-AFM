# Create a new file: sft/mhqa_trajectory_to_dataset.py
import json
import sys
from pathlib import Path

def parse_mhqa_trajectory(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_rows = []
    
    # Extract the question from task_id
    question = data.get("task_id", "")
    
    # Process each step in the trajectory
    for step in data.get("steps", []):
        if step.get("role") == "assistant":
            content = step.get("content", "")
            phase = step.get("phase", "")
            
            # Create a training example for each step
            if content.strip():
                prompt = f"Question: {question}\n\nContext: {content}"
                completion = f"Phase: {phase}\nContent: {content}"
                
                dataset_rows.append({
                    "prompt": prompt,
                    "completion": completion,
                    "label": "true"
                })
    
    return dataset_rows

def main():
    if len(sys.argv) < 2:
        print("Usage: python mhqa_trajectory_to_dataset.py <folder_with_json_files>")
        sys.exit(1)

    root_path = Path(sys.argv[1])
    if not root_path.is_dir():
        print(f"❌ {root_path} is not a directory")
        sys.exit(1)

    all_rows = []
    for traj_file in root_path.rglob("*.json"):
        try:
            rows = parse_mhqa_trajectory(traj_file)
            all_rows.extend(rows)
            print(f"✓ Parsed {traj_file} ({len(rows)} rows)")
        except Exception as e:
            print(f"⚠️ Skipped {traj_file}: {e}")

    if not all_rows:
        print("⚠️ No .json files found, nothing written.")
        sys.exit(1)

    # Write to a single output file
    output_path = root_path / "mhqa_training_dataset.jsonl"
    with open(output_path, "w", encoding="utf-8") as out_f:
        for row in all_rows:
            out_f.write(json.dumps(row) + "\n")

    print(f"✅ Dataset written to: {output_path.resolve()} (total rows: {len(all_rows)})")

if __name__ == "__main__":
    main()
