import json
import sys
from pathlib import Path

def parse_trajectory(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    TARGET_TOOL = "cd"

    dataset_rows = []  
    for step in data["trajectory"]:
        # query is the previus context that llm has access to in this turn.
        query = step["query"]

        thought = step["thought"]
        action = step["action"] # this is a tool call
        observation = step["observation"] # tool result
        response = step["response"]

        # if step has tool call, and it matches
        if action != "" and action.startswith(TARGET_TOOL):
            prompt = normalize_query(query)
            completion = [{
                "thought": thought,
                "action": action,
                # we dont include observation since that would cause tool reuslt hallucination.
                # but do we include response?
                # likley not.
                #"response": response
            }]

            new_row = {
                "prompt": prompt,
                "completion": completion,
                "label": "true"
            }

            dataset_rows.append(normalize_query(new_row))

    return dataset_rows

def normalize_query(query):
    def _to_text(x):
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get("text") or x.get("content") or str(x)
        if isinstance(x, list):
            return "".join(_to_text(i) for i in x)
        return str(x) if x is not None else ""
    
    for msg in query:
        if isinstance(msg, dict) and "content" in msg:
            msg["content"] = _to_text(msg["content"])
    return query

def main():
    if len(sys.argv) < 2:
        print("Usage: python trajectories_to_dataset.py <folder_of_folders_with_traj>")
        sys.exit(1)

    root_path = Path(sys.argv[1])
    if not root_path.is_dir():
        print(f"❌ {root_path} is not a directory")
        sys.exit(1)

    all_rows = []
    for traj_file in root_path.rglob("*.traj"):
        try:
            rows = parse_trajectory(traj_file)
            all_rows.extend(rows)
            print(f"✓ Parsed {traj_file} ({len(rows)} rows)")
        except Exception as e:
            print(f"⚠️ Skipped {traj_file}: {e}")

    if not all_rows:
        print("⚠️ No .traj files found, nothing written.")
        sys.exit(1)

    # Write to a single output file in root folder
    output_path = root_path.with_suffix(".jsonl")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for row in all_rows:
            out_f.write(json.dumps(row) + "\n")

    print(f"✅ Dataset written to: {output_path.resolve()} (total rows: {len(all_rows)})")


if __name__ == "__main__":
    main()
