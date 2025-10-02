# scripts/validate_afm.py
import argparse, json

REQUIRED_KEYS = {"id", "task", "question", "steps", "answer", "meta"}
REQUIRED_STEP_KEYS = {"function", "thought", "tool_call", "tool_result"}

def validate_line(obj, line_no):
    missing = REQUIRED_KEYS - set(obj.keys())
    if missing:
        return f"Line {line_no}: missing keys {missing}"
    if not isinstance(obj["steps"], list):
        return f"Line {line_no}: steps is not a list"
    for i, step in enumerate(obj["steps"], 1):
        if not isinstance(step, dict):
            return f"Line {line_no}, step {i}: not a dict"
        m2 = REQUIRED_STEP_KEYS - set(step.keys())
        if m2:
            return f"Line {line_no}, step {i}: missing {m2}"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("afm_file", help="Path to AFM JSONL file")
    args = ap.parse_args()

    errors = 0
    with open(args.afm_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"Line {i}: JSON parse error {e}")
                errors += 1
                continue
            msg = validate_line(obj, i)
            if msg:
                print(msg)
                errors += 1
    if errors == 0:
        print(f"[OK] {args.afm_file} passed schema validation")
    else:
        print(f"[FAIL] {errors} issue(s) in {args.afm_file}")

if __name__ == "__main__":
    main()
