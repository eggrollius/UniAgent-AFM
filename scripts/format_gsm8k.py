import json, argparse, os

def format_file(in_path: str, out_path: str, limit: int = 200):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        first = f_in.read(1); f_in.seek(0)
        if first == "[":  # JSON array
            data = json.load(f_in)
            for i, ex in enumerate(data):
                if n >= limit: break
                q = (ex.get("question") or ex.get("question_text") or ex.get("problem") or "").strip()
                if not q: continue
                rid = ex.get("id") or f"gsm8k_{i:05d}"
                f_out.write(json.dumps({"id": rid, "question": q}, ensure_ascii=False) + "\n")
                n += 1
        else:  # JSONL
            for i, line in enumerate(f_in):
                if n >= limit: break
                if not line.strip(): continue
                ex = json.loads(line)
                q = (ex.get("question") or ex.get("question_text") or ex.get("problem") or "").strip()
                if not q: continue
                rid = ex.get("id") or f"gsm8k_{i:05d}"
                f_out.write(json.dumps({"id": rid, "question": q}, ensure_ascii=False) + "\n")
                n += 1
    print(f"[OK] wrote {n} items -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Path to GSM8K json or jsonl")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL with {id,question}")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    format_file(args.in_json, args.out_jsonl, args.limit)