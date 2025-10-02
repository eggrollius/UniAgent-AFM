import os, json, argparse, uuid
from .agent import solve
from .afm_schema import AFMTrajectory

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="JSONL with {id, question}")
    ap.add_argument("--out", required=True, help="Output AFM JSONL path")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--task", default="MathQA")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for ex in read_jsonl(args.dataset):
            qid = ex.get("id") or f"math-{uuid.uuid4().hex[:8]}"
            steps, final = solve(ex["question"], model=args.model, max_steps=args.max_steps)
            traj = AFMTrajectory(
                id=qid, task=args.task, question=ex["question"], context="",
                steps=steps, answer=str(final),
                meta={"model": args.model, "max_steps": args.max_steps}
            )
            fout.write(traj.to_json() + "\n")
            n += 1
    print(f"[OK] Wrote {n} AFM trajectories -> {args.out}")

if __name__ == "__main__":
    main()