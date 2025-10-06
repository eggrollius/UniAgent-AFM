import argparse
import json
import os
import time
import gzip
from typing import Dict, Any, Iterable, Iterator, Union
from dataclasses import asdict

from .agent import solve


def _open_text(path: str):
    """Open text file, supporting gzip if *.gz."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _is_probably_jsonl(path: str) -> bool:
    """Heuristics: extension or 'one-JSON-per-line' sniff."""
    low = path.lower()
    if low.endswith((".jsonl", ".ndjson")):
        return True
    # If extension is ambiguous (.json or no ext), we sniff first ~1KB
    # If we see a newline early and the first non-space char is '{',
    # assume JSONL; otherwise we'll try JSON array/object later.
    try:
        with _open_text(path) as f:
            sample = f.read(1024)
        first_non_ws = next((c for c in sample if not c.isspace()), "")
        if first_non_ws == "{":
            # Could be JSONL or single object; we’ll still treat JSONL first.
            return "\n{" in sample or "\n" in sample
    except Exception:
        pass
    return False


def read_records(path: str) -> Iterator[Dict[str, Any]]:
    """
    Robust dataset loader:
      - JSONL/NDJSON: one JSON object per line
      - JSON: either an array of objects or a single object { ... } (wrapped)
      - *.gz supported
    Yields dict rows with at least 'question' and (optionally) 'id','context'.
    """
    # Prefer JSONL if we are confident; fall back to JSON array/object.
    if _is_probably_jsonl(path):
        with _open_text(path) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Bad JSONL at line {ln}: {e}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"JSONL line {ln} is not an object")
                yield obj
        return

    # JSON array or single object
    with _open_text(path) as f:
        try:
            data: Union[Dict[str, Any], Iterable[Dict[str, Any]]] = json.load(f)
        except json.JSONDecodeError as e:
            # Helpful message if they passed JSONL with .json extension
            raise ValueError(
                f"Failed to parse JSON file {path}. "
                f"If your file is JSON Lines, rename to .jsonl and try again. "
                f"Underlying error: {e}"
            ) from e

    if isinstance(data, list):
        for i, ex in enumerate(data, 1):
            if not isinstance(ex, dict):
                raise ValueError(f"JSON array element {i} is not an object")
            yield ex
    elif isinstance(data, dict):
        # Single object: emit as one record (wrap)
        yield data
    else:
        raise ValueError("Top-level JSON must be an object or array of objects")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--task", default="LegalQA")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as fout:
        for ex in read_records(args.dataset):
            q = (ex.get("question") or ex.get("query") or "").strip()
            if not q:
                # Skip empty questions but keep going
                continue
            steps, final = solve(q, model=args.model, max_steps=args.max_steps)
            rec = {
                "id": ex.get("id"),
                "task": args.task,
                "question": q,
                "context": ex.get("context", ""),
                "steps": [asdict(s) for s in steps],  # dataclasses.asdict fix
                "answer": final,
                "meta": {
                    "model": args.model,
                    "max_steps": args.max_steps,
                    "afm_version": "0.1",
                    "ts": int(time.time()),
                },
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
