#!/usr/bin/env python3
"""
evaluate_all_models.py

Usage:
  python evaluate_all_models.py /path/to/models_folder

- Scans the given folder for subdirectories (each assumed to be a model).
- Runs lm_eval on HumanEval for each model dir.
- Saves outputs/logs under results/humaneval/<timestamp>/<model_name>/
- Writes the exact lm_eval command used to command.txt.
- Creates a gzip tarball of the entire results root at the end.
"""

import argparse
import os
import shlex
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

def run_one_model(model_dir: Path, out_dir: Path) -> int:
    """
    Run lm_eval on HumanEval for a single local model directory.
    Saves:
      - command.txt  (the exact command used)
      - stdout.log   (captured stdout)
      - stderr.log   (captured stderr)
    Returns the process exit code.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd_file   = out_dir / "command.txt"
    stdout_log = out_dir / "stdout.log"
    stderr_log = out_dir / "stderr.log"
    results_json = out_dir / "results_humaneval.json"

    # lm_eval command
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={str(model_dir)},dtype=bfloat16,trust_remote_code=True",
        "--tasks", "humaneval",
        "--device", "cuda:0",
        "--batch_size", "1",
        "--gen_kwargs", "temperature=0.5,max_gen_toks=2048",
        "--log_samples",
        "--output_path", str(results_json),
        "--confirm_run_unsafe_code",
    ]

    # Save command
    cmd_file.write_text(" ".join(shlex.quote(p) for p in cmd) + "\n", encoding="utf-8")

    # Run and capture
    with stdout_log.open("w", encoding="utf-8") as so, stderr_log.open("w", encoding="utf-8") as se:
        proc = subprocess.run(cmd, stdout=so, stderr=se, text=True)

    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Evaluate all models in a folder on HumanEval and tar results.")
    ap.add_argument("models_folder", help="Folder containing model subdirectories.")
    args = ap.parse_args()

    models_root = Path(args.models_folder).resolve()
    if not models_root.exists() or not models_root.is_dir():
        print(f"[error] '{models_root}' is not a directory", file=sys.stderr)
        sys.exit(2)

    # Timestamped results root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = Path("results") / "humaneval" / timestamp
    results_root.mkdir(parents=True, exist_ok=True)
    print(f"[*] Results root: {results_root}")

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # Discover model subdirectories
    model_dirs = sorted([p for p in models_root.iterdir() if p.is_dir()])
    if not model_dirs:
        print(f"[warn] No model subdirectories found in {models_root}")
        sys.exit(0)

    # Helpful env for code-eval tasks
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # Evaluate each model
    failures = []
    for d in model_dirs:
        model_name = d.name
        out_dir = results_root / model_name
        print(f"[*] Evaluating: {d} -> {out_dir}")
        rc = run_one_model(d, out_dir)
        status = "OK" if rc == 0 else f"EXIT {rc}"
        print(f"[+] Done: {model_name} ({status})")
        if rc != 0:
            failures.append((model_name, rc))

    # Tar up entire results folder
    tar_path = Path.cwd() / f"results_humaneval_{timestamp}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(results_root, arcname=str(results_root.relative_to(results_root.parent.parent)))  # keep results/humaneval/<ts> structure

    print(f"[ok] Archived results to: {tar_path}")

    if failures:
        print("[warn] Some evaluations failed:")
        for name, code in failures:
            print(f"  - {name}: exit {code}")
        sys.exit(1)

if __name__ == "__main__":
    main()
