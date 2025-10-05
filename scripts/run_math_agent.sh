#!/usr/bin/env bash
set -euo pipefail
python -m agent_systems.math_agent.main \
  --dataset data/math/gsm8k_sample.jsonl \
  --out trajectories/math/gsm8k_sample.afm.jsonl \
  --model gpt-4o-mini \
  --max-steps 8 \
  --task MathQA