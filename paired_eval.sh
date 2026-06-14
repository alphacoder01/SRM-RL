#!/bin/bash
# Paired McNemar comparison of two eval-only runs (RL-tuned vs baseline).
# Each run must use rl.eval.dump_samples=true with the same num_samples/num_fill.
# Usage: bash paired_eval.sh <run_a_dir_or_jsonl> <run_b_dir_or_jsonl>
python src/rl/paired_eval.py "$@"
