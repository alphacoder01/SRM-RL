#!/bin/bash
# Visualize an RL run's metrics.jsonl, saving plots into the run folder.
# Usage: bash plot_metrics.sh outputs_rl/ms1000_28/run3 [--smooth N] [--out DIR]
# Run as a plain script (not -m) so plotting needs only matplotlib + numpy,
# not the full src package.
python src/rl/plot_metrics.py "$@"
