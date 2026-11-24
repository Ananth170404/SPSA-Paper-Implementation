#!/usr/bin/env bash
# Reproduce every figure of the paper (Linux / macOS / Git Bash).
#   bash scripts/reproduce_all.sh           # full protocol
#   bash scripts/reproduce_all.sh --quick   # smoke test, a few minutes
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/check_setup.py
python scripts/download_mnist.py
python experiments/run_didactic.py "$@"
python experiments/run_mnist.py "$@"
python experiments/run_mnist.py "$@" --order holistic --skip-gd --name mnist_holistic
python experiments/run_mnist.py "$@" --no-split-bias --skip-gd --name mnist_no_split_bias
python experiments/run_mnist.py "$@" --hidden 196 --name mnist_155k
python experiments/run_mnist.py "$@" --hidden 196 --order holistic --skip-gd --name mnist_155k_holistic
python experiments/run_mnist.py "$@" --hidden 196 --no-split-bias --skip-gd --name mnist_155k_no_split_bias
python experiments/summarize.py --out results/SUMMARY.md
echo "Done. Figures and SUMMARY.md are in results/ (results/quick/ for --quick)."
