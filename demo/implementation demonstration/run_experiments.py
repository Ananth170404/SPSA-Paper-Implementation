"""
Master Runner Script: Complete SPSA Demonstration (Part 1 & Part 2)
==================================================================
Runs both:
  1. Part 1: Mathematical Function Optimization (Complex Cubic Polynomial: Holistic SPSA, Layer-Wise SPSA, GD, GD+Momentum)
  2. Part 2: Neural Network Classification (Holistic SPSA vs Layer-Wise SPSA vs SGD vs Adam)

All output plots are saved to the 'figures/' directory.
"""

import os
import sys
import time

# Ensure current directory is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from part1_spsa_math import run_math_benchmarks
from part2_spsa_nn import run_nn_benchmarks


def main():
    print("#" * 80)
    print(" SPSA COMPREHENSIVE EXPERIMENTAL STUDY")
    print(" 1. Mathematical Function Optimization")
    print(" 2. Deep Neural Network Training (Holistic SPSA, Layer-Wise SPSA, SGD, Adam)")
    print("#" * 80)

    fig_dir = os.path.join(CURRENT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    t_start = time.time()

    # Execute Part 1
    print("\n>>> EXECUTING PART 1: SPSA ON MATHEMATICAL BENCHMARK FUNCTIONS...")
    math_results = run_math_benchmarks(save_dir=fig_dir)

    # Execute Part 2
    print("\n>>> EXECUTING PART 2: SPSA ON NEURAL NETWORK CLASSIFICATION...")
    nn_results = run_nn_benchmarks(save_dir=fig_dir)

    t_total = time.time() - t_start
    print("\n" + "#" * 80)
    print(f" ALL BENCHMARKS COMPLETED SUCCESSFULLY IN {t_total:.2f}s!")
    print(f" All publication-quality figures generated in: {fig_dir}")
    print("#" * 80)


if __name__ == "__main__":
    main()
