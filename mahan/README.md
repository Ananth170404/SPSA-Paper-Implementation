# Mahan's SPSA Implementation & Benchmark Suite

This directory contains the executed Jupyter Notebook, standalone Python source code, high-resolution figures, and CSV data logs for **Mahan's SPSA Implementation**.

---

## 1. Directory Contents

```text
mahan/
├── mahan.ipynb                        # Fully executed Jupyter Notebook with embedded plots and outputs
├── mahan_code.py                      # Standalone, runnable Python script extracted from the notebook
├── README.md                          # Documentation of experiments and results (this file)
├── part1_optimization_summary.csv     # Part 1 final summary metrics
├── part1_convergence_checkpoints.csv  # Part 1 iteration checkpoints (every 15 steps)
├── part2_final_comparison.csv         # Part 2 neural network 100-epoch results
├── part2_early_epochs_2a.csv          # Table 2A: Early epoch dynamics (epochs 0 to 20)
├── part2_full_progression_2b.csv      # Table 2B: Full progression (every 20 epochs to 100)
└── figures/
    ├── part1_trajectories_convergence.png  # Part 1 2D contour paths & log-loss convergence
    ├── part2_loss_evals_accuracy.png       # Part 2 training loss, evals, and held-out test accuracy
    └── part2_decision_boundaries.png       # Part 2 2D decision boundaries across all 5 models
```

---

## 2. Experimental Summary & Key Findings

### Part 1: Cubic Polynomial Optimization ($f(x, y) = x^3 + y^3 - 3xy + 1$)
* **Initial State**: $\mathbf{x}_0 = (2.0, 0.3)$, Loss $= 7.227000$
* **Target Minimum**: $\mathbf{x}^* = (1.0, 1.0)$, $f(\mathbf{x}^*) = 0.0$

| Method | Evals/Step | Final Loss | Final Coordinates $(x, y)$ |
|:---|:---:|:---:|:---:|
| **1. Holistic SPSA** | 2 forward | 0.001556 | (1.0232, 1.0219) |
| **2. Coordinate-Wise FDSA** | 4 forward | 0.000112 | (1.0004, 0.9941) |
| **3. GD (Exact Analytical)** | Exact Grad | 0.000000 | (1.0000, 1.0000) |
| **4. SGD (Noisy Gradient)** | Noisy Grad | 0.000005 | (1.0011, 1.0015) |

---

### Part 2: Neural Network Classification on Concentric Circles
* **Dataset**: $N=250$ concentric circles ($200$ train / $50$ held-out test samples).
* **Architecture**: $2 \to 5 \to 5 \to 5 \to 1$ ($3$ tanh hidden layers + $1$ linear output layer, $81$ parameters).

#### Final Performance Comparison (Epoch 100):
| Optimizer | Backprop? | Train Loss | Train Acc | Test Acc | Full Evals | Runtime (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Holistic SPSA** | **No** | 692831070122409984.0000 | 29.5% | 24.0% | 24000 | 9.89 s |
| **2. Layer-Wise SPSA** | **No** | 1334806291486343168.0000 | 50.0% | 50.0% | 24000 | 5.30 s |
| **3. GD (Full-Batch)** | Yes | nan | 50.0% | 50.0% | 300 | 1.67 s |
| **4. SGD (Mini-Batch)** | Yes | 8.5783 | 100.0% | 100.0% | 300 | 0.58 s |

---

## 3. How to Run

1. **Jupyter Notebook**:
   ```bash
   jupyter notebook mahan.ipynb
   ```
2. **Python Code**:
   ```bash
   python mahan_code.py
   ```
