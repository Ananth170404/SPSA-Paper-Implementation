# SPSA Implementation Demonstration
## Mathematical Function Optimization & Forward-Only Deep Learning Without Backpropagation

**Implementation by**: Ananth Shyam, Rama Narasimha, and Mahan Bhimireddy  
**Reference Paper**: *B. Wulff, J. Schuecker, C. Bauckhage. SPSA for Layer-Wise Training of Deep Networks. ICANN 2018, LNCS 11141, pp. 564–573. [https://doi.org/10.1007/978-3-030-01424-7_55](https://doi.org/10.1007/978-3-030-01424-7_55)*

---

## 1. Overview

This directory contains a complete, self-contained interactive experimental suite investigating **Simultaneous Perturbation Stochastic Approximation (SPSA)** across continuous function optimization and deep neural network training.

### Core Evaluated Optimizers (4 Methods Across Both Parts):
1. **Holistic SPSA** (Spall 1992, 1998): Derivative-free simultaneous perturbation of all parameters (2 forward evaluations per update; 0 backprop).
2. **Layer-Wise SPSA** (Wulff et al., ICANN 2018 Algorithm 2): Derivative-free, layer-by-layer localized perturbation while freezing remaining parameters (0 backprop).
3. **GD (Gradient Descent)**: Full-batch analytical gradient descent using PyTorch's built-in `torch.optim.SGD`.
4. **SGD (Stochastic Gradient Descent)**: Mini-batch / stochastic gradient descent using PyTorch's built-in `torch.optim.SGD`.

---

## 2. Directory Layout

All interactive code, figures, tables, and mathematical proofs are self-contained in the Jupyter Notebook:

```text
demo/implementation demonstration/
├── spsa_demonstration.ipynb   # Master interactive Jupyter Notebook (Math, Code, Figures, Tables, Proofs)
├── README.md                  # Comprehensive scientific documentation (This file)
└── figures/                   # Generated publication-quality figures & checkpoint CSV logs
    ├── part1_spsa_math_convergence.png       # 2D contour paths & semi-log loss curves (Part 1)
    ├── part1_math_convergence_intervals.csv   # Checkpoint logs every 15 iterations / epochs (Part 1)
    ├── part2_loss_accuracy_comparison.png    # Loss vs Epochs and Classification Accuracy curves (1x2 layout)
    ├── part2_decision_boundaries.png        # 2D decision boundary landscapes for all 4 models (Part 2)
    └── loss_every_20_epochs.csv              # Neural network 20-epoch checkpoint logs (Part 2)
```

---

## 3. Part 1: Continuous Function Optimization on a Complex Landscape

### 3.1 Benchmark Landscape
Coupled Non-Convex Cubic Polynomial:
$$f(x, y) = x^3 + y^3 - 3xy + 1.0$$
- **Pure cubic degree**: Contains $x^3$ and $y^3$, with **no quadratic $x^2 + y^2$ terms**.
- **Coupled Curvature**: Cross-product $-3xy$ term generates non-separable Hessian coupling ($H_{12} = H_{21} = -3$).
- **Saddle Point**: $\mathbf{x}_{\text{saddle}} = (0.0, 0.0)$, $f(0, 0) = 1.0$, eigenvalues $\lambda = \pm 3$.
- **Local Minimum (Minima)**: $\mathbf{x}^* = (1.0, 1.0)$, $f(1, 1) = 0.0$, eigenvalues $\lambda = 3, 9$.
- **Starting Point**: $\mathbf{x}_0 = (2.0, 0.3)$ with $f(\mathbf{x}_0) = 7.2270$.

### 3.2 Part 1 Optimization Results:

| Optimizer Method | Gradient Source | Backprop / Analytic? | Forward Evals / Step | Final Loss | Final Coordinates $(x, y)$ | Target Local Minima |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Holistic SPSA** | Derivative-Free (Spall) | **No (0 bp)** | **2 forward** | **0.001602** | $(1.0232, 1.0219)$ | $(1.0, 1.0)$ |
| **2. Layer-Wise SPSA** | Coordinate-Wise SPSA | **No (0 bp)** | **4 forward** | **0.000116** | $(1.0004, 0.9941)$ | $(1.0, 1.0)$ |
| **3. GD (Built-in)** | PyTorch `torch.optim.SGD` | **Yes (Autodiff)** | Exact Grad | **0.000000** | $(1.0000, 1.0000)$ | $(1.0, 1.0)$ |
| **4. SGD (Built-in)** | PyTorch `torch.optim.SGD` | **Yes (Autodiff)** | Noisy Grad | **0.000023** | $(1.0010, 1.0013)$ | $(1.0, 1.0)$ |

### 3.3 Epoch-Wise Convergence Comparison (10 Epoch Cycles, 15 Updates / Epoch):

| Epoch | Iteration | 1. Holistic SPSA Loss | 2. Layer-Wise SPSA Loss | 3. Built-in GD Loss | 4. Built-in SGD Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | 0 | 7.227000 | 7.227000 | 7.227000 | 7.227000 |
| **1** | 15 | 0.226168 | 0.578728 | 0.000919 | 0.000918 |
| **2** | 30 | 0.036917 | 0.112899 | 0.000003 | 0.000045 |
| **3** | 45 | 0.015183 | 0.030824 | 0.000000 | 0.000016 |
| **4** | 60 | 0.007560 | 0.010251 | 0.000000 | 0.000002 |
| **5** | 75 | 0.004893 | 0.003904 | 0.000000 | 0.000060 |
| **6** | 90 | 0.003374 | 0.001648 | 0.000000 | 0.000032 |
| **7** | 105 | 0.002515 | 0.000757 | 0.000000 | 0.000152 |
| **8** | 120 | 0.002213 | 0.000374 | 0.000000 | 0.000052 |
| **9** | 135 | 0.001874 | 0.000198 | 0.000000 | 0.000005 |
| **10** | 150 | **0.001602** | **0.000116** | **0.000000** | **0.000023** |

*Insight*: Layer-Wise SPSA achieves higher precision than Holistic SPSA because decoupling coordinates eliminates destructive cross-coupling gradient noise.

---

## 4. Part 2: Neural Network Classification Without Backpropagation

### 4.1 Task & Architecture Specification
- **Dataset**: Non-linear Concentric Circles ($N=250$, factor $= 0.4$, noise $\sigma = 0.06$).
- **Network**: $2 \to 5 \to 5 \to 5 \to 1$ ($L=4$ layers, $\tanh$ activations, $\beta = 0.5$, **81 total parameters**).
- **Objective**: Soft Margin Quadratic Loss $L = \sum_i (f(x_i) - y_i)^2$.
- **Fair Initialization**: All 4 optimizers start from the exact same initial parameter weights $p_0$.

### 4.2 Part 2 Overall Performance Summary:

| Method | Backpropagation? | Initial Loss | Final Loss | Accuracy (%) | Runtime (s) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1. Holistic SPSA** | **No (0 bp)** | 249.92 | 352.0000 | 64.8% | 12.66 s |
| **2. Layer-Wise SPSA** | **No (0 bp)** | 249.92 | **0.0216** | **100.0%** | 7.62 s |
| **3. Built-in GD** | Yes (`torch.optim.SGD`) | 249.92 | **6.1283** | **100.0%** | 0.15 s |
| **4. Built-in SGD** | Yes (`torch.optim.SGD`) | 249.92 | **0.2977** | **100.0%** | 1.03 s |

### 4.3 Table 2A: Initial Epoch Dynamics (Epochs 0 to 20 — Loss & Accuracy):

| Epoch | 1. Holistic SPSA (0 bp) | 2. Layer-Wise SPSA (0 bp) | 3. Built-in GD | 4. Built-in SGD |
|:---:|:---:|:---:|:---:|:---:|
| **0** | Loss: 249.92 (42.0%) | Loss: 249.92 (42.0%) | Loss: 249.92 (42.0%) | Loss: 249.92 (42.0%) |
| **1** | Loss: 215.00 (78.4%) | **Loss: 186.18 (81.2%)** | Loss: 247.89 (66.4%) | Loss: 241.11 (58.8%) |
| **2** | Loss: 336.00 (66.4%) | **Loss: 209.51 (66.8%)** | Loss: 246.87 (37.6%) | Loss: 238.46 (66.4%) |
| **3** | Loss: 352.00 (64.8%) | **Loss: 112.21 (86.4%)** | Loss: 248.64 (64.0%) | Loss: 234.20 (66.4%) |
| **5** | Loss: 352.00 (64.8%) | **Loss: 83.75 (89.6%)** | Loss: 261.73 (58.8%) | Loss: 218.11 (68.0%) |
| **10** | Loss: 352.00 (64.8%) | **Loss: 72.88 (92.8%)** | Loss: 258.93 (50.0%) | Loss: 163.52 (79.6%) |
| **15** | Loss: 352.00 (64.8%) | **Loss: 1.29 (100.0%)** | Loss: 251.70 (60.8%) | Loss: 129.18 (83.2%) |
| **20** | Loss: 352.00 (64.8%) | **Loss: 0.26 (100.0%)** | Loss: 242.83 (50.0%) | **Loss: 47.13 (96.4%)** |

### 4.4 Table 2B: Full Training Progression (Logged Every 20 Epochs to Epoch 100):

| Epoch | 1. Holistic SPSA (0 bp) | 2. Layer-Wise SPSA (0 bp) | 3. Built-in GD | 4. Built-in SGD |
|:---:|:---:|:---:|:---:|:---:|
| **0** | Loss: 249.92 (42.0%) | Loss: 249.92 (42.0%) | Loss: 249.92 (42.0%) | Loss: 249.92 (42.0%) |
| **20** | Loss: 352.00 (64.8%) | **Loss: 0.26 (100.0%)** | Loss: 242.83 (50.0%) | Loss: 47.13 (96.4%) |
| **40** | Loss: 352.00 (64.8%) | **Loss: 0.06 (100.0%)** | Loss: 206.03 (56.0%) | **Loss: 1.53 (100.0%)** |
| **60** | Loss: 352.00 (64.8%) | **Loss: 0.04 (100.0%)** | Loss: 144.42 (79.2%) | **Loss: 0.67 (100.0%)** |
| **80** | Loss: 352.00 (64.8%) | **Loss: 0.03 (100.0%)** | Loss: 111.61 (87.6%) | **Loss: 0.42 (100.0%)** |
| **100** | Loss: 352.00 (64.8%) | **Loss: 0.02 (100.0%)** | **Loss: 6.13 (100.0%)** | **Loss: 0.30 (100.0%)** |

---

## 5. Scientific Paper Verification (Wulff et al. ICANN 2018)

1. **Claim 1 (Initial Epoch Faster Learning)**: Layer-Wise SPSA drops loss to $186.18$ ($81.2\%$ accuracy) in Epoch 1, and reaches $100.0\%$ accuracy by Epoch 15, significantly faster than GD ($60.8\%$) and SGD ($83.2\%$). Finite exploratory perturbations $c_t \Delta$ detect gradient slopes across flat saddle plateaus where infinitesimal backprop gradients $\nabla L \approx 0$ vanish.
2. **Claim 2 (Asymptotic Equivalence)**: By Epoch 100, Layer-Wise SPSA matches backpropagation baselines (GD and SGD) at **100.0% accuracy**, forming identical circular decision boundaries without calculating any analytical gradients or storing backward computational graphs.
3. **Claim 3 (Holistic SPSA Degradation)**: Simultaneous perturbation over all 81 parameters scales variance as $O(D)$ and causes destructive inter-layer interference, stalling Holistic SPSA at $64.8\%$ accuracy.
4. **Claim 4 (Derivative-Free Plausibility)**: Layer-Wise SPSA trains deep architectures using forward passes only, providing biological plausibility and enabling neuromorphic hardware execution.

---

## 6. How to Run the Demonstration

Launch the self-contained interactive notebook:

```bash
jupyter notebook spsa_demonstration.ipynb
```
