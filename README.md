# SPSA: Simultaneous Perturbation Stochastic Approximation & Layer-Wise Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A comprehensive implementation and empirical reproduction of **Simultaneous Perturbation Stochastic Approximation (SPSA)** for continuous mathematical optimization and forward-only deep neural network training without backpropagation, based on:

> **B. Wulff, J. Schuecker, C. Bauckhage.**  
> *SPSA for Layer-Wise Training of Deep Networks.*  
> International Conference on Artificial Neural Networks (**ICANN 2018**), LNCS 11141, pp. 564–573.  
> DOI: [10.1007/978-3-030-01424-7_55](https://doi.org/10.1007/978-3-030-01424-7_55)

**Implementation Contributors**:
- **Ananth Shyam**
- **Rama Narasimha**
- **Mahan Bhimireddy**

---

## 1. Repository Architecture

```text
.
├── demo/
│   └── implementation demonstration/    # Interactive, standalone SPSA demonstration & experimental study
│       ├── spsa_demonstration.ipynb     # Interactive Jupyter Notebook (All code, plots, & paper verification)
│       ├── part1_spsa_math.py           # Part 1: Optimization on Coupled Non-Convex Cubic Polynomial
│       ├── part2_spsa_nn.py             # Part 2: Neural Network Classification on Concentric Circles
│       ├── run_experiments.py           # Master runner executing both Part 1 and Part 2 end-to-end
│       ├── README.md                    # Detailed documentation for the demonstration suite
│       └── figures/                     # Publication-quality figures & periodic checkpoint CSV logs
│           ├── part1_spsa_math_convergence.png       # 2D contour paths & semi-log loss curves
│           ├── part1_math_convergence_intervals.csv   # Checkpoint logs every 15 iterations (Part 1)
│           ├── part2_loss_accuracy_comparison.png    # Loss vs Epochs, Loss vs Evals, Accuracy curves
│           ├── part2_decision_boundaries.png        # 2D decision boundary landscapes for all models
│           └── loss_every_20_epochs.csv              # Neural network 20-epoch checkpoint logs
│
├── code/                                # Full reproduction of the original ICANN 2018 paper
│   ├── README.md                        # Dedicated comprehensive README for the paper reproduction
│   ├── spsa_nn/                         # Core PyTorch library (Algorithm 1 & 2, datasets, device, MLP)
│   ├── experiments/                     # Scripts reproducing Fig. 1 (XOR/Moons/Circles) and Fig. 2 (MNIST)
│   ├── scripts/                         # Setup checks, data downloaders, reproduction shell scripts
│   ├── notebooks/                       # Full-protocol walkthrough notebooks (SPSA, MNIST vs Adam)
│   ├── paper_results/                   # Full-protocol reference outputs, data JSONs, and summary tables
│   ├── tests/                           # Pytest test suite
│   ├── requirements.txt                 # Dependency specification
│   ├── setup.py                         # Installable package setup
│   └── CITATION.cff                     # Citation metadata
│
└── README.md                            # Main project overview (This file)
```

---

## 2. Implementation Demonstration (`demo/implementation demonstration/`)

The `demo/implementation demonstration/` directory delivers a self-contained, publication-grade experimental study that investigates SPSA's behavior across two distinct regimes: continuous mathematical function optimization and forward-only deep neural network classification.

### 2.1 Part 1: Continuous Function Optimization on a Complex Landscape

Part 1 tests SPSA against analytical gradient descent on a challenging, non-convex mathematical landscape: the **Coupled Non-Convex Cubic Polynomial**:

$$f(x, y) = x^3 + y^3 - 3xy + 1.0$$

#### Landscape Properties:
- **Pure Cubic Monomials**: Contains pure degree-3 cubic terms ($x^3, y^3$) with **no quadratic $x^2 + y^2$ terms**.
- **Non-Linear Cross-Coupling**: The $-3xy$ interaction generates non-separable Hessian coupling ($H_{12} = H_{21} = -3$).
- **Critical Points & Saddle Landscape**:
  - **Saddle Point**: Located at $\mathbf{x}_{\text{saddle}} = (0.0, 0.0)$ with $f(0, 0) = 1.0$ and mixed eigenvalues $\lambda = \pm 3$.
  - **Global Attractor Minimum**: Located at $\mathbf{x}^* = (1.0, 1.0)$ with optimal value $f(1, 1) = 0.0$.
- **Starting Point**: Initialized at an asymmetric coordinate $\mathbf{x}_0 = (2.0, 0.3)$ with starting loss $f(\mathbf{x}_0) = 7.2270$.

#### Four Compared Optimizers:
1. **Holistic SPSA** (Spall 1992, 1998): Standard simultaneous Rademacher perturbation ($\Delta \in \{-1, +1\}^2$; strictly 2 forward evaluations per iteration).
2. **SPSA with Layer-Wise / Coordinate-Wise Update**: Perturbs one coordinate at a time while freezing the other ($2p = 4$ forward evaluations per iteration; eliminates cross-coupling perturbation noise).
3. **Normal Analytical Gradient Descent (GD)**: Exact analytical gradient ($\nabla f(x, y) = [3x^2 - 3y, \; 3y^2 - 3x]^T$).
4. **Analytical GD with Momentum**: Polyak classical momentum ($\beta = 0.70$) accelerating through anisotropic curvature.

#### Part 1 Performance Summary:

| Optimizer Method | Gradient Source | Backprop / Analytic? | Forward Evals / Step | Final Loss | Final Coordinates $(x, y)$ | Target Minimum |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Holistic SPSA** | Derivative-Free (Spall) | **No (0 bp)** | **2 forward** | **0.001556** | $(1.0232, 1.0219)$ | $(1.0, 1.0)$ |
| **2. Layer-Wise SPSA** | Coordinate-Wise SPSA | **No (0 bp)** | **4 forward** | **0.000112** | $(1.0004, 0.9941)$ | $(1.0, 1.0)$ |
| **3. Normal GD** | Analytical Calculus | **Yes (Analytic)** | Exact Grad | **0.000000** | $(1.0000, 1.0000)$ | $(1.0, 1.0)$ |
| **4. GD with Momentum** | Analytical + Polyak ($\beta=0.70$) | **Yes (Analytic)** | Exact Grad | **0.000000** | $(1.0000, 1.0000)$ | $(1.0, 1.0)$ |

#### Checkpoint Convergence Across Iterations:

| Iteration | 1. Holistic SPSA Loss | 2. Layer-Wise SPSA Loss | 3. Normal GD Loss | 4. GD with Momentum Loss |
|:---:|:---:|:---:|:---:|:---:|
| **0** | 7.227000 | 7.227000 | 7.227000 | 7.227000 |
| **15** | 0.226168 | 0.578728 | 0.000506 | 0.001364 |
| **30** | 0.036917 | 0.112899 | 0.000003 | 0.000143 |
| **45** | 0.015183 | 0.030824 | 0.000000 | 0.000001 |
| **60** | 0.007560 | 0.010251 | 0.000000 | 0.000000 |
| **90** | 0.003374 | 0.001648 | 0.000000 | 0.000000 |
| **120** | 0.002213 | 0.000374 | 0.000000 | 0.000000 |
| **150** | **0.001556** | **0.000112** | **0.000000** | **0.000000** |

*Insight*: Layer-Wise SPSA achieves **14x higher precision** than Holistic SPSA because decoupling coordinates eliminates destructive cross-coupling gradient noise.

---

### 2.2 Part 2: Neural Network Classification Without Backpropagation

Part 2 trains an 81-parameter multi-layer feed-forward neural network on the non-linear **Concentric Circles Dataset** ($N=250$, factor $= 0.4$, noise $\sigma = 0.06$).

- **Architecture**: $2 \to 5 \to 5 \to 5 \to 1$ ($L=4$ layers, tanh activations).
- **Objective**: Soft Margin Quadratic Loss $L = \frac{1}{2} \sum_i (1 - y_i \cdot f(x_i))^2$.
- **Fair Initialization**: All 5 optimizers start from the exact same initial parameter weights $p_0$.

#### Five Compared Optimizers:
1. **Layer-Wise SPSA** (Wulff et al., ICANN 2018): Updates each layer independently using derivative-free forward perturbations; **strictly 0 backprop**.
2. **Holistic SPSA** (Spall, 1992): Perturbs all 81 parameters across all 4 layers simultaneously; **strictly 0 backprop**.
3. **Pure Vanilla SGD** (Autograd, $\mu = 0.0$): Suffers from vanishing gradients in flat plateaus; traps at 50% accuracy.
4. **SGD with Momentum** (Autograd, $\mu = 0.9$): Overcomes plateaus, achieving 100% accuracy.
5. **Adam Optimizer** (Autograd, $\alpha = 0.02$): Rapid convergence via adaptive moments; 100% accuracy.

#### Part 2 Overall Performance Summary:

| Method | Backpropagation? | Initial Loss | Final Loss | Accuracy (%) | Runtime (s) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Layer-Wise SPSA** | **No (0 bp)** | 249.92 | **0.0045** | **100.0%** | 7.30 s |
| **Holistic SPSA** | **No (0 bp)** | 249.92 | 352.0000 | 64.8% | 12.74 s |
| **Pure Vanilla SGD** | Yes (Autograd) | 249.92 | 499.8501 | 50.0% | 0.54 s |
| **SGD (Momentum=0.9)** | Yes (Autograd) | 249.92 | **0.0003** | **100.0%** | 0.48 s |
| **Adam Optimizer** | Yes (Autograd) | 249.92 | **0.0490** | **100.0%** | 0.70 s |

#### Periodic Checkpoint Table (Logged Every 20 Epochs):

| Epoch | Layer-Wise SPSA | Holistic SPSA | Pure SGD | SGD (Momentum) | Adam |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | Loss: 249.92 (42.0%) | Loss: 249.92 (42.0%) | Loss: 249.92 (42.0%) | Loss: 249.92 (42.0%) | Loss: 249.92 (42.0%) |
| **20** | Loss: 8.30 (99.2%) | Loss: 352.00 (64.8%) | Loss: 499.98 (50.0%) | Loss: 0.00 (100.0%) | Loss: 2.42 (100.0%) |
| **40** | Loss: 0.01 (100.0%) | Loss: 352.00 (64.8%) | Loss: 499.98 (50.0%) | Loss: 0.00 (100.0%) | Loss: 0.23 (100.0%) |
| **60** | Loss: 0.01 (100.0%) | Loss: 352.00 (64.8%) | Loss: 499.97 (50.0%) | Loss: 0.00 (100.0%) | Loss: 0.12 (100.0%) |
| **80** | Loss: 0.01 (100.0%) | Loss: 352.00 (64.8%) | Loss: 499.95 (50.0%) | Loss: 0.00 (100.0%) | Loss: 0.07 (100.0%) |
| **100** | **Loss: 0.00 (100.0%)** | Loss: 352.00 (64.8%) | Loss: 499.85 (50.0%) | **Loss: 0.00 (100.0%)** | **Loss: 0.05 (100.0%)** |

---

### 2.3 Scientific Verification of ICANN 2018 Paper Claims

#### Claim 1: Layer-Wise SPSA Achieves Faster Learning than GD in Initial Epochs
Section 4.2 & Figure 2 of Wulff et al. (ICANN 2018) observe that Layer-Wise SPSA learns faster than gradient descent in the initial epochs. Our experimental logs directly verify this:

| Epoch | Layer-Wise SPSA (0 bp) | Holistic SPSA (0 bp) | Pure Vanilla GD | GD + Momentum (0.9) | Adam Optimizer |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **249.92** | 249.92 | 249.92 | 249.92 | 249.92 |
| **1** | **186.18** | 215.00 | 323.91 | 227.81 | 239.84 |
| **2** | **209.30** | 336.00 | 499.58 | 290.82 | 230.93 |
| **3** | **111.27** | 352.00 | 499.45 | 196.71 | 221.70 |
| **5** | **36.22** | 352.00 | 498.61 | 133.72 | 196.86 |
| **10** | **0.83** | 352.00 | 333.46 | 0.15 | 132.23 |
| **20** | **8.30** | 352.00 | 499.98 | 0.00 | 2.42 |

**Theoretical Explanation**:
1. **Finite Exploratory Jumps ($c_t \Delta$)**: SPSA evaluates loss at $\theta \pm c_t \Delta$. Unlike infinitesimal gradients ($\nabla L \approx 0$) that vanish on flat saddles, finite displacement acts as a stochastic probe that explores the local neighborhood and escapes plateau regions immediately.
2. **Directional Update Frequency**: Layer-Wise SPSA performs $t_{\text{max}} \times L = 120$ directional parameter adjustments per epoch, compared to 1 update per epoch for full-batch GD.

#### Claim 2: Asymptotic Equivalence (All Successful Optimizers Converge Similarly at the End)
By Epoch 100, Layer-Wise SPSA (**100.0% accuracy**, loss $0.0045$) matches the performance of exact analytical backpropagation optimizers (SGD+Momentum: **100.0%**, loss $0.0003$; Adam: **100.0%**, loss $0.0490$). All three close identical circular decision boundaries.

#### Claim 3: Holistic SPSA Fails on Deep Networks
Perturbing all 81 parameters simultaneously yields gradient variance scaling as $O(D)$. Destructive cross-layer interference dilutes updates, causing Holistic SPSA to stagnate at $64.8\%$ accuracy.

#### Claim 4: Derivative-Free / 0 Backpropagation
Layer-Wise SPSA achieves these results using strictly forward evaluations with zero backward computation graph storage, providing biological plausibility and compatibility with neuromorphic and analog hardware.

---

### 2.4 How to Run the Demonstration Suite

```bash
# Navigate to the demonstration directory
cd "demo/implementation demonstration"

# Run master benchmark runner (executes Part 1 and Part 2)
python run_experiments.py

# Or run individual modules:
python part1_spsa_math.py    # Generates 2D contour paths & interval CSV
python part2_spsa_nn.py      # Generates curves, decision boundaries, & 20-epoch logs

# Or launch the interactive Jupyter Notebook:
jupyter notebook spsa_demonstration.ipynb
```

---

## 3. Paper Implementation Reproduction (`code/`)

The `code/` directory contains the complete PyTorch reproduction of the full experimental protocol from Wulff et al. (ICANN 2018):

- **Didactic Problems (Section 4.1, Figure 1)**: Reproductions on XOR ($2 \times 5 \times 5 \times 5 \times 1$), Moons ($2 \times 5 \times 5 \times 5 \times 1$), and Circles ($2 \times 10^5 \times 3$), testing bottom-up, top-down, random, and holistic layer ordering.
- **MNIST Benchmarks (Section 4.2, Figure 2)**: Full-scale reproduction on MNIST using both the 771,270-parameter network ($784 \times 784 \times 196 \times 10$) and the 155,830-parameter network ($784 \times 196 \times 10$), comparing SPSA against full-batch Gradient Descent and Adam.
- **Ablation Studies**: Evaluation of the weights/biases split and holistic SPSA failure modes on MNIST.
- **PyTorch Library (`spsa_nn`)**: Modular implementation including Algorithm 1, Algorithm 2, GPU/CPU auto-dispatch, datasets, and plotting routines.
- **Unit Tests & CI**: Automated testing suite via `pytest` and GitHub Actions.

> **Note**: For full reproduction commands, environment configuration, GPU flags, and extended protocol documentation, please refer to the dedicated [code/README.md](code/README.md).

---

## 4. Citation

If you use this codebase or reproduction in your research, please cite:

```bibtex
@inproceedings{wulff2018spsa,
  title     = {SPSA for Layer-Wise Training of Deep Networks},
  author    = {Wulff, Benjamin and Schuecker, Jannis and Bauckhage, Christian},
  booktitle = {Artificial Neural Networks and Machine Learning -- ICANN 2018},
  series    = {LNCS},
  volume    = {11141},
  pages     = {564--573},
  publisher = {Springer},
  year      = {2018},
  doi       = {10.1007/978-3-030-01424-7_55}
}
```
