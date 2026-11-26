"""
Part 2: Neural Network Classification via SPSA vs Conventional Optimizers
========================================================================
This module trains a feed-forward neural network for non-linear binary
classification on Concentric Circles without using backpropagation (derivative-free SPSA),
and compares its performance against conventional gradient-based optimizers.

Optimizers Compared (All 5 Methods):
  1. Layer-Wise SPSA (Derivative-free, layer-by-layer perturbation; Wulff et al. ICANN 2018)
  2. Holistic SPSA (Derivative-free, all parameters perturbed simultaneously)
  3. Pure Vanilla SGD (No momentum, exact backpropagation)
  4. SGD with Momentum (Momentum = 0.9, exact backpropagation)
  5. Adam Optimizer (Adaptive moment estimation, exact backpropagation)

All optimizers start from the EXACT SAME initial parameter weights for complete fairness.
"""

import os
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
import torch
import torch.nn as nn


# Set seeds for complete reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# =====================================================================
# 1. Dataset Generation
# =====================================================================

def generate_classification_data(
    dataset_type: str = "circles",
    n_samples: int = 250,
    noise: float = 0.06,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates synthetic 2D non-linear classification dataset.
    Concentric circles: Inner disc (y = +1.0) vs Outer ring (y = -1.0).
    """
    if dataset_type == "circles":
        x_raw, y_cls = make_circles(n_samples=n_samples, noise=noise, factor=0.4, random_state=seed)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    y_target = np.where(y_cls == 1, 1.0, -1.0)[:, None]
    x = torch.tensor(x_raw, dtype=torch.float32)
    y = torch.tensor(y_target, dtype=torch.float32)
    labels = torch.tensor(y_cls, dtype=torch.long)
    return x, y, labels


# =====================================================================
# 2. Forward-Only Deep Neural Network Architecture
# =====================================================================

def init_network_params(
    sizes: List[int] = [2, 5, 5, 5, 1],
    init: str = "standard",
    seed: int = 42,
) -> List[torch.Tensor]:
    """
    Initializes network parameters for all layers.
    W: (1, n_out, n_in), b: (1, n_out)
    Matches ICANN 2018 paper: 2 x 5 x 5 x 5 x 1 (81 total parameters).
    """
    gen = torch.Generator().manual_seed(seed)
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        if init == "standard":
            W = torch.randn(1, n_out, n_in, generator=gen)
            b = torch.randn(1, n_out, generator=gen)
        elif init == "glorot":
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = (torch.rand(1, n_out, n_in, generator=gen) * 2.0 - 1.0) * limit
            b = torch.zeros(1, n_out)
        else:
            raise ValueError(f"Unknown init: {init}")
        params.extend([W, b])
    return params


def forward_pass(x: torch.Tensor, params: List[torch.Tensor], beta: float = 0.5) -> torch.Tensor:
    """
    Evaluates forward pass: a^{l} = tanh(beta * (a^{l-1} @ W^T + b)).
    Strictly forward passes only. No autograd tape or backward pass required.
    """
    a = x
    num_layers = len(params) // 2
    for l in range(num_layers):
        W = params[2 * l]
        b = params[2 * l + 1]
        z = torch.matmul(a, W.transpose(-1, -2)) + b.unsqueeze(-2)
        a = torch.tanh(beta * z)
    return a


def compute_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sum of squared residuals loss (Equation 9 in ICANN 2018 paper)."""
    return (pred - y).pow(2).sum()


def compute_accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Binary classification accuracy in percent."""
    correct = (pred.sign() == y).float().mean()
    return float(correct.item() * 100.0)


# =====================================================================
# 3. Optimization Algorithms (All 5 Methods)
# =====================================================================

def train_layerwise_spsa(
    x: torch.Tensor,
    y: torch.Tensor,
    init_params: List[torch.Tensor],
    epochs: int = 100,
    t_max: int = 30,
    a: float = 0.02,
    c: float = 0.10,
    alpha: float = 1.0,
    gamma: float = 1.0 / 6.0,
    beta: float = 0.5,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
    """
    Layer-Wise SPSA (Algorithm 2 in Wulff et al. ICANN 2018).
    Updates each layer l while freezing all other layers.
    Derivative-free: No backpropagation!
    """
    torch.manual_seed(seed)
    params = [p.clone() for p in init_params]
    num_layers = len(params) // 2

    history = {
        "loss": [compute_loss(forward_pass(x, params, beta), y).item()],
        "accuracy": [compute_accuracy(forward_pass(x, params, beta), y)],
        "evals": [0],
    }

    total_evals = 0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        for l in range(num_layers):
            idx_w = 2 * l
            idx_b = 2 * l + 1
            w = params[idx_w]
            b = params[idx_b]

            for t in range(1, t_max + 1):
                at = a / (t ** alpha)
                ct = c / (t ** gamma)

                # Rademacher perturbation vector Delta in {-1, +1}
                delta_w = (torch.randint(0, 2, w.shape) * 2 - 1).float()
                delta_b = (torch.randint(0, 2, b.shape) * 2 - 1).float()

                # Positive perturbation
                params[idx_w] = w + ct * delta_w
                params[idx_b] = b + ct * delta_b
                loss_plus = compute_loss(forward_pass(x, params, beta), y)

                # Negative perturbation
                params[idx_w] = w - ct * delta_w
                params[idx_b] = b - ct * delta_b
                loss_minus = compute_loss(forward_pass(x, params, beta), y)

                total_evals += 2  # exactly 2 forward evaluations

                # Gradient approximation
                g_hat_w = (loss_plus - loss_minus) / (2.0 * ct * delta_w)
                g_hat_b = (loss_plus - loss_minus) / (2.0 * ct * delta_b)

                # Update layer parameters
                w = w - at * g_hat_w
                b = b - at * g_hat_b

            params[idx_w] = w
            params[idx_b] = b

        curr_loss = compute_loss(forward_pass(x, params, beta), y).item()
        curr_acc = compute_accuracy(forward_pass(x, params, beta), y)
        history["loss"].append(curr_loss)
        history["accuracy"].append(curr_acc)
        history["evals"].append(total_evals)

    history["runtime"] = time.perf_counter() - t0
    return params, history


def train_holistic_spsa(
    x: torch.Tensor,
    y: torch.Tensor,
    init_params: List[torch.Tensor],
    epochs: int = 100,
    t_max: int = 30,
    a: float = 0.02,
    c: float = 0.10,
    alpha: float = 1.0,
    gamma: float = 1.0 / 6.0,
    beta: float = 0.5,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
    """
    Holistic SPSA (Spall 1992 Algorithm 1 applied to all parameters simultaneously).
    Matches computation budget (L * t_max steps per epoch).
    Derivative-free: No backpropagation!
    """
    torch.manual_seed(seed)
    params = [p.clone() for p in init_params]
    num_layers = len(params) // 2

    history = {
        "loss": [compute_loss(forward_pass(x, params, beta), y).item()],
        "accuracy": [compute_accuracy(forward_pass(x, params, beta), y)],
        "evals": [0],
    }

    total_evals = 0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        for _ in range(num_layers):
            for t in range(1, t_max + 1):
                at = a / (t ** alpha)
                ct = c / (t ** gamma)

                deltas = [(torch.randint(0, 2, p.shape) * 2 - 1).float() for p in params]

                # Positive perturbation
                for i in range(len(params)):
                    params[i] += ct * deltas[i]
                loss_plus = compute_loss(forward_pass(x, params, beta), y)

                # Negative perturbation
                for i in range(len(params)):
                    params[i] -= 2.0 * ct * deltas[i]
                loss_minus = compute_loss(forward_pass(x, params, beta), y)

                total_evals += 2

                # Parameter update
                for i in range(len(params)):
                    g_hat = (loss_plus - loss_minus) / (2.0 * ct * deltas[i])
                    params[i] = (params[i] + ct * deltas[i]) - at * g_hat

        curr_loss = compute_loss(forward_pass(x, params, beta), y).item()
        curr_acc = compute_accuracy(forward_pass(x, params, beta), y)
        history["loss"].append(curr_loss)
        history["accuracy"].append(curr_acc)
        history["evals"].append(total_evals)

    history["runtime"] = time.perf_counter() - t0
    return params, history


def train_pure_sgd(
    x: torch.Tensor,
    y: torch.Tensor,
    init_params: List[torch.Tensor],
    epochs: int = 100,
    steps_per_epoch: int = 4,
    lr: float = 0.03,
    beta: float = 0.5,
) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
    """
    Pure Vanilla SGD (No momentum, momentum=0.0) using exact analytical backpropagation.
    """
    params = [p.clone().detach().requires_grad_(True) for p in init_params]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.0)

    history = {
        "loss": [compute_loss(forward_pass(x, params, beta), y).item()],
        "accuracy": [compute_accuracy(forward_pass(x, params, beta), y)],
        "evals": [0],
    }

    total_evals = 0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            loss = compute_loss(forward_pass(x, params, beta), y)
            loss.backward()  # Backpropagation
            optimizer.step()
            total_evals += 3  # 1 forward + 1 backward (approx 2 forward equivalents) = 3

        with torch.no_grad():
            curr_loss = compute_loss(forward_pass(x, params, beta), y).item()
            curr_acc = compute_accuracy(forward_pass(x, params, beta), y)
            history["loss"].append(curr_loss)
            history["accuracy"].append(curr_acc)
            history["evals"].append(total_evals)

    history["runtime"] = time.perf_counter() - t0
    return [p.detach() for p in params], history


def train_sgd_momentum(
    x: torch.Tensor,
    y: torch.Tensor,
    init_params: List[torch.Tensor],
    epochs: int = 100,
    steps_per_epoch: int = 4,
    lr: float = 0.005,
    momentum: float = 0.9,
    beta: float = 0.5,
) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
    """
    SGD with Momentum (momentum=0.9) using exact analytical backpropagation.
    """
    params = [p.clone().detach().requires_grad_(True) for p in init_params]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)

    history = {
        "loss": [compute_loss(forward_pass(x, params, beta), y).item()],
        "accuracy": [compute_accuracy(forward_pass(x, params, beta), y)],
        "evals": [0],
    }

    total_evals = 0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            loss = compute_loss(forward_pass(x, params, beta), y)
            loss.backward()  # Backpropagation
            optimizer.step()
            total_evals += 3

        with torch.no_grad():
            curr_loss = compute_loss(forward_pass(x, params, beta), y).item()
            curr_acc = compute_accuracy(forward_pass(x, params, beta), y)
            history["loss"].append(curr_loss)
            history["accuracy"].append(curr_acc)
            history["evals"].append(total_evals)

    history["runtime"] = time.perf_counter() - t0
    return [p.detach() for p in params], history


def train_conventional_adam(
    x: torch.Tensor,
    y: torch.Tensor,
    init_params: List[torch.Tensor],
    epochs: int = 100,
    steps_per_epoch: int = 4,
    lr: float = 0.02,
    beta: float = 0.5,
) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
    """
    Adam optimizer using exact Backpropagation (analytic gradients and adaptive moments).
    """
    params = [p.clone().detach().requires_grad_(True) for p in init_params]
    optimizer = torch.optim.Adam(params, lr=lr)

    history = {
        "loss": [compute_loss(forward_pass(x, params, beta), y).item()],
        "accuracy": [compute_accuracy(forward_pass(x, params, beta), y)],
        "evals": [0],
    }

    total_evals = 0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            loss = compute_loss(forward_pass(x, params, beta), y)
            loss.backward()  # Backpropagation
            optimizer.step()
            total_evals += 3

        with torch.no_grad():
            curr_loss = compute_loss(forward_pass(x, params, beta), y).item()
            curr_acc = compute_accuracy(forward_pass(x, params, beta), y)
            history["loss"].append(curr_loss)
            history["accuracy"].append(curr_acc)
            history["evals"].append(total_evals)

    history["runtime"] = time.perf_counter() - t0
    return [p.detach() for p in params], history


# =====================================================================
# 4. Master Benchmark & Visualizations
# =====================================================================

DEFAULT_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

def run_nn_benchmarks(save_dir: str = DEFAULT_FIG_DIR):
    """
    Runs comprehensive neural network benchmark comparing all 5 optimizers:
    Layer-Wise SPSA, Holistic SPSA, Pure SGD, SGD+Momentum, and Adam.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 88)
    print(" PART 2: NEURAL NETWORK CLASSIFICATION BENCHMARK (CONCENTRIC CIRCLES)")
    print(" Comparing: Layer-Wise SPSA, Holistic SPSA, Pure SGD, SGD+Momentum, and Adam")
    print("=" * 88)

    # 1. Dataset
    x, y, labels = generate_classification_data("circles", n_samples=250, noise=0.06, seed=SEED)
    sizes = [2, 5, 5, 5, 1]  # Topology from ICANN 2018 paper (81 parameters)
    epochs = 100
    t_max = 30
    beta = 0.5

    # 2. Shared Initialization
    p0 = init_network_params(sizes, init="standard", seed=SEED)
    init_loss = compute_loss(forward_pass(x, p0, beta), y).item()
    init_acc = compute_accuracy(forward_pass(x, p0, beta), y)

    print(f"Network Topology : {' -> '.join(map(str, sizes))} (81 parameters)")
    print(f"Dataset          : Concentric Circles (N={len(x)} samples)")
    print(f"Initial State    : Loss = {init_loss:.2f} | Accuracy = {init_acc:.1f}%")
    print("-" * 88)

    # 3. Train all 5 optimizers
    print("\n[1/5] Training Layer-Wise SPSA (Forward passes only, No backprop)...")
    p_lw, hist_lw = train_layerwise_spsa(x, y, p0, epochs=epochs, t_max=t_max, a=0.02, c=0.10, beta=beta)
    print(f"      -> Final Loss: {hist_lw['loss'][-1]:.4f} | Acc: {hist_lw['accuracy'][-1]:.1f}% ({hist_lw['runtime']:.2f}s)")

    print("\n[2/5] Training Holistic SPSA (Forward passes only, No backprop)...")
    p_hol, hist_hol = train_holistic_spsa(x, y, p0, epochs=epochs, t_max=t_max, a=0.02, c=0.10, beta=beta)
    print(f"      -> Final Loss: {hist_hol['loss'][-1]:.4f} | Acc: {hist_hol['accuracy'][-1]:.1f}% ({hist_hol['runtime']:.2f}s)")

    print("\n[3/5] Training Pure Vanilla SGD (No momentum, Exact backprop)...")
    p_sgd, hist_sgd = train_pure_sgd(x, y, p0, epochs=epochs, steps_per_epoch=4, lr=0.03, beta=beta)
    print(f"      -> Final Loss: {hist_sgd['loss'][-1]:.4f} | Acc: {hist_sgd['accuracy'][-1]:.1f}% ({hist_sgd['runtime']:.2f}s)")

    print("\n[4/5] Training SGD with Momentum (Momentum=0.9, Exact backprop)...")
    p_mom, hist_mom = train_sgd_momentum(x, y, p0, epochs=epochs, steps_per_epoch=4, lr=0.005, momentum=0.9, beta=beta)
    print(f"      -> Final Loss: {hist_mom['loss'][-1]:.4f} | Acc: {hist_mom['accuracy'][-1]:.1f}% ({hist_mom['runtime']:.2f}s)")

    print("\n[5/5] Training Adam Optimizer (Adaptive moments, Exact backprop)...")
    p_adam, hist_adam = train_conventional_adam(x, y, p0, epochs=epochs, steps_per_epoch=4, lr=0.02, beta=beta)
    print(f"      -> Final Loss: {hist_adam['loss'][-1]:.4f} | Acc: {hist_adam['accuracy'][-1]:.1f}% ({hist_adam['runtime']:.2f}s)")

    # -------------------------------------------------------------
    # Figure 1: Comparative Loss and Accuracy Curves (All 5 Optimizers)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    colors = {
        "Layer-Wise SPSA": "#1f77b4",
        "Holistic SPSA": "#d62728",
        "Pure SGD": "#9467bd",
        "SGD (Momentum)": "#2ca02c",
        "Adam": "#ff7f0e",
    }
    styles = {
        "Layer-Wise SPSA": "-",
        "Holistic SPSA": "--",
        "Pure SGD": "-.",
        "SGD (Momentum)": "-",
        "Adam": ":",
    }

    # Subplot A: Loss vs Epochs
    ax = axes[0]
    ax.plot(hist_lw["loss"], label="Layer-Wise SPSA (0 bp)", color=colors["Layer-Wise SPSA"], linestyle=styles["Layer-Wise SPSA"], linewidth=2.2)
    ax.plot(hist_hol["loss"], label="Holistic SPSA (0 bp)", color=colors["Holistic SPSA"], linestyle=styles["Holistic SPSA"], linewidth=2.0)
    ax.plot(hist_sgd["loss"], label="Pure SGD (No mom)", color=colors["Pure SGD"], linestyle=styles["Pure SGD"], linewidth=1.8)
    ax.plot(hist_mom["loss"], label="SGD (Momentum=0.9)", color=colors["SGD (Momentum)"], linestyle=styles["SGD (Momentum)"], linewidth=1.8)
    ax.plot(hist_adam["loss"], label="Adam", color=colors["Adam"], linestyle=styles["Adam"], linewidth=1.8)
    ax.set_title("(a) Training Loss vs Epochs", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss E(D, W)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.5)

    # Subplot B: Loss vs Function Evaluations
    ax = axes[1]
    ax.plot(hist_lw["evals"], hist_lw["loss"], label="Layer-Wise SPSA", color=colors["Layer-Wise SPSA"], linewidth=2.2)
    ax.plot(hist_hol["evals"], hist_hol["loss"], label="Holistic SPSA", color=colors["Holistic SPSA"], linestyle="--", linewidth=2.0)
    ax.plot(hist_sgd["evals"], hist_sgd["loss"], label="Pure SGD", color=colors["Pure SGD"], linestyle="-.", linewidth=1.8)
    ax.plot(hist_mom["evals"], hist_mom["loss"], label="SGD (Momentum)", color=colors["SGD (Momentum)"], linewidth=1.8)
    ax.plot(hist_adam["evals"], hist_adam["loss"], label="Adam", color=colors["Adam"], linestyle=":", linewidth=1.8)
    ax.set_title("(b) Loss vs Function Evaluations", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Forward / Backward Evals", fontsize=10)
    ax.set_ylabel("Loss E(D, W)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.5)

    # Subplot C: Accuracy vs Epochs
    ax = axes[2]
    ax.plot(hist_lw["accuracy"], label="Layer-Wise SPSA", color=colors["Layer-Wise SPSA"], linewidth=2.2)
    ax.plot(hist_hol["accuracy"], label="Holistic SPSA", color=colors["Holistic SPSA"], linestyle="--", linewidth=2.0)
    ax.plot(hist_sgd["accuracy"], label="Pure SGD", color=colors["Pure SGD"], linestyle="-.", linewidth=1.8)
    ax.plot(hist_mom["accuracy"], label="SGD (Momentum)", color=colors["SGD (Momentum)"], linewidth=1.8)
    ax.plot(hist_adam["accuracy"], label="Adam", color=colors["Adam"], linestyle=":", linewidth=1.8)
    ax.set_title("(c) Classification Accuracy vs Epochs", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.5)

    plt.tight_layout()
    fig1_path = os.path.join(save_dir, "part2_loss_accuracy_comparison.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"\n[Plot Saved] Loss and accuracy comparison saved to: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Learned Decision Boundaries (All 5 Optimizers)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.8))
    x_min, x_max = x[:, 0].min().item() - 0.4, x[:, 0].max().item() + 0.4
    y_min, y_max = x[:, 1].min().item() - 0.4, x[:, 1].max().item() + 0.4
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    models = [
        ("Layer-Wise SPSA\n(0 Backprop)", p_lw, hist_lw["accuracy"][-1]),
        ("Holistic SPSA\n(0 Backprop)", p_hol, hist_hol["accuracy"][-1]),
        ("Pure Vanilla SGD\n(Autograd)", p_sgd, hist_sgd["accuracy"][-1]),
        ("SGD + Momentum\n(Autograd)", p_mom, hist_mom["accuracy"][-1]),
        ("Adam Optimizer\n(Autograd)", p_adam, hist_adam["accuracy"][-1]),
    ]

    for idx, (title, p_model, acc) in enumerate(models):
        ax = axes[idx]
        with torch.no_grad():
            preds_grid = forward_pass(grid_tensor, p_model, beta).squeeze().numpy()
        zz = preds_grid.reshape(xx.shape)

        cf = ax.contourf(xx, yy, zz, levels=30, cmap="coolwarm", alpha=0.65)
        ax.contour(xx, yy, zz, levels=[0.0], colors="black", linewidths=2.0)

        ax.scatter(x[labels == 0, 0], x[labels == 0, 1], c="#1f77b4", edgecolor="k", s=30, label="Class 0 (Outer Ring)")
        ax.scatter(x[labels == 1, 0], x[labels == 1, 1], c="#d62728", edgecolor="k", s=30, label="Class 1 (Inner Disc)")
        ax.set_title(f"{title}\nAccuracy: {acc:.1f}%", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Feature x1")
        if idx == 0:
            ax.set_ylabel("Feature x2")
            ax.legend(loc="lower left", fontsize=7.5)
        ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    fig2_path = os.path.join(save_dir, "part2_decision_boundaries.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Decision boundaries comparison saved to: {fig2_path}")

    # -------------------------------------------------------------
    # Summary Comparison Table
    # -------------------------------------------------------------
    print("\n" + "=" * 94)
    print(" SUMMARY COMPARISON TABLE: SPSA VS CONVENTIONAL OPTIMIZERS (ALL 5 METHODS)")
    print("=" * 94)
    print(f"{'Method':<20} | {'Backprop?':<12} | {'Init Loss':<10} | {'Final Loss':<11} | {'Accuracy':<9} | {'Runtime (s)':<11}")
    print("-" * 94)
    print(f"{'Layer-Wise SPSA':<20} | {'No (0 bp)':<12} | {hist_lw['loss'][0]:<10.2f} | {hist_lw['loss'][-1]:<11.4f} | {hist_lw['accuracy'][-1]:<8.1f}% | {hist_lw['runtime']:<11.2f}")
    print(f"{'Holistic SPSA':<20} | {'No (0 bp)':<12} | {hist_hol['loss'][0]:<10.2f} | {hist_hol['loss'][-1]:<11.4f} | {hist_hol['accuracy'][-1]:<8.1f}% | {hist_hol['runtime']:<11.2f}")
    print(f"{'Pure Vanilla SGD':<20} | {'Yes (Autograd)':<12} | {hist_sgd['loss'][0]:<10.2f} | {hist_sgd['loss'][-1]:<11.4f} | {hist_sgd['accuracy'][-1]:<8.1f}% | {hist_sgd['runtime']:<11.2f}")
    print(f"{'SGD (Momentum=0.9)':<20} | {'Yes (Autograd)':<12} | {hist_mom['loss'][0]:<10.2f} | {hist_mom['loss'][-1]:<11.4f} | {hist_mom['accuracy'][-1]:<8.1f}% | {hist_mom['runtime']:<11.2f}")
    print(f"{'Adam':<20} | {'Yes (Autograd)':<12} | {hist_adam['loss'][0]:<10.2f} | {hist_adam['loss'][-1]:<11.4f} | {hist_adam['accuracy'][-1]:<8.1f}% | {hist_adam['runtime']:<11.2f}")
    print("=" * 94)

    # -------------------------------------------------------------
    # Periodic Loss & Accuracy Table (Every 20 Epochs)
    # -------------------------------------------------------------
    interval = 20
    epoch_checkpoints = list(range(0, epochs + 1, interval))
    if epoch_checkpoints[-1] != epochs:
        epoch_checkpoints.append(epochs)

    csv_rows = ["Epoch,LayerWise_SPSA_Loss,LayerWise_SPSA_Acc,Holistic_SPSA_Loss,Holistic_SPSA_Acc,Pure_SGD_Loss,Pure_SGD_Acc,SGD_Momentum_Loss,SGD_Momentum_Acc,Adam_Loss,Adam_Acc\n"]

    print("\n" + "=" * 115)
    print(" LOSS & ACCURACY LOGGED EVERY 20 EPOCHS (ALL 5 OPTIMIZERS)")
    print("=" * 115)
    print(f"{'Epoch':<6} | {'Layer-Wise SPSA':<20} | {'Holistic SPSA':<20} | {'Pure SGD':<20} | {'SGD (Momentum)':<20} | {'Adam':<20}")
    print(f"{'':<6} | {'Loss':<9} {'Acc (%)':<9} | {'Loss':<9} {'Acc (%)':<9} | {'Loss':<9} {'Acc (%)':<9} | {'Loss':<9} {'Acc (%)':<9} | {'Loss':<9} {'Acc (%)':<9}")
    print("-" * 115)

    for ep in epoch_checkpoints:
        lw_l, lw_a = hist_lw['loss'][ep], hist_lw['accuracy'][ep]
        h_l, h_a = hist_hol['loss'][ep], hist_hol['accuracy'][ep]
        ps_l, ps_a = hist_sgd['loss'][ep], hist_sgd['accuracy'][ep]
        sm_l, sm_a = hist_mom['loss'][ep], hist_mom['accuracy'][ep]
        ad_l, ad_a = hist_adam['loss'][ep], hist_adam['accuracy'][ep]

        print(f"{ep:<6d} | {lw_l:<9.2f} {lw_a:<9.1f} | {h_l:<9.2f} {h_a:<9.1f} | {ps_l:<9.2f} {ps_a:<9.1f} | {sm_l:<9.2f} {sm_a:<9.1f} | {ad_l:<9.2f} {ad_a:<9.1f}")
        csv_rows.append(f"{ep},{lw_l:.4f},{lw_a:.2f},{h_l:.4f},{h_a:.2f},{ps_l:.4f},{ps_a:.2f},{sm_l:.4f},{sm_a:.2f},{ad_l:.4f},{ad_a:.2f}\n")

    print("=" * 115)

    # Save to CSV
    csv_path = os.path.join(save_dir, "loss_every_20_epochs.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.writelines(csv_rows)
    print(f"[CSV Saved] Periodic loss table saved to: {csv_path}")

    # -------------------------------------------------------------
    # Early Epoch Comparison (Paper Investigation)
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" EARLY EPOCH COMPARISON (Testing SPSA Initial Epoch Dynamics)")
    print("=" * 90)
    print(f"{'Epoch':<6} | {'Layer-Wise SPSA':<18} | {'Holistic SPSA':<18} | {'Pure SGD':<16} | {'SGD + Mom':<16} | {'Adam':<16}")
    print("-" * 90)
    for ep in [0, 1, 2, 3, 5, 10, 15, 20]:
        print(f"{ep:<6d} | {hist_lw['loss'][ep]:<18.2f} | {hist_hol['loss'][ep]:<18.2f} | {hist_sgd['loss'][ep]:<16.2f} | {hist_mom['loss'][ep]:<16.2f} | {hist_adam['loss'][ep]:<16.2f}")
    print("=" * 90)

    return {
        "layerwise": hist_lw,
        "holistic": hist_hol,
        "pure_sgd": hist_sgd,
        "sgd_momentum": hist_mom,
        "adam": hist_adam,
    }


if __name__ == "__main__":
    run_nn_benchmarks()
