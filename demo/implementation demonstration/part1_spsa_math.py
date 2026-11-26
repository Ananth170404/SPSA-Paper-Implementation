"""
Part 1: Mathematical Function Optimization on a Complex Function
================================================================
Comprehensive comparative evaluation of 4 optimizers on a challenging, non-convex
mathematical benchmark landscape:

Optimizers:
  1. Holistic SPSA (Spall 1992, 1998) - 2 function evals per step
  2. SPSA with Layer-Wise / Coordinate-Wise Update - 2p function evals per step
  3. Normal Analytical Gradient Descent (GD) - Exact analytical gradient
  4. GD with Momentum (Polyak Momentum, beta = 0.70) - Accelerates along ravines

Selected Complex Function:
  Coupled Non-Convex Cubic Polynomial with Cross-Coupling and Saddle Point:
      f(x, y) = x^3 + y^3 - 3*x*y + 1.0
  Properties:
    - Pure cubic degree (x^3, y^3); contains no quadratic x^2 + y^2 terms.
    - Non-linear cross-product coupling term (-3*x*y).
    - Unstable saddle point at (0, 0) with Hessian eigenvalues +3, -3.
    - Strict local/global attractor minimum at (1, 1) with f(1, 1) = 0.0.
    - Started from an asymmetric non-trivial coordinate (2.0, 0.3).
"""

import os
from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


class SPSAOptimizer:
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) Optimizer
    supporting both Holistic simultaneous perturbations and Layer-Wise / Coordinate-Wise updates.
    """

    def __init__(
        self,
        a: float = 0.05,
        c: float = 0.06,
        A: float = 12.0,
        alpha: float = 0.602,
        gamma: float = 0.101,
        max_grad_norm: float = 20.0,
        seed: int = 42,
    ):
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        loss_fn: Callable[[np.ndarray], float],
        theta_init: np.ndarray,
        iterations: int = 150,
    ) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        """
        Method 1: Holistic SPSA (Spall 1992/1998).
        Simultaneously perturbs all dimensions using a Rademacher random vector.
        Requires strictly 2 forward evaluations per iteration.
        """
        theta = np.array(theta_init, dtype=np.float64)
        history_loss = [float(loss_fn(theta))]
        history_theta = [theta.copy()]

        for k in range(iterations):
            # Decay sequences (Spall 1998 standard formula)
            ak = self.a / ((k + 1 + self.A) ** self.alpha)
            ck = self.c / ((k + 1) ** self.gamma)

            # Symmetric Bernoulli perturbation vector Delta in {-1, +1}^p
            delta = self.rng.choice([-1.0, 1.0], size=theta.shape)

            # 2 forward passes only
            theta_plus = theta + ck * delta
            theta_minus = theta - ck * delta

            y_plus = loss_fn(theta_plus)
            y_minus = loss_fn(theta_minus)

            # Simultaneous gradient approximation
            g_hat = (y_plus - y_minus) / (2.0 * ck * delta)

            # Gradient clipping for numerical stability
            gnorm = np.linalg.norm(g_hat)
            if gnorm > self.max_grad_norm:
                g_hat = g_hat * (self.max_grad_norm / gnorm)

            # Parameter update
            theta = theta - ak * g_hat

            history_loss.append(float(loss_fn(theta)))
            history_theta.append(theta.copy())

        return theta, history_loss, history_theta

    def optimize_layerwise(
        self,
        loss_fn: Callable[[np.ndarray], float],
        theta_init: np.ndarray,
        iterations: int = 150,
    ) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        """
        Method 2: SPSA with Layer-Wise / Coordinate-Wise Update.
        Perturbs one dimension/block at a time while freezing others.
        Requires 2p forward evaluations per iteration, eliminating cross-coordinate interference.
        """
        theta = np.array(theta_init, dtype=np.float64)
        history_loss = [float(loss_fn(theta))]
        history_theta = [theta.copy()]
        p = len(theta)

        for k in range(iterations):
            ak = self.a / ((k + 1 + self.A) ** self.alpha)
            ck = self.c / ((k + 1) ** self.gamma)

            for i in range(p):
                delta_i = float(self.rng.choice([-1.0, 1.0]))
                theta_p = theta.copy()
                theta_m = theta.copy()
                theta_p[i] += ck * delta_i
                theta_m[i] -= ck * delta_i

                yp = loss_fn(theta_p)
                ym = loss_fn(theta_m)

                ghat_i = (yp - ym) / (2.0 * ck * delta_i)
                if abs(ghat_i) > self.max_grad_norm:
                    ghat_i = np.sign(ghat_i) * self.max_grad_norm

                theta[i] -= ak * ghat_i

            history_loss.append(float(loss_fn(theta)))
            history_theta.append(theta.copy())

        return theta, history_loss, history_theta


# =====================================================================
# Analytical Gradient Descent Methods (Baseline Comparison)
# =====================================================================

def gradient_descent(
    loss_fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    theta_init: np.ndarray,
    lr: float = 0.03,
    iterations: int = 150,
    max_grad_norm: float = 20.0,
) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
    """Method 3: Standard Analytical Gradient Descent (GD)."""
    theta = np.array(theta_init, dtype=np.float64)
    history_loss = [float(loss_fn(theta))]
    history_theta = [theta.copy()]

    for _ in range(iterations):
        grad = grad_fn(theta)
        gnorm = np.linalg.norm(grad)
        if gnorm > max_grad_norm:
            grad = grad * (max_grad_norm / gnorm)
        theta = theta - lr * grad
        history_loss.append(float(loss_fn(theta)))
        history_theta.append(theta.copy())

    return theta, history_loss, history_theta


def gradient_descent_momentum(
    loss_fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    theta_init: np.ndarray,
    lr: float = 0.03,
    beta: float = 0.70,
    iterations: int = 150,
    max_grad_norm: float = 20.0,
) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
    """Method 4: Analytical Gradient Descent with Polyak Momentum."""
    theta = np.array(theta_init, dtype=np.float64)
    history_loss = [float(loss_fn(theta))]
    history_theta = [theta.copy()]
    v = np.zeros_like(theta)

    for _ in range(iterations):
        grad = grad_fn(theta)
        gnorm = np.linalg.norm(grad)
        if gnorm > max_grad_norm:
            grad = grad * (max_grad_norm / gnorm)
        v = beta * v + lr * grad
        theta = theta - v
        history_loss.append(float(loss_fn(theta)))
        history_theta.append(theta.copy())

    return theta, history_loss, history_theta


# =====================================================================
# Selected Complex Benchmark Function
# =====================================================================

def complex_cubic_func(theta: np.ndarray) -> float:
    """
    Coupled Non-Convex Cubic Polynomial:
        f(x, y) = x^3 + y^3 - 3*x*y + 1.0

    Landscape Characteristics:
      - Pure degree-3 polynomial in x and y (contains x^3 and y^3, no x^2 + y^2).
      - Cross-variable coupling term -3*x*y generates non-separable curvature.
      - Saddle point at (0, 0) with Hessian eigenvalues lambda_1 = +3, lambda_2 = -3.
      - Global attractor minimum at (1, 1) with f(1, 1) = 0.0.
    """
    x, y = theta[0], theta[1]
    return float(x ** 3 + y ** 3 - 3.0 * x * y + 1.0)


def complex_cubic_grad(theta: np.ndarray) -> np.ndarray:
    """
    Analytical gradient of the complex cubic polynomial:
        df/dx = 3*x^2 - 3*y
        df/dy = 3*y^2 - 3*x
    """
    x, y = theta[0], theta[1]
    return np.array([3.0 * (x ** 2) - 3.0 * y, 3.0 * (y ** 2) - 3.0 * x], dtype=np.float64)


# =====================================================================
# Execution & Visualization Routine
# =====================================================================

DEFAULT_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def run_math_benchmarks(save_dir: str = DEFAULT_FIG_DIR):
    """
    Executes all 4 optimizers on the complex cubic polynomial function and generates
    publication-quality trajectory contours and semi-log convergence curves.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Initial condition and parameters
    theta_init = np.array([2.0, 0.3], dtype=np.float64)
    target_min = np.array([1.0, 1.0], dtype=np.float64)
    saddle_pt = np.array([0.0, 0.0], dtype=np.float64)
    iters = 150

    print("=" * 90)
    print(" PART 1: OPTIMIZING COMPLEX CUBIC POLYNOMIAL ACROSS 4 OPTIMIZERS")
    print(" Function: f(x, y) = x^3 + y^3 - 3*x*y + 1.0")
    print(f" Starting Point: {theta_init} | Initial Loss: {complex_cubic_func(theta_init):.6f}")
    print(f" Target Minimum: {target_min} | Target Loss: 0.000000")
    print(f" Saddle Point:   {saddle_pt} | Saddle Loss: 1.000000")
    print("=" * 90)

    opt = SPSAOptimizer(a=0.05, c=0.06, A=12.0, alpha=0.602, gamma=0.101, max_grad_norm=20.0, seed=42)

    # 1. Holistic SPSA
    th_spsa, loss_spsa, traj_spsa = opt.optimize(complex_cubic_func, theta_init, iterations=iters)

    # 2. Layer-Wise SPSA (Coordinate-wise)
    th_lw, loss_lw, traj_lw = opt.optimize_layerwise(complex_cubic_func, theta_init, iterations=iters)

    # 3. Normal Analytical GD
    th_gd, loss_gd, traj_gd = gradient_descent(complex_cubic_func, complex_cubic_grad, theta_init, lr=0.03, iterations=iters)

    # 4. Analytical GD with Momentum
    th_mom, loss_mom, traj_mom = gradient_descent_momentum(
        complex_cubic_func, complex_cubic_grad, theta_init, lr=0.03, beta=0.70, iterations=iters
    )

    traj_spsa = np.array(traj_spsa)
    traj_lw = np.array(traj_lw)
    traj_gd = np.array(traj_gd)
    traj_mom = np.array(traj_mom)

    # -------------------------------------------------------------
    # Visualization: 2-Panel Figure
    # -------------------------------------------------------------
    fig, (ax_contour, ax_loss) = plt.subplots(1, 2, figsize=(15, 6.5))
    plt.subplots_adjust(wspace=0.25)

    # Panel 1: 2D Contour & Optimization Paths
    x_grid = np.linspace(-0.5, 2.5, 300)
    y_grid = np.linspace(-0.5, 2.5, 300)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.maximum(X ** 3 + Y ** 3 - 3.0 * X * Y + 1.0, 0.0)

    cs = ax_contour.contour(X, Y, Z, levels=35, cmap="viridis", alpha=0.65)
    ax_contour.clabel(cs, inline=True, fontsize=7.5, fmt="%.1f")

    # Plot paths
    ax_contour.plot(traj_spsa[:, 0], traj_spsa[:, 1], "r.-", label="1. Holistic SPSA (2 evals/step)", linewidth=1.6, markersize=3, alpha=0.85)
    ax_contour.plot(traj_lw[:, 0], traj_lw[:, 1], "m.-", label="2. SPSA with Layer-Wise Update (4 evals/step)", linewidth=1.6, markersize=3, alpha=0.85)
    ax_contour.plot(traj_gd[:, 0], traj_gd[:, 1], "b--", label="3. Normal GD (Analytic grad)", linewidth=2.0, alpha=0.85)
    ax_contour.plot(traj_mom[:, 0], traj_mom[:, 1], "g-", label="4. GD with Momentum (beta=0.70)", linewidth=2.2, alpha=0.90)

    # Key landmarks
    ax_contour.scatter([theta_init[0]], [theta_init[1]], color="lime", edgecolor="black", s=110, label="Start Point (2.0, 0.3)", zorder=6)
    ax_contour.scatter([saddle_pt[0]], [saddle_pt[1]], color="orange", marker="X", edgecolor="black", s=110, label="Saddle Point (0, 0)", zorder=6)
    ax_contour.scatter([target_min[0]], [target_min[1]], color="gold", marker="*", edgecolor="black", s=220, label="Global Min (1, 1)", zorder=6)

    ax_contour.set_title("Complex Cubic Polynomial: 2D Trajectories\nf(x, y) = x^3 + y^3 - 3xy + 1", fontsize=11.5, fontweight="bold")
    ax_contour.set_xlabel("x Coordinate", fontsize=10.5)
    ax_contour.set_ylabel("y Coordinate", fontsize=10.5)
    ax_contour.set_xlim(-0.3, 2.3)
    ax_contour.set_ylim(-0.3, 2.3)
    ax_contour.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    ax_contour.grid(True, linestyle=":", alpha=0.55)

    # Panel 2: Loss vs Iterations
    ax_loss.plot(loss_spsa, "r-", label="1. Holistic SPSA (Spall 1998)", linewidth=2.0)
    ax_loss.plot(loss_lw, "m-", label="2. SPSA with Layer-Wise Update", linewidth=2.0)
    ax_loss.plot(loss_gd, "b--", label="3. Normal GD (Exact grad)", linewidth=2.0)
    ax_loss.plot(loss_mom, "g-", label="4. GD with Momentum (Polyak)", linewidth=2.2)

    ax_loss.set_yscale("log")
    ax_loss.set_title("Convergence Speed Comparison: Loss vs Iterations", fontsize=11.5, fontweight="bold")
    ax_loss.set_xlabel("Iteration (k)", fontsize=10.5)
    ax_loss.set_ylabel("Objective Loss f(theta) [Log Scale]", fontsize=10.5)
    ax_loss.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_loss.legend(fontsize=9.0, framealpha=0.92)

    out_path = os.path.join(save_dir, "part1_spsa_math_convergence.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n[Figure Saved] High-resolution comparative plot saved to: {out_path}")

    # -------------------------------------------------------------
    # Summary Tables
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print(" PART 1: OPTIMIZATION SUMMARY ON COMPLEX CUBIC POLYNOMIAL (ALL 4 OPTIMIZERS)")
    print("=" * 95)
    print(f"{'Optimizer Method':<30} | {'Backprop?':<12} | {'Evals/Step':<11} | {'Final Loss':<12} | {'Final Coordinates'}")
    print("-" * 95)
    print(f"{'1. Holistic SPSA':<30} | {'No (0 bp)':<12} | {'2 forward':<11} | {loss_spsa[-1]:<12.6f} | ({th_spsa[0]:.4f}, {th_spsa[1]:.4f})")
    print(f"{'2. SPSA with Layer-Wise Update':<30} | {'No (0 bp)':<12} | {'4 forward':<11} | {loss_lw[-1]:<12.6f} | ({th_lw[0]:.4f}, {th_lw[1]:.4f})")
    print(f"{'3. Normal GD (Analytical)':<30} | {'Exact Grad':<12} | {'Exact Grad':<11} | {loss_gd[-1]:<12.6f} | ({th_gd[0]:.4f}, {th_gd[1]:.4f})")
    print(f"{'4. GD with Momentum (beta=0.70)':<30} | {'Exact Grad':<12} | {'Exact Grad':<11} | {loss_mom[-1]:<12.6f} | ({th_mom[0]:.4f}, {th_mom[1]:.4f})")
    print("=" * 95)

    # -------------------------------------------------------------
    # Interval Logging CSV (every 15 iterations)
    # -------------------------------------------------------------
    checkpoints = list(range(0, iters + 1, 15))
    csv_rows = ["Iteration,Holistic_SPSA_Loss,LayerWise_SPSA_Loss,Normal_GD_Loss,GD_Momentum_Loss\n"]

    print("\n" + "=" * 90)
    print(" PERIODIC CONVERGENCE PROGRESSION (EVERY 15 ITERATIONS)")
    print("=" * 90)
    print(f"{'Iteration':<10} | {'Holistic SPSA':<16} | {'Layer-Wise SPSA':<17} | {'Normal GD':<15} | {'GD + Momentum':<15}")
    print("-" * 90)

    for it in checkpoints:
        sl = loss_spsa[it]
        ll = loss_lw[it]
        gl = loss_gd[it]
        ml = loss_mom[it]
        print(f"{it:<10d} | {sl:<16.6f} | {ll:<17.6f} | {gl:<15.6f} | {ml:<15.6f}")
        csv_rows.append(f"{it},{sl:.6f},{ll:.6f},{gl:.6f},{ml:.6f}\n")
    print("=" * 90)

    csv_path = os.path.join(save_dir, "part1_math_convergence_intervals.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.writelines(csv_rows)
    print(f"[CSV Saved] Checkpoint logs saved to: {csv_path}")

    return {
        "holistic_spsa": {"final_theta": th_spsa, "final_loss": loss_spsa[-1], "loss_hist": loss_spsa},
        "layerwise_spsa": {"final_theta": th_lw, "final_loss": loss_lw[-1], "loss_hist": loss_lw},
        "normal_gd": {"final_theta": th_gd, "final_loss": loss_gd[-1], "loss_hist": loss_gd},
        "gd_momentum": {"final_theta": th_mom, "final_loss": loss_mom[-1], "loss_hist": loss_mom},
    }


if __name__ == "__main__":
    run_math_benchmarks()
