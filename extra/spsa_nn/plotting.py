"""Figures in the style of Fig. 1 and Fig. 2 of the paper."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: always write PNG files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from . import mlp  # noqa: E402

VARIANT_COLORS = {"bottom-up": "#3b4cc0", "top-down": "#f39c12", "random": "#1e8c3a", "holistic": "#d62728"}
CLASS_COLORS = ["#f5a623", "#3a5fcd", "#222222"]           # points
REGION_COLORS = ["#fde7bd", "#c3cfec", "#bdbdbd"]          # decision regions


@torch.no_grad()
def plot_decision_boundary(ax, x: torch.Tensor, labels: torch.Tensor, params: mlp.Params,
                           beta: float, n_classes: int, title: Optional[str] = None,
                           resolution: int = 300) -> None:
    """Decision regions of a single trained network (params with K = 1)."""
    xs = x.cpu().numpy()
    pad = 0.3
    x0 = np.linspace(xs[:, 0].min() - pad, xs[:, 0].max() + pad, resolution)
    x1 = np.linspace(xs[:, 1].min() - pad, xs[:, 1].max() + pad, resolution)
    g0, g1 = np.meshgrid(x0, x1)
    grid = torch.tensor(np.stack([g0.ravel(), g1.ravel()], axis=1), dtype=torch.float32)
    p_cpu = [p.detach().cpu() for p in params]
    pred = mlp.predict_classes(grid, p_cpu, beta)[0].numpy().reshape(g0.shape)
    ax.contourf(g0, g1, pred, levels=np.arange(n_classes + 1) - 0.5,
                cmap=ListedColormap(REGION_COLORS[:n_classes]), alpha=0.9)
    lab = labels.cpu().numpy()
    for k in range(n_classes):
        m = lab == k
        ax.scatter(xs[m, 0], xs[m, 1], s=12, c=CLASS_COLORS[k], alpha=0.75, edgecolors="none")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def plot_error_curves(ax, errors: Dict[str, Sequence], title: Optional[str] = None,
                      statistic: str = "mean") -> None:
    """errors[variant] has shape (epochs + 1, trials). Plots mean (or median) over trials."""
    for variant, err in errors.items():
        e = np.asarray(err, dtype=float)
        curve = np.nanmean(e, axis=1) if statistic == "mean" else np.nanmedian(e, axis=1)
        ax.plot(np.arange(len(curve)), curve, label=variant, lw=1.3,
                color=VARIANT_COLORS.get(variant))
    ax.set_xlabel("training epoch e")
    ax.set_ylabel(r"error $E(\mathcal{D}, \mathcal{W}_e)$")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)


def plot_mnist(spsa: Dict[str, list], gd: Optional[Dict[str, list]], path: str,
               split_at: Optional[int] = None, adam: Optional[Dict[str, list]] = None) -> None:
    """Fig. 2: training error and test accuracy, SPSA vs GD (and Adam), with a broken x-axis."""
    fig = plt.figure(figsize=(11, 4))
    keys = [("error", r"error $E(\mathcal{D}, \mathcal{W}_e)$", "MNIST training error"),
            ("test_acc", "accuracy", "MNIST classification accuracy")]
    n_spsa = len(spsa["error"]) - 1 if spsa else 12
    split = split_at or max(n_spsa + 3, 15)
    for i, (key, ylabel, title) in enumerate(keys):
        left = fig.add_subplot(1, 4, 2 * i + 1)
        right = fig.add_subplot(1, 4, 2 * i + 2, sharey=left)
        for name, hist, color in (("GD", gd, VARIANT_COLORS["top-down"]),
                                  ("Adam", adam, VARIANT_COLORS["random"]),
                                  ("SPSA", spsa, VARIANT_COLORS["bottom-up"])):
            if hist is None or key not in hist:
                continue
            v = np.nanmean(np.asarray(hist[key], dtype=float), axis=1)
            ep = np.arange(len(v))
            left.plot(ep[: split + 1], v[: split + 1], color=color, label=name, lw=1.2)
            if len(v) > split:
                right.plot(ep[split:], v[split:], color=color, lw=1.2)
        left.set_xlim(0, split)
        longest = max((len(h.get(key, [])) for h in (gd, adam) if h is not None), default=0)
        if longest > split + 1:
            right.set_xlim(split, longest - 1)
        else:
            right.set_visible(False)
        left.set_ylabel(ylabel)
        left.set_title(title, loc="left")
        left.set_xlabel("training epoch e")
        right.tick_params(labelleft=False)
        right.spines["left"].set_visible(False)
        left.spines["right"].set_visible(False)
        left.grid(alpha=0.3)
        right.grid(alpha=0.3)
        if i == 0:
            left.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
