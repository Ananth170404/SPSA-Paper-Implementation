"""Re-create the figures from saved results (no training).

    python experiments/plot_results.py            # fig1*.png and fig2.png from results/*.json
"""
from __future__ import annotations

from _common import load_json  # noqa: F401  (sets sys.path)

import matplotlib.pyplot as plt
import torch

from spsa_nn import datasets
from spsa_nn.paths import results_dir
from spsa_nn.plotting import plot_decision_boundary, plot_error_curves, plot_mnist

TITLES = {"xor": "xor", "moons": "moons", "circles": "circles"}


def _boundary(ax, name: str, res: dict) -> bool:
    ppath = results_dir() / f"didactic_{name}_params.pt"
    if not ppath.exists():
        ax.set_visible(False)
        return False
    saved = torch.load(ppath, map_location="cpu")
    best = int(torch.argmin(saved["final_error"]))           # best trial of that variant
    params = [p[best:best + 1] for p in saved["params"]]
    data = datasets.load_didactic(name, n=res["preset"]["n"], seed=res.get("data_seed", 0))
    plot_decision_boundary(ax, data.x, data.labels, params, saved["beta"], data.n_classes,
                           title=f"{TITLES[name]} data and classifier ({saved['variant']})")
    return True


def plot_didactic() -> None:
    found = {}
    for name in ("xor", "moons", "circles"):
        path = results_dir() / f"didactic_{name}.json"
        if path.exists():
            found[name] = load_json(path)

    for name, res in found.items():
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        _boundary(axes[0], name, res)
        plot_error_curves(axes[1], {v: h["error"] for v, h in res["variants"].items()},
                          title=f"{name} training error (mean over {res['trials']} trials)")
        fig.tight_layout()
        out = results_dir() / f"fig1_{name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")

    if len(found) == 3:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
        for j, (name, res) in enumerate(found.items()):
            _boundary(axes[0, j], name, res)
            plot_error_curves(axes[1, j], {v: h["error"] for v, h in res["variants"].items()},
                              title=f"({'def'[j]}) {name} training error")
        fig.tight_layout()
        out = results_dir() / "fig1.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")


def plot_mnist_results(name: str = "mnist") -> None:
    path = results_dir() / f"{name}.json"
    if not path.exists():
        return
    res = load_json(path)
    out = results_dir() / ("fig2.png" if name == "mnist" else f"fig2_{name}.png")
    plot_mnist(res.get("spsa"), res.get("gd"), str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    plot_didactic()
    for p in sorted(results_dir().glob("mnist*.json")):
        plot_mnist_results(p.stem)
