"""Reproduce Fig. 1: xor / moons / circles with layer-wise vs holistic SPSA.

Examples
--------
    python experiments/run_didactic.py                      # all three, 100 trials (paper setting)
    python experiments/run_didactic.py --datasets xor       # just one
    python experiments/run_didactic.py --quick              # 10 trials, fewer epochs (smoke test)
    python experiments/run_didactic.py --datasets circles --a 0.005 --t-max 100

Outputs (in results/):
    didactic_<name>.json         config + per-epoch errors for every trial and variant
    didactic_<name>_params.pt    trained bottom-up networks (for the decision boundary)
    fig1_<name>.png              boundary + error curves for that data set
    fig1.png                     the full 2x3 figure (once all three data sets have been run)
"""
from __future__ import annotations

import argparse
import os
import copy
import time

from _common import environment_info, save_json  # noqa: F401  (sets sys.path)

import torch

from spsa_nn import datasets, layerwise, mlp
from spsa_nn.device import describe, get_device, seed_everything
from spsa_nn.paths import results_dir

# Per-data-set settings. Architecture, beta, labels, epochs and #trials are
# taken from Sect. 4.1 of the paper. The paper does not report a, c, alpha,
# gamma, t_max, the init or the number of samples; the values below were chosen
# so that the initial error matches Fig. 1 (N(0,1) init) and are documented in
# the README.
PRESETS = {
    "xor": dict(sizes=[2, 5, 5, 5, 1], beta=0.5, epochs=25, n=200,
                a=0.01, c=0.1, alpha=1.0, gamma=1 / 6, t_max=50),
    "moons": dict(sizes=[2, 5, 5, 5, 1], beta=0.5, epochs=25, n=200,
                  a=0.02, c=0.1, alpha=1.0, gamma=1 / 6, t_max=50),
    "circles": dict(sizes=[2, 10, 10, 10, 10, 10, 3], beta=0.25, epochs=500, n=120,
                    a=0.005, c=0.1, alpha=1.0, gamma=1 / 6, t_max=50),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(PRESETS), choices=list(PRESETS))
    p.add_argument("--variants", nargs="+", default=list(layerwise.ORDERS), choices=list(layerwise.ORDERS))
    p.add_argument("--trials", type=int, default=100, help="independent random initialisations (paper: 100)")
    p.add_argument("--epochs", type=int, default=None, help="override preset number of epochs")
    p.add_argument("--n", type=int, default=None, help="override number of data points")
    p.add_argument("--a", type=float, default=None)
    p.add_argument("--c", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--t-max", type=int, default=None)
    p.add_argument("--init", default="standard", choices=["standard", "glorot", "normal"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu | mps")
    p.add_argument("--double", action="store_true", help="use float64")
    p.add_argument("--quick", action="store_true", help="10 trials and 1/10 of the epochs")
    p.add_argument("--log-every", type=int, default=0, help="print progress every N epochs (0 = auto)")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def run_dataset(name: str, args: argparse.Namespace, device: torch.device) -> dict:
    preset = copy.deepcopy(PRESETS[name])
    for key in ("epochs", "n", "a", "c", "alpha", "gamma"):
        if getattr(args, key) is not None:
            preset[key] = getattr(args, key)
    if args.t_max is not None:
        preset["t_max"] = args.t_max
    trials = args.trials
    if args.quick:
        trials = min(trials, 10)
        preset["epochs"] = max(5, preset["epochs"] // 10)

    data = datasets.load_didactic(name, n=preset["n"], seed=args.data_seed)
    dtype = torch.float64 if args.double else torch.float32
    x, y = data.x.to(device, dtype), data.y.to(device, dtype)
    labels = data.labels.to(device)

    print(f"\n=== {name}: {'x'.join(map(str, preset['sizes']))} network, "
          f"{mlp.count_parameters(preset['sizes'])} parameters, n={len(x)}, {trials} trials, "
          f"{preset['epochs']} epochs, device {describe(device)}")
    log_every = args.log_every or max(1, preset["epochs"] // 5)

    out = {"dataset": name, "preset": preset, "trials": trials, "init": args.init,
           "seed": args.seed, "data_seed": args.data_seed, "env": environment_info(device),
           "variants": {}}
    for variant in args.variants:
        cfg = layerwise.TrainConfig(
            sizes=preset["sizes"], order=variant, epochs=preset["epochs"], beta=preset["beta"],
            n_trials=trials, reduction="sum", init=args.init, seed=args.seed,
            spsa=layerwise.SPSAConfig(a=preset["a"], c=preset["c"], alpha=preset["alpha"],
                                      gamma=preset["gamma"], t_max=preset["t_max"]),
        )
        t0 = time.perf_counter()
        params, hist = layerwise.train(
            x, y, cfg, device,
            eval_fn=lambda p: {"train_acc": mlp.accuracy(x, labels, p, cfg.beta)},
            log_every=log_every, log_prefix=f"[{name} {variant:9s}] ",
        )
        final = torch.tensor(hist["error"][-1])
        acc = torch.tensor(hist["train_acc"][-1])
        print(f"[{name} {variant:9s}] done in {time.perf_counter() - t0:.1f}s | final E mean "
              f"{final.mean():.2f} median {final.median():.2f} | train acc mean {acc.mean():.3f} | "
              f"trials with acc>=0.95: {(acc >= 0.95).float().mean() * 100:.0f}%")
        out["variants"][variant] = {"config": cfg.to_dict(), **hist}
        if variant == "bottom-up" or "bottom-up" not in args.variants and variant == args.variants[0]:
            torch.save({"variant": variant, "params": [p.cpu() for p in params],
                        "beta": cfg.beta, "final_error": final},
                       results_dir() / f"didactic_{name}_params.pt")

    save_json(out, results_dir() / f"didactic_{name}.json")
    return out


def main() -> None:
    args = parse_args()
    if args.quick:  # never overwrite full-protocol results
        os.environ["SPSA_RESULTS_DIR"] = str(results_dir() / "quick")
    seed_everything(args.seed)
    device = get_device(args.device)
    for name in args.datasets:
        run_dataset(name, args, device)
    if not args.no_plot:
        from plot_results import plot_didactic

        plot_didactic()


if __name__ == "__main__":
    main()
