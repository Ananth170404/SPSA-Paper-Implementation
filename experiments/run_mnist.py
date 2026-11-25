"""Reproduce Fig. 2: MNIST, layer-wise SPSA vs gradient descent, plus an Adam baseline.

Network: 784x784x196x10 tanh network = 771,270 parameters, the count the paper states
(p. 570) and its "close to a million parameters" claims. The paper's prose topology
"784x196x10" (155,830 parameters) is available with --hidden 196.

Paper settings (Sect. 4.2): bottom-up order, weights and biases of every layer
updated by two separate SPSA calls, a = 0.1, c = 0.1, t_max = 50, tanh with
beta = 1, one-hot targets in {0, 1}^10. GD baseline: learning rate 0.01.

Examples
--------
    python experiments/run_mnist.py                    # SPSA (3 runs) + GD, full paper protocol
    python experiments/run_mnist.py --skip-gd --skip-adam   # only SPSA
    python experiments/run_mnist.py --quick            # 10k training images, 3 SPSA / 20 GD epochs
    python experiments/run_mnist.py --device cpu

Outputs (in results/): <name>.json and fig2[_<name>].png  (default name: mnist)

Ablations from the text of Sect. 4.2:
    python experiments/run_mnist.py --order holistic --skip-gd --name mnist_holistic
    python experiments/run_mnist.py --no-split-bias --skip-gd --name mnist_no_split_bias

Literal text topology 784x196x10 (155,830 parameters):
    python experiments/run_mnist.py --hidden 196 --name mnist_155k
"""
from __future__ import annotations

import argparse
import os
import time

from _common import environment_info, save_json  # noqa: F401  (sets sys.path)

import torch

from spsa_nn import datasets, gd, layerwise, mlp
from spsa_nn.device import describe, get_device, seed_everything
from spsa_nn.paths import results_dir



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # SPSA (paper values; alpha/gamma are not given in the paper -> Spall's standard 0.602 / 0.101)
    p.add_argument("--a", type=float, default=0.1)
    p.add_argument("--c", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.602)
    p.add_argument("--gamma", type=float, default=0.101)
    p.add_argument("--t-max", type=int, default=50)
    p.add_argument("--epochs", type=int, default=12, help="SPSA epochs (paper plots 12)")
    p.add_argument("--runs", type=int, default=3, help="independent SPSA runs, trained in parallel")
    p.add_argument("--order", default="bottom-up", choices=list(layerwise.ORDERS))
    p.add_argument("--no-split-bias", action="store_true",
                   help="update weights and biases of a layer jointly (paper: did not learn)")
    p.add_argument("--objective", default="elementwise_mean", choices=["elementwise_mean", "mean", "sum"],
                   help="scale of E used inside SPSA; elementwise_mean = tf.reduce_mean of squared residuals")
    p.add_argument("--batch-size", type=int, default=None,
                   help="evaluate E on a random mini-batch per SPSA iteration (default: full training set)")
    # GD baseline
    p.add_argument("--gd-lr", type=float, default=0.01)
    p.add_argument("--gd-epochs", type=int, default=250)
    p.add_argument("--gd-batch-size", type=int, default=None, help="default: full-batch GD")
    p.add_argument("--skip-gd", action="store_true")
    # Adam baseline (not in the paper; added as a second conventional optimizer)
    p.add_argument("--adam-lr", type=float, default=0.0003,
                   help="Adam learning rate; 3e-4 (1e-3 collapses to a single predicted class on the 771k network)")
    p.add_argument("--adam-epochs", type=int, default=None, help="default: same as --gd-epochs")
    p.add_argument("--adam-batch-size", type=int, default=None, help="default: full batch, like the GD baseline")
    p.add_argument("--skip-adam", action="store_true")
    p.add_argument("--skip-spsa", action="store_true")
    # general
    p.add_argument("--hidden", type=int, nargs="+", default=[784, 196],
                   help="hidden layer sizes; default '784 196' -> 771,270 params (paper's count), "
                        "'196' -> 155,830 params (paper's prose topology)")
    p.add_argument("--init", default="glorot", choices=["glorot", "normal", "standard"])
    p.add_argument("--n-train", type=int, default=None, help="use only the first N training images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu | mps")
    p.add_argument("--quick", action="store_true", help="10k train images, 3 SPSA epochs, 20 GD epochs")
    p.add_argument("--name", default="mnist", help="output file stem in results/")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:  # never overwrite full-protocol results
        os.environ["SPSA_RESULTS_DIR"] = str(results_dir() / "quick")
    if args.quick:
        args.n_train = args.n_train or 10000
        args.epochs = min(args.epochs, 3)
        args.gd_epochs = min(args.gd_epochs, 20)
        args.adam_epochs = min(args.adam_epochs or args.gd_epochs, 20)
    seed_everything(args.seed)
    device = get_device(args.device)
    SIZES = [784, *args.hidden, 10]

    train, test = datasets.load_mnist(n_train=args.n_train)
    x_tr, y_tr = train.x.to(device), train.y.to(device)
    x_te, lab_te = test.x.to(device), test.labels.to(device)
    lab_tr = train.labels.to(device)
    print(f"MNIST: {len(x_tr)} train / {len(x_te)} test, network {'x'.join(map(str, SIZES))} "
          f"({mlp.count_parameters(SIZES)} parameters), device {describe(device)}")

    def eval_fn(params):
        return {"test_acc": mlp.accuracy(x_te, lab_te, params, 1.0),
                "train_acc": mlp.accuracy(x_tr, lab_tr, params, 1.0)}

    out = {"args": vars(args), "env": environment_info(device)}

    if not args.skip_spsa:
        cfg = layerwise.TrainConfig(
            sizes=SIZES, order=args.order, epochs=args.epochs, beta=1.0, n_trials=args.runs,
            reduction=args.objective, report_reduction="mean", split_bias=not args.no_split_bias,
            batch_size=args.batch_size, init=args.init, seed=args.seed,
            spsa=layerwise.SPSAConfig(a=args.a, c=args.c, alpha=args.alpha, gamma=args.gamma,
                                      t_max=args.t_max),
        )
        print(f"\n--- SPSA ({args.runs} runs in parallel) ---")
        t0 = time.perf_counter()
        _, hist = layerwise.train(x_tr, y_tr, cfg, device, eval_fn=eval_fn, log_every=1, log_prefix="[SPSA] ")
        acc = torch.tensor(hist["test_acc"][-1])
        print(f"[SPSA] {time.perf_counter() - t0:.1f}s, final test accuracy per run: "
              f"{[round(v, 4) for v in acc.tolist()]}")
        out["spsa"] = {"config": cfg.to_dict(), **hist}
        save_json(out, results_dir() / f"{args.name}.json")

    if not args.skip_gd:
        gcfg = gd.GDConfig(sizes=SIZES, lr=args.gd_lr, epochs=args.gd_epochs,
                           batch_size=args.gd_batch_size, beta=1.0, init=args.init, seed=args.seed)
        print("\n--- Gradient descent baseline ---")
        _, ghist = gd.train(x_tr, y_tr, gcfg, device, eval_fn=eval_fn,
                            log_every=max(1, args.gd_epochs // 10), log_prefix="[GD] ")
        out["gd"] = {"config": gcfg.to_dict(), **ghist}

    if not args.skip_adam:
        acfg = gd.GDConfig(sizes=SIZES, lr=args.adam_lr, epochs=args.adam_epochs or args.gd_epochs,
                           batch_size=args.adam_batch_size, optimizer="adam", beta=1.0, init=args.init,
                           seed=args.seed)
        print("\n--- Adam baseline ---")
        _, ahist = gd.train(x_tr, y_tr, acfg, device, eval_fn=eval_fn,
                            log_every=max(1, acfg.epochs // 10), log_prefix="[Adam] ")
        out["adam"] = {"config": acfg.to_dict(), **ahist}

    if not any(k in out for k in ("spsa", "gd", "adam")):
        return
    # keep results of a previous run of the other method if only one was re-run
    prev_path = results_dir() / f"{args.name}.json"
    if prev_path.exists() and (args.skip_gd or args.skip_spsa or args.skip_adam):
        from _common import load_json

        prev = load_json(prev_path)
        for k in ("spsa", "gd", "adam"):
            if k not in out and k in prev:
                out[k] = prev[k]
    save_json(out, prev_path)

    if not args.no_plot:
        from plot_results import plot_mnist_results

        plot_mnist_results(args.name)


if __name__ == "__main__":
    main()
