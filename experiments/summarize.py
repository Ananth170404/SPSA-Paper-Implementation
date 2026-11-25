"""Print (and optionally write) a Markdown summary of all saved results.

    python experiments/summarize.py                      # reads results/
    python experiments/summarize.py --dir paper_results  # reads the shipped reference results
    python experiments/summarize.py --out SUMMARY.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import ROOT, load_json

import numpy as np


def didactic_table(d: Path) -> list[str]:
    lines = ["| data set | variant | trials | epochs | E start (mean) | E final (mean) | E final (median) "
             "| train acc (mean) | trials >= 95 % acc | time |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("xor", "moons", "circles"):
        p = d / f"didactic_{name}.json"
        if not p.exists():
            continue
        res = load_json(p)
        for variant, h in res["variants"].items():
            e = np.asarray(h["error"], float)
            acc = np.asarray(h["train_acc"][-1], float)
            lines.append(f"| {name} | {variant} | {res['trials']} | {len(e) - 1} | {e[0].mean():.1f} | "
                         f"{e[-1].mean():.2f} | {np.median(e[-1]):.2f} | {acc.mean():.3f} | "
                         f"{(acc >= 0.95).mean() * 100:.0f} % | {h['time'][-1]:.0f} s |")
    return lines


def mnist_table(d: Path) -> list[str]:
    lines = ["| run | network (params) | method | order | split W/b | epochs | E start -> final (mean) | test acc at epoch 12 "
             "(per run) | final test acc (mean) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(d.glob("mnist*.json")):
        res = load_json(p)
        sizes = [784, *res.get("args", {}).get("hidden", [784, 196]), 10]
        n_params = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
        net = f"{'x'.join(map(str, sizes))} ({n_params:,})"
        for key, label in (("spsa", "SPSA"), ("gd", "GD"), ("adam", "Adam")):
            if key not in res:
                continue
            h, cfg = res[key], res[key]["config"]
            e = np.asarray(h["error"], float)
            acc = np.asarray(h["test_acc"], float)
            at12 = acc[min(12, len(acc) - 1)]
            order = cfg.get("order", "-")
            split = ("yes" if cfg.get("split_bias") else "no") if key == "spsa" else "-"
            lines.append(f"| {p.stem} | {net} | {label} | {order} | {split} | {len(e) - 1} | "
                         f"{e[0].mean():.2f} -> {e[-1].mean():.2f} | "
                         f"{', '.join(f'{v:.3f}' for v in at12)} | {acc[-1].mean():.3f} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    d = args.dir if args.dir.is_absolute() else ROOT / args.dir
    text = "\n".join(["## Fig. 1 - didactic problems", "", *didactic_table(d), "",
                      "## Fig. 2 and Sect. 4.2 - MNIST", "", *mnist_table(d), ""])
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
