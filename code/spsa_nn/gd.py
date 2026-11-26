"""Backpropagation baselines used for comparison in Fig. 2.

optimizer='sgd'  plain gradient descent, p <- p - lr * grad; with the default
                 full batch this mirrors ``tf.train.GradientDescentOptimizer``
                 as in the paper.
optimizer='adam' Adam (Kingma & Ba, 2015) with the usual defaults
                 (betas 0.9 / 0.999, eps 1e-8), via ``torch.optim.Adam``.

Both use the same network, initialisation and squared-residual objective
(mean over the batch) as SPSA.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import torch

from . import mlp
from .device import make_generator


@dataclass
class GDConfig:
    sizes: List[int]
    lr: float = 0.01
    epochs: int = 250
    batch_size: Optional[int] = None   # None = full-batch gradient descent
    optimizer: str = "sgd"             # 'sgd' (plain gradient descent) or 'adam'
    beta: float = 1.0
    init: str = "glorot"
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def train(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: GDConfig,
    device: torch.device,
    eval_fn: Optional[Callable[[mlp.Params], Dict[str, torch.Tensor]]] = None,
    log_every: int = 0,
    log_prefix: str = "",
) -> tuple[mlp.Params, Dict[str, list]]:
    x, y = x.to(device), y.to(device)
    gen = make_generator(cfg.seed, device)
    params = mlp.init_params(cfg.sizes, 1, cfg.init, device, x.dtype, gen)
    for p in params:
        p.requires_grad_(True)
    n = x.shape[0]
    bs = cfg.batch_size or n
    if cfg.optimizer == "adam":
        adam = torch.optim.Adam(params, lr=cfg.lr)
    elif cfg.optimizer != "sgd":
        raise ValueError(f"optimizer must be 'sgd' or 'adam', got {cfg.optimizer!r}")

    history: Dict[str, list] = {"error": [], "time": []}

    def record(elapsed: float) -> None:
        with torch.no_grad():
            history["error"].append(mlp.evaluate_error(x, y, params, cfg.beta, "mean").cpu().tolist())
            history["time"].append(elapsed)
            if eval_fn is not None:
                for k, v in eval_fn(params).items():
                    history.setdefault(k, []).append(v.cpu().tolist())

    t0 = time.perf_counter()
    record(0.0)
    for e in range(1, cfg.epochs + 1):
        perm = torch.randperm(n, generator=gen, device=device) if bs < n else None
        for i in range(0, n, bs):
            if perm is None:
                xb, yb = x, y
            else:
                idx = perm[i:i + bs]
                xb, yb = x[idx], y[idx]
            loss = mlp.error(mlp.forward(xb, params, cfg.beta), yb, "mean").sum()
            if cfg.optimizer == "adam":
                adam.zero_grad(set_to_none=True)
                loss.backward()
                adam.step()
            else:
                grads = torch.autograd.grad(loss, params)
                with torch.no_grad():
                    for p, g in zip(params, grads):
                        p.sub_(cfg.lr * g)
        record(time.perf_counter() - t0)
        if log_every and (e % log_every == 0 or e == cfg.epochs):
            msg = f"{log_prefix}epoch {e:4d}/{cfg.epochs}  E {history['error'][-1][0]:.4f}"
            for k in history:
                if k not in ("error", "time"):
                    msg += f"  {k} {history[k][-1][0]:.4f}"
            print(msg + f"  ({history['time'][-1]:.1f}s)", flush=True)
    return [p.detach() for p in params], history
