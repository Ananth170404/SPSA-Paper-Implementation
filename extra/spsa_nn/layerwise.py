"""Algorithm 2 of the paper: SPSA-based, layer-wise network training.

One *epoch* iterates over the layers of the network in a given order
(bottom-up, top-down or random, Eqs. 11-13). For every layer l the other weight
matrices are kept fixed and Algorithm 1 (SPSA, t = 1..t_max with gains reset to
a_1, c_1 -- the "warm restart" mentioned in the conclusion) optimises E(D, W)
with respect to W^l only.

'holistic' is the baseline that perturbs *all* parameters simultaneously. To
give it the same budget of forward passes as the layer-wise variants it runs L
SPSA calls of t_max iterations per epoch (configurable).

MNIST variant (Sect. 4.2): with ``split_bias=True`` each layer is handled as two
consecutive SPSA calls, first on its weights and then on its biases.

Implementation notes
--------------------
* Everything runs in PyTorch under ``torch.no_grad()`` -- only forward passes,
  no backpropagation. Works on CUDA, MPS or CPU.
* K independent trials (different random initialisations) are trained at once:
  every tensor has a leading trial dimension and every trial gets its own
  perturbation vector and its own scalar finite difference.
* Since layers below the one being optimised are fixed during an SPSA call,
  their output is computed once and cached; each SPSA iteration then only runs
  the forward pass from the active layer upwards.
* delta_i in {-1, +1} so 1/delta_i == delta_i, and the SPSA gradient estimate is
  g = (f+ - f-) / (2 c_t) * delta.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from . import mlp

ORDERS = ("bottom-up", "top-down", "random", "holistic")


@dataclass
class SPSAConfig:
    a: float = 0.1          # step-size gain a
    c: float = 0.1          # perturbation gain c
    alpha: float = 1.0      # a_t = a / t^alpha
    gamma: float = 1 / 6    # c_t = c / t^gamma
    t_max: int = 50         # SPSA iterations per call of Algorithm 1

    def gains(self, t: int) -> tuple[float, float]:
        return self.a / t**self.alpha, self.c / t**self.gamma


@dataclass
class TrainConfig:
    sizes: List[int]
    order: str = "bottom-up"
    epochs: int = 25
    beta: float = 0.5
    n_trials: int = 1
    reduction: str = "sum"            # objective: 'sum' = Eq. (9); 'mean' = /n; 'elementwise_mean' = /(n*m)
    report_reduction: Optional[str] = None  # scale of the logged training error (default: = reduction)
    split_bias: bool = False          # MNIST modification of Sect. 4.2
    batch_size: Optional[int] = None  # None: full data set E(D, W) in every evaluation
    holistic_calls_per_epoch: Optional[int] = None  # default: number of layers
    init: str = "glorot"
    seed: int = 0
    spsa: SPSAConfig = field(default_factory=SPSAConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _rademacher(shape, generator, device, dtype) -> torch.Tensor:
    return torch.randint(0, 2, shape, generator=generator, device=device).to(dtype).mul_(2).sub_(1)


def epoch_schedule(
    order: str, n_layers: int, n_trials: int, split_bias: bool,
    generator: torch.Generator, holistic_calls: Optional[int] = None,
) -> List[torch.Tensor]:
    """List of SPSA calls for one epoch.

    Each entry is a bool mask of shape (n_params, n_trials): which parameter
    tensors (W1, b1, W2, b2, ...) are perturbed/updated in that call, per trial.
    'random' draws an independent permutation per trial and per epoch.
    """
    P = 2 * n_layers
    groups = [[0, 1]] if not split_bias else [[0], [1]]  # offsets within a layer

    if order == "holistic":
        calls = holistic_calls or n_layers
        return [torch.ones(P, n_trials, dtype=torch.bool) for _ in range(calls)]

    if order == "bottom-up":
        perms = [list(range(n_layers))] * n_trials
    elif order == "top-down":
        perms = [list(range(n_layers - 1, -1, -1))] * n_trials
    elif order == "random":
        perms = [torch.randperm(n_layers, generator=generator).tolist() for _ in range(n_trials)]
    else:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")

    steps = []
    for s in range(n_layers):
        for g in groups:
            mask = torch.zeros(P, n_trials, dtype=torch.bool)
            for k in range(n_trials):
                for off in g:
                    mask[2 * perms[k][s] + off, k] = True
            steps.append(mask)
    return steps


@torch.no_grad()
def spsa_call(
    params: mlp.Params,
    active: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
    generator: torch.Generator,
) -> None:
    """One call of Algorithm 1 on the active parameter tensors (in place)."""
    K = params[0].shape[0]
    device, dtype = params[0].device, params[0].dtype
    active_idx = [p for p in range(len(params)) if bool(active[p].any())]
    start_layer = min(active_idx) // 2

    # Lower layers are fixed during this call -> cache their output once.
    # The + and - perturbations are evaluated together as 2K stacked networks,
    # so the cached input is duplicated along the trial axis when needed.
    a_in = mlp.activations_upto(x, params, cfg.beta, start_layer)
    if a_in.dim() == 3:
        a_in = torch.cat([a_in, a_in], dim=0)
    n = x.shape[0]
    fixed = {p: torch.cat([params[p], params[p]], dim=0)
             for p in range(start_layer * 2, len(params)) if p not in active_idx}

    # Per-trial masks broadcastable to each tensor (trials that do not train
    # this tensor in the current call get delta = 0, i.e. no change).
    masks = {}
    for p in active_idx:
        m = active[p].to(device=device, dtype=dtype)
        masks[p] = None if bool(active[p].all()) else m.view(K, *([1] * (params[p].dim() - 1)))

    for t in range(1, cfg.spsa.t_max + 1):
        a_t, c_t = cfg.spsa.gains(t)

        if cfg.batch_size is not None and cfg.batch_size < n:
            idx = torch.randint(0, n, (cfg.batch_size,), generator=generator, device=device)
            xb = a_in.index_select(-2, idx)
            yb = y.index_select(0, idx)
        else:
            xb, yb = a_in, y

        deltas = {}
        both = list(params)  # entries below start_layer are never touched
        for p in range(start_layer * 2, len(params)):
            if p in fixed:
                both[p] = fixed[p]
                continue
            d = _rademacher(params[p].shape, generator, device, dtype)
            if masks[p] is not None:
                d.mul_(masks[p])
            deltas[p] = d
            cd = c_t * d
            both[p] = torch.cat([params[p] + cd, params[p] - cd], dim=0)  # [W+, W-]

        f_both = mlp.error(mlp.forward(xb, both, cfg.beta, start_layer), yb, cfg.reduction)
        f_plus, f_minus = f_both[:K], f_both[K:]
        g_scale = (f_plus - f_minus) / (2.0 * c_t)          # (K,)
        g_scale = torch.nan_to_num(g_scale, nan=0.0, posinf=0.0, neginf=0.0)

        for p in active_idx:
            step = (a_t * g_scale).view(K, *([1] * (params[p].dim() - 1)))
            params[p].sub_(step * deltas[p])                # x <- x - a_t * g_hat


EvalFn = Callable[[mlp.Params], Dict[str, torch.Tensor]]


@torch.no_grad()
def train(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    eval_fn: Optional[EvalFn] = None,
    params: Optional[mlp.Params] = None,
    log_every: int = 0,
    log_prefix: str = "",
) -> tuple[mlp.Params, Dict[str, list]]:
    """Algorithm 2. Returns final params and a history dict.

    history['error'] has epochs+1 entries (entry 0 = initialisation), each a
    list with one training error E(D, W_e) per trial (with cfg.reduction).
    ``eval_fn(params)`` may add further per-epoch metrics (e.g. test accuracy).
    """
    from .device import make_generator

    x = x.to(device)
    y = y.to(device)
    gen = make_generator(cfg.seed, device)
    cpu_gen = torch.Generator().manual_seed(cfg.seed + 1)
    if params is None:
        params = mlp.init_params(cfg.sizes, cfg.n_trials, cfg.init, device, x.dtype, gen)
    L = mlp.n_layers(params)

    history: Dict[str, list] = {"error": [], "time": []}

    def record(elapsed: float) -> None:
        rep = cfg.report_reduction or cfg.reduction
        history["error"].append(mlp.evaluate_error(x, y, params, cfg.beta, rep).cpu().tolist())
        history["time"].append(elapsed)
        if eval_fn is not None:
            for k, v in eval_fn(params).items():
                history.setdefault(k, []).append(v.cpu().tolist() if torch.is_tensor(v) else v)

    t0 = time.perf_counter()
    record(0.0)
    for e in range(1, cfg.epochs + 1):
        for active in epoch_schedule(cfg.order, L, cfg.n_trials, cfg.split_bias, cpu_gen,
                                     cfg.holistic_calls_per_epoch):
            spsa_call(params, active, x, y, cfg, gen)
        record(time.perf_counter() - t0)
        if log_every and (e % log_every == 0 or e == cfg.epochs):
            err = torch.tensor(history["error"][-1])
            msg = f"{log_prefix}epoch {e:4d}/{cfg.epochs}  E mean {err.mean():9.4f}  median {err.median():9.4f}"
            for k in history:
                if k not in ("error", "time"):
                    msg += f"  {k} {torch.tensor(history[k][-1]).float().mean():.4f}"
            msg += f"  ({history['time'][-1]:.1f}s)"
            print(msg, flush=True)
    return params, history
