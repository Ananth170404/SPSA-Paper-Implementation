"""Fast unit tests (CPU, < 1 minute). Run with:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spsa_nn import datasets, gd, layerwise, mlp, spsa  # noqa: E402
from spsa_nn.device import get_device  # noqa: E402

CPU = torch.device("cpu")


def test_algorithm1_minimises_quadratic():
    target = np.array([1.0, -2.0, 0.5, 3.0])
    f = lambda x: float(np.sum((x - target) ** 2))  # noqa: E731
    x = spsa.minimize(f, np.zeros(4), a=0.2, alpha=0.602, c=0.1, gamma=0.101, t_max=2000,
                      rng=np.random.default_rng(0))
    assert f(x) < 1e-2


def test_rademacher_values():
    d = spsa.rademacher((1000,), np.random.default_rng(0))
    assert set(np.unique(d)) == {-1.0, 1.0}
    assert abs(d.mean()) < 0.1


@pytest.mark.parametrize("sizes,expected", [([2, 5, 5, 5, 1], 81), ([2, 10, 10, 10, 10, 10, 3], 503),
                                            ([784, 196, 10], 155830),
                                            ([784, 784, 196, 10], 771270)])
def test_parameter_counts_match_paper(sizes, expected):
    assert mlp.count_parameters(sizes) == expected


def test_forward_shapes_and_cached_start_layer():
    g = torch.Generator().manual_seed(0)
    params = mlp.init_params([2, 4, 3, 2], n_trials=5, init="standard", generator=g)
    x = torch.randn(7, 2)
    y = mlp.forward(x, params, beta=0.5)
    assert y.shape == (5, 7, 2)
    a1 = mlp.activations_upto(x, params, 0.5, 1)
    assert torch.allclose(mlp.forward(a1, params, 0.5, start_layer=1), y, atol=1e-6)


def test_error_reductions():
    y_hat = torch.zeros(2, 4, 3)
    y = torch.ones(4, 3)
    assert torch.allclose(mlp.error(y_hat, y, "sum"), torch.tensor([12.0, 12.0]))
    assert torch.allclose(mlp.error(y_hat, y, "mean"), torch.tensor([3.0, 3.0]))
    assert torch.allclose(mlp.error(y_hat, y, "elementwise_mean"), torch.tensor([1.0, 1.0]))


@pytest.mark.parametrize("order", layerwise.ORDERS)
def test_schedule_covers_every_layer(order):
    g = torch.Generator().manual_seed(0)
    steps = layerwise.epoch_schedule(order, n_layers=4, n_trials=3, split_bias=False, generator=g)
    assert len(steps) == 4
    total = torch.stack(steps).sum(0)  # (P, K): how often each tensor is updated per trial
    if order == "holistic":
        assert (total == 4).all()
    else:
        assert (total == 1).all()


def test_split_bias_schedule_weights_then_biases():
    g = torch.Generator()
    steps = layerwise.epoch_schedule("bottom-up", 2, 1, split_bias=True, generator=g)
    active = [torch.nonzero(s[:, 0]).flatten().tolist() for s in steps]
    assert active == [[0], [1], [2], [3]]  # W1, b1, W2, b2


def test_spsa_call_only_changes_active_layer():
    g = torch.Generator().manual_seed(0)
    params = mlp.init_params([2, 5, 5, 1], n_trials=4, init="standard", generator=g)
    before = [p.clone() for p in params]
    data = datasets.make_xor(n=40)
    cfg = layerwise.TrainConfig(sizes=[2, 5, 5, 1], spsa=layerwise.SPSAConfig(a=0.01, t_max=5))
    mask = torch.zeros(6, 4, dtype=torch.bool)
    mask[2:4] = True  # layer 2 only
    layerwise.spsa_call(params, mask, data.x, data.y, cfg, g)
    for i in (0, 1, 4, 5):
        assert torch.equal(params[i], before[i])
    assert not torch.equal(params[2], before[2])


def test_spsa_call_per_trial_masks():
    """With the random order each trial may train a different layer in the same call."""
    g = torch.Generator().manual_seed(1)
    params = mlp.init_params([2, 3, 1], n_trials=2, init="standard", generator=g)
    before = [p.clone() for p in params]
    data = datasets.make_xor(n=40)
    cfg = layerwise.TrainConfig(sizes=[2, 3, 1], spsa=layerwise.SPSAConfig(a=0.01, t_max=3))
    mask = torch.zeros(4, 2, dtype=torch.bool)
    mask[0:2, 0] = True  # trial 0 trains layer 1
    mask[2:4, 1] = True  # trial 1 trains layer 2
    layerwise.spsa_call(params, mask, data.x, data.y, cfg, g)
    assert not torch.equal(params[0][0], before[0][0]) and torch.equal(params[0][1], before[0][1])
    assert torch.equal(params[2][0], before[2][0]) and not torch.equal(params[2][1], before[2][1])


def test_layerwise_training_reduces_xor_error():
    data = datasets.make_xor(n=200)
    cfg = layerwise.TrainConfig(sizes=[2, 5, 5, 5, 1], order="bottom-up", epochs=10, beta=0.5,
                                n_trials=8, init="standard",
                                spsa=layerwise.SPSAConfig(a=0.01, c=0.1, alpha=1.0, gamma=1 / 6, t_max=50))
    _, hist = layerwise.train(data.x, data.y, cfg, CPU)
    start = np.median(hist["error"][0])
    end = np.median(hist["error"][-1])
    assert end < 0.25 * start


def test_gd_baseline_runs():
    data = datasets.make_moons(n=100)
    cfg = gd.GDConfig(sizes=[2, 5, 1], lr=0.01, epochs=3, beta=0.5)
    _, hist = gd.train(data.x, data.y, cfg, CPU)
    assert len(hist["error"]) == 4


def test_adam_baseline_reduces_error():
    data = datasets.make_moons(n=100)
    cfg = gd.GDConfig(sizes=[2, 5, 1], lr=0.01, epochs=30, beta=0.5, optimizer="adam")
    _, hist = gd.train(data.x, data.y, cfg, CPU)
    assert len(hist["error"]) == 31 and hist["error"][-1][0] < hist["error"][0][0]
    with pytest.raises(ValueError):
        gd.train(data.x, data.y, gd.GDConfig(sizes=[2, 5, 1], epochs=1, optimizer="rmsprop"), CPU)


def test_datasets_shapes_and_labels():
    for name, m in (("xor", 1), ("moons", 1), ("circles", 3)):
        d = datasets.load_didactic(name)
        assert d.x.shape[1] == 2 and d.y.shape[1] == m
        assert set(torch.unique(d.y).tolist()) <= {-1.0, 1.0}


def test_device_auto():
    assert get_device("auto").type in ("cuda", "mps", "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_shapes():
    dev = torch.device("cuda")
    data = datasets.make_xor(n=64)
    cfg = layerwise.TrainConfig(sizes=[2, 5, 1], epochs=2, n_trials=3, init="standard",
                                spsa=layerwise.SPSAConfig(a=0.01, t_max=5))
    params, hist = layerwise.train(data.x, data.y, cfg, dev)
    assert params[0].device.type == "cuda" and len(hist["error"]) == 3
