"""Minimal use of the library API: train one XOR network with layer-wise SPSA.

    python examples/layerwise_xor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spsa_nn import datasets, get_device, layerwise, mlp  # noqa: E402

device = get_device("auto")
data = datasets.make_xor(n=200)
cfg = layerwise.TrainConfig(
    sizes=[2, 5, 5, 5, 1], order="bottom-up", epochs=25, beta=0.5, n_trials=1, init="standard", seed=3,
    spsa=layerwise.SPSAConfig(a=0.01, c=0.1, alpha=1.0, gamma=1 / 6, t_max=50),
)
params, hist = layerwise.train(data.x, data.y, cfg, device, log_every=5)
acc = mlp.accuracy(data.x.to(device), data.labels.to(device), params, cfg.beta)
print(f"final error {hist['error'][-1][0]:.3f}, training accuracy {acc.item():.3f}")
