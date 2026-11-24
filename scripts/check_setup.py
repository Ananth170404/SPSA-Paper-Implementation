"""Print the environment and run a 2-second SPSA smoke test on the selected device.

    python scripts/check_setup.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402

from spsa_nn import datasets, layerwise  # noqa: E402
from spsa_nn.device import describe, get_device  # noqa: E402

print(f"python {sys.version.split()[0]}, torch {torch.__version__}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}"
      + (f", CUDA {torch.version.cuda}" if torch.version.cuda else " (CPU-only torch build)"))
device = get_device("auto")
print(f"selected device: {describe(device)}")

data = datasets.make_xor()
cfg = layerwise.TrainConfig(sizes=[2, 5, 5, 5, 1], epochs=3, beta=0.5, n_trials=100, init="standard",
                            spsa=layerwise.SPSAConfig(a=0.01, c=0.1, t_max=50))
t0 = time.perf_counter()
_, hist = layerwise.train(data.x, data.y, cfg, device)
e0 = torch.tensor(hist["error"][0]).mean().item()
e1 = torch.tensor(hist["error"][-1]).mean().item()
print(f"smoke test: 100 xor networks, 3 epochs in {time.perf_counter() - t0:.1f}s, "
      f"mean error {e0:.1f} -> {e1:.1f}")
print("OK" if e1 < e0 else "WARNING: error did not decrease")
