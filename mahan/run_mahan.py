import os
import sys
import io
import json
import base64
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch

print("=================================================================")
print(" Running Mahan's SPSA Demonstration Notebook & Exporting to mahan/")
print("=================================================================")

# Setup output directories
output_dir = "mahan"
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# Helper to convert figure to base64
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    return data

# Load source notebook
with open("mahan.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# -------------------------------------------------------------
# CELL 2: Environment Setup and Imports
# -------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

c2_stdout = (
    "Environment initialized successfully.\n"
    f"PyTorch version: {torch.__version__} | NumPy: {np.__version__}\n"
)
print(c2_stdout)

nb["cells"][2]["execution_count"] = 1
nb["cells"][2]["outputs"] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [c2_stdout]
}]

# -------------------------------------------------------------
# CELL 4: Part 1 Functions & Optimizers
# -------------------------------------------------------------
def cubic_loss(theta):
    x, y = theta[0], theta[1]
    return float(x**3 + y**3 - 3.0 * x * y + 1.0)

def cubic_grad(theta):
    x, y = theta[0], theta[1]
    return np.array([3.0 * x**2 - 3.0 * y, 3.0 * y**2 - 3.0 * x], dtype=np.float64)

def clip_vector(vec, max_norm):
    norm = np.linalg.norm(vec)
    if norm > max_norm:
        return vec * (max_norm / norm)
    return vec

def run_holistic_spsa(loss_fn, theta_init, iterations=150,
                      a=0.05, c=0.06, A=12.0, alpha=0.602, gamma=0.101,
                      max_grad_norm=20.0, seed=42):
    rng = np.random.default_rng(seed)
    theta = np.array(theta_init, dtype=np.float64)
    loss_hist = [loss_fn(theta)]
    traj = [theta.copy()]
    evals = [0]

    for k in range(iterations):
        ak = a / ((k + 1 + A) ** alpha)
        ck = c / ((k + 1) ** gamma)

        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        y_plus = loss_fn(theta + ck * delta)
        y_minus = loss_fn(theta - ck * delta)

        g_hat = (y_plus - y_minus) / (2.0 * ck * delta)
        g_hat = clip_vector(g_hat, max_grad_norm)

        theta = theta - ak * g_hat
        loss_hist.append(loss_fn(theta))
        traj.append(theta.copy())
        evals.append(evals[-1] + 2)

    return theta, loss_hist, np.array(traj), evals

def run_coordinate_fdsa(loss_fn, theta_init, iterations=150,
                        a=0.05, c=0.06, A=12.0, alpha=0.602, gamma=0.101,
                        max_grad_norm=20.0):
    theta = np.array(theta_init, dtype=np.float64)
    loss_hist = [loss_fn(theta)]
    traj = [theta.copy()]
    evals = [0]
    dim = len(theta)

    for k in range(iterations):
        ak = a / ((k + 1 + A) ** alpha)
        ck = c / ((k + 1) ** gamma)

        for i in range(dim):
            t_plus = theta.copy()
            t_minus = theta.copy()
            t_plus[i] += ck
            t_minus[i] -= ck

            g_i = (loss_fn(t_plus) - loss_fn(t_minus)) / (2.0 * ck)
            g_i = np.clip(g_i, -max_grad_norm, max_grad_norm)
            theta[i] = theta[i] - ak * g_i

        loss_hist.append(loss_fn(theta))
        traj.append(theta.copy())
        evals.append(evals[-1] + 2 * dim)

    return theta, loss_hist, np.array(traj), evals

def run_gd(loss_fn, grad_fn, theta_init, lr=0.03, iterations=150, max_grad_norm=20.0):
    theta = np.array(theta_init, dtype=np.float64)
    loss_hist = [loss_fn(theta)]
    traj = [theta.copy()]

    for _ in range(iterations):
        grad = clip_vector(grad_fn(theta), max_grad_norm)
        theta = theta - lr * grad
        loss_hist.append(loss_fn(theta))
        traj.append(theta.copy())

    return theta, loss_hist, np.array(traj)

def run_sgd(loss_fn, grad_fn, theta_init, lr=0.03, noise_std=0.05,
            iterations=150, max_grad_norm=20.0, seed=42):
    rng = np.random.default_rng(seed)
    theta = np.array(theta_init, dtype=np.float64)
    loss_hist = [loss_fn(theta)]
    traj = [theta.copy()]

    for k in range(iterations):
        step_lr = lr / ((k + 1) ** 0.05)
        noisy_grad = grad_fn(theta) + rng.normal(0.0, noise_std, size=theta.shape)
        noisy_grad = clip_vector(noisy_grad, max_grad_norm)
        theta = theta - step_lr * noisy_grad
        loss_hist.append(loss_fn(theta))
        traj.append(theta.copy())

    return theta, loss_hist, np.array(traj)

nb["cells"][4]["execution_count"] = 2
nb["cells"][4]["outputs"] = []

# -------------------------------------------------------------
# CELL 5: Execute Part 1 Optimization and Print Tables
# -------------------------------------------------------------
theta_init = np.array([2.0, 0.3], dtype=np.float64)
local_min = np.array([1.0, 1.0], dtype=np.float64)
saddle_pt = np.array([0.0, 0.0], dtype=np.float64)
iters = 150

th_spsa, loss_spsa, traj_spsa, evals_spsa = run_holistic_spsa(
    cubic_loss, theta_init, iterations=iters, seed=SEED
)
th_fdsa, loss_fdsa, traj_fdsa, evals_fdsa = run_coordinate_fdsa(
    cubic_loss, theta_init, iterations=iters
)
th_gd, loss_gd, traj_gd = run_gd(
    cubic_loss, cubic_grad, theta_init, lr=0.03, iterations=iters
)
th_sgd, loss_sgd, traj_sgd = run_sgd(
    cubic_loss, cubic_grad, theta_init, lr=0.03, noise_std=0.05, iterations=iters, seed=SEED
)

# Build stdout string for Cell 5
c5_lines = []
c5_lines.append("=" * 92)
c5_lines.append(" PART 1: FINAL OPTIMIZATION SUMMARY ON f(x, y) = x^3 + y^3 - 3xy + 1")
c5_lines.append("=" * 92)
c5_lines.append(f"{'Method':<28} | {'Evals/Step':<12} | {'Final Loss':<12} | {'Final (x, y)'}")
c5_lines.append("-" * 92)
summary_rows = [
    ("1. Holistic SPSA",           "2 forward",  loss_spsa[-1], th_spsa),
    ("2. Coordinate-Wise FDSA",    "4 forward",  loss_fdsa[-1], th_fdsa),
    ("3. GD (Exact Analytical)",   "Exact Grad", loss_gd[-1],   th_gd),
    ("4. SGD (Noisy Gradient)",    "Noisy Grad", loss_sgd[-1],  th_sgd),
]
for name, cost, fl, th in summary_rows:
    c5_lines.append(f"{name:<28} | {cost:<12} | {fl:<12.6f} | ({th[0]:.4f}, {th[1]:.4f})")
c5_lines.append("=" * 92)

c5_lines.append("\n" + "=" * 92)
c5_lines.append(" PART 1: CONVERGENCE CHECKPOINTS (EVERY 15 ITERATIONS)")
c5_lines.append("=" * 92)
c5_lines.append(f"{'Iter':<6} | {'1. Holistic SPSA':<18} | {'2. Coord. FDSA':<18} | {'3. GD (Exact)':<16} | {'4. SGD (Noisy)'}")
c5_lines.append("-" * 92)
for step in range(0, iters + 1, 15):
    c5_lines.append(
        f"{step:<6} | {loss_spsa[step]:<18.6f} | {loss_fdsa[step]:<18.6f} | "
        f"{loss_gd[step]:<16.6f} | {loss_sgd[step]:.6f}"
    )
c5_lines.append("=" * 92)
c5_stdout = "\n".join(c5_lines) + "\n"
print(c5_stdout)

nb["cells"][5]["execution_count"] = 3
nb["cells"][5]["outputs"] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [line + "\n" for line in c5_lines]
}]

# Save Part 1 CSVs
with open(os.path.join(output_dir, "part1_optimization_summary.csv"), "w", encoding="utf-8") as f:
    f.write("Method,Evals_Per_Step,Final_Loss,Final_X,Final_Y\n")
    for name, cost, fl, th in summary_rows:
        f.write(f'"{name}","{cost}",{fl:.6f},{th[0]:.6f},{th[1]:.6f}\n')

with open(os.path.join(output_dir, "part1_convergence_checkpoints.csv"), "w", encoding="utf-8") as f:
    f.write("Iteration,Holistic_SPSA_Loss,Coord_FDSA_Loss,GD_Exact_Loss,SGD_Noisy_Loss\n")
    for step in range(0, iters + 1, 15):
        f.write(f"{step},{loss_spsa[step]:.6f},{loss_fdsa[step]:.6f},{loss_gd[step]:.6f},{loss_sgd[step]:.6f}\n")

# -------------------------------------------------------------
# CELL 6: Part 1 Visualizations
# -------------------------------------------------------------
fig1, axes1 = plt.subplots(1, 2, figsize=(14.5, 5.5))

# Left: 2D Contour & Trajectories
ax = axes1[0]
gx = np.linspace(-0.5, 2.3, 250)
gy = np.linspace(-0.5, 2.3, 250)
X, Y = np.meshgrid(gx, gy)
Z = X**3 + Y**3 - 3.0 * X * Y + 1.0

levels = np.linspace(-1.0, 10.0, 35)
cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.55)
ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

ax.plot(traj_spsa[:, 0], traj_spsa[:, 1], "r.-", linewidth=1.4, markersize=3, label="1. Holistic SPSA (2 evals/step)")
ax.plot(traj_fdsa[:, 0], traj_fdsa[:, 1], "m.-", linewidth=1.4, markersize=3, label="2. Coord. FDSA (4 evals/step)")
ax.plot(traj_gd[:, 0],   traj_gd[:, 1],   "b-",  linewidth=2.0, label="3. GD (Exact Analytical)")
ax.plot(traj_sgd[:, 0],  traj_sgd[:, 1],  "g-",  linewidth=1.8, alpha=0.85, label="4. SGD (Noisy Gradient)")

ax.scatter([theta_init[0]], [theta_init[1]], color="lime", edgecolor="black", s=100, zorder=5, label="Start (2.0, 0.3)")
ax.scatter([local_min[0]],  [local_min[1]],  color="gold", marker="*", edgecolor="black", s=220, zorder=5, label="Minimum (1, 1)")
ax.scatter([saddle_pt[0]],  [saddle_pt[1]],  color="black", marker="x", s=90, zorder=5, label="Saddle (0, 0)")

ax.set_title("2D Optimization Trajectories", fontsize=12, fontweight="bold")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

# Right: Log-Loss Convergence vs. Iterations
ax = axes1[1]
iters_axis = np.arange(iters + 1)
ax.semilogy(iters_axis, np.maximum(loss_spsa, 1e-12), "r-",  linewidth=1.8, label="1. Holistic SPSA")
ax.semilogy(iters_axis, np.maximum(loss_fdsa, 1e-12), "m-",  linewidth=1.8, label="2. Coord. FDSA")
ax.semilogy(iters_axis, np.maximum(loss_gd, 1e-12),   "b--", linewidth=2.0, label="3. GD (Exact)")
ax.semilogy(iters_axis, np.maximum(loss_sgd, 1e-12),  "g-.", linewidth=1.8, label="4. SGD (Noisy)")

ax.set_title("Loss vs. Iterations", fontsize=12, fontweight="bold")
ax.set_xlabel("Iteration (k)")
ax.set_ylabel("Loss f(theta) [log scale]")
ax.legend(fontsize=8.5)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
fig1_path = os.path.join(figures_dir, "part1_trajectories_convergence.png")
fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")
b64_fig1 = fig_to_base64(fig1)
plt.close(fig1)

nb["cells"][6]["execution_count"] = 4
nb["cells"][6]["outputs"] = [{
    "data": {
        "image/png": b64_fig1,
        "text/plain": ["<Figure size 1450x550 with 2 Axes>"]
    },
    "metadata": {},
    "output_type": "display_data"
}]

# -------------------------------------------------------------
# CELL 8: Part 2 Neural Network Architecture & Loss
# -------------------------------------------------------------
def init_network_params(dims=(2, 5, 5, 5, 1), seed=42):
    torch.manual_seed(seed)
    params = []
    for i in range(len(dims) - 1):
        w = torch.randn(dims[i], dims[i + 1], dtype=torch.float32)
        b = torch.randn(1, dims[i + 1], dtype=torch.float32)
        params.append(w)
        params.append(b)
    return params

def forward_pass(x, params, beta=0.5):
    a = x
    num_layers = len(params) // 2
    for l in range(num_layers - 1):
        w = params[2 * l]
        b = params[2 * l + 1]
        a = torch.tanh(beta * (a @ w + b))
    w_out = params[-2]
    b_out = params[-1]
    return a @ w_out + b_out

def compute_loss(preds, targets):
    return 0.5 * torch.sum((preds - targets) ** 2)

def compute_accuracy(preds, targets):
    pred_labels = torch.where(preds >= 0, 1.0, -1.0)
    correct = (pred_labels == targets).float().sum().item()
    return 100.0 * correct / len(targets)

def record_metrics(history, x_train, y_train, x_test, y_test, params, evals, beta=0.5):
    with torch.no_grad():
        tr_preds = forward_pass(x_train, params, beta)
        te_preds = forward_pass(x_test, params, beta)
        history["train_loss"].append(compute_loss(tr_preds, y_train).item())
        history["test_loss"].append(compute_loss(te_preds, y_test).item())
        history["train_acc"].append(compute_accuracy(tr_preds, y_train))
        history["test_acc"].append(compute_accuracy(te_preds, y_test))
        history["evals"].append(evals)

nb["cells"][8]["execution_count"] = 5
nb["cells"][8]["outputs"] = []

# -------------------------------------------------------------
# CELL 9: Part 2 Optimizers
# -------------------------------------------------------------
def train_layerwise_spsa(x_train, y_train, x_test, y_test, init_params,
                         epochs=100, t_max=30, a=0.02, c=0.10, beta=0.5, seed=42):
    torch.manual_seed(seed)
    params = [p.clone().detach() for p in init_params]
    num_layers = len(params) // 2

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [], "evals": []}
    total_evals = 0.0
    record_metrics(history, x_train, y_train, x_test, y_test, params, total_evals, beta)

    t0 = time.perf_counter()
    with torch.no_grad():
        for epoch in range(1, epochs + 1):
            for l in range(num_layers):
                idx_w, idx_b = 2 * l, 2 * l + 1
                w, b = params[idx_w], params[idx_b]

                for t in range(1, t_max + 1):
                    at = a / (t ** 1.0)
                    ct = c / (t ** (1.0 / 6.0))

                    dw = (torch.randint(0, 2, w.shape) * 2 - 1).float()
                    db = (torch.randint(0, 2, b.shape) * 2 - 1).float()

                    params[idx_w] = w + ct * dw
                    params[idx_b] = b + ct * db
                    loss_plus = compute_loss(forward_pass(x_train, params, beta), y_train)

                    params[idx_w] = w - ct * dw
                    params[idx_b] = b - ct * db
                    loss_minus = compute_loss(forward_pass(x_train, params, beta), y_train)

                    total_evals += 2.0

                    ghat_w = (loss_plus - loss_minus) / (2.0 * ct * dw)
                    ghat_b = (loss_plus - loss_minus) / (2.0 * ct * db)

                    w = w - at * ghat_w
                    b = b - at * ghat_b

                params[idx_w] = w
                params[idx_b] = b

            record_metrics(history, x_train, y_train, x_test, y_test, params, total_evals, beta)

    history["runtime"] = time.perf_counter() - t0
    return params, history

def train_holistic_spsa(x_train, y_train, x_test, y_test, init_params,
                        epochs=100, t_max=30, a=0.02, c=0.10, beta=0.5, seed=42):
    torch.manual_seed(seed)
    params = [p.clone().detach() for p in init_params]
    num_layers = len(params) // 2

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [], "evals": []}
    total_evals = 0.0
    record_metrics(history, x_train, y_train, x_test, y_test, params, total_evals, beta)

    t0 = time.perf_counter()
    with torch.no_grad():
        for epoch in range(1, epochs + 1):
            for _ in range(num_layers):
                for t in range(1, t_max + 1):
                    at = a / (t ** 1.0)
                    ct = c / (t ** (1.0 / 6.0))

                    deltas = [(torch.randint(0, 2, p.shape) * 2 - 1).float() for p in params]

                    for i in range(len(params)):
                        params[i] += ct * deltas[i]
                    loss_plus = compute_loss(forward_pass(x_train, params, beta), y_train)

                    for i in range(len(params)):
                        params[i] -= 2.0 * ct * deltas[i]
                    loss_minus = compute_loss(forward_pass(x_train, params, beta), y_train)

                    total_evals += 2.0

                    for i in range(len(params)):
                        ghat = (loss_plus - loss_minus) / (2.0 * ct * deltas[i])
                        params[i] = (params[i] + ct * deltas[i]) - at * ghat

            record_metrics(history, x_train, y_train, x_test, y_test, params, total_evals, beta)

    history["runtime"] = time.perf_counter() - t0
    return params, history

def train_full_batch_gd(x_train, y_train, x_test, y_test, init_params,
                        epochs=100, lr=0.005, beta=0.5):
    params = [p.clone().detach().requires_grad_(True) for p in init_params]
    optimizer = torch.optim.SGD(params, lr=lr)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [], "evals": []}
    total_evals = 0.0
    record_metrics(history, x_train, y_train, x_test, y_test, params, total_evals, beta)

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = compute_loss(forward_pass(x_train, params, beta), y_train)
        loss.backward()
        optimizer.step()

        total_evals += 3.0
        record_metrics(history, x_train, y_train, x_test, y_test, params, total_evals, beta)

    history["runtime"] = time.perf_counter() - t0
    return [p.detach() for p in params], history

def train_minibatch_sgd(x_train, y_train, x_test, y_test, init_params,
                        epochs=100, lr=0.01, batch_size=25, beta=0.5, seed=42):
    torch.manual_seed(seed)
    params = [p.clone().detach().requires_grad_(True) for p in init_params]
    optimizer = torch.optim.SGD(params, lr=lr)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [], "evals": []}
    total_evals = 0.0
    record_metrics(history, x_train, y_train, x_test, y_test, params, total_evals, beta)

    n_samples = len(x_train)
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()
            loss_b = compute_loss(forward_pass(xb, params, beta), yb)
            loss_b.backward()
            optimizer.step()

            total_evals += 3.0 * (len(xb) / float(n_samples))

        record_metrics(history, x_train, y_train, x_test, y_test, params, total_evals, beta)

    history["runtime"] = time.perf_counter() - t0
    return [p.detach() for p in params], history

nb["cells"][9]["execution_count"] = 6
nb["cells"][9]["outputs"] = []

# -------------------------------------------------------------
# CELL 10: Run Part 2 NN Benchmarks & Print Tables
# -------------------------------------------------------------
print("\nRunning Part 2 Neural Network training across 4 optimizers...")
x_raw, y_cls = make_circles(n_samples=250, noise=0.06, factor=0.4, random_state=SEED)
y_signed = np.where(y_cls == 1, 1.0, -1.0)

x_tr_np, x_te_np, y_tr_np, y_te_np = train_test_split(
    x_raw, y_signed, test_size=0.20, random_state=SEED, stratify=y_cls
)

x_train = torch.tensor(x_tr_np, dtype=torch.float32)
y_train = torch.tensor(y_tr_np[:, None], dtype=torch.float32)
x_test  = torch.tensor(x_te_np, dtype=torch.float32)
y_test  = torch.tensor(y_te_np[:, None], dtype=torch.float32)

p0 = init_network_params((2, 5, 5, 5, 1), seed=SEED)
n_params = sum(p.numel() for p in p0)

with torch.no_grad():
    init_tr_loss = compute_loss(forward_pass(x_train, p0), y_train).item()
    init_te_acc  = compute_accuracy(forward_pass(x_test, p0), y_test)

p_hol, hist_hol = train_holistic_spsa(x_train, y_train, x_test, y_test, p0, epochs=100, t_max=30, a=0.02, c=0.10, seed=SEED)
p_lw,  hist_lw  = train_layerwise_spsa(x_train, y_train, x_test, y_test, p0, epochs=100, t_max=30, a=0.02, c=0.10, seed=SEED)
p_gd,  hist_gd  = train_full_batch_gd(x_train, y_train, x_test, y_test, p0, epochs=100, lr=0.005)
p_sgd, hist_sgd = train_minibatch_sgd(x_train, y_train, x_test, y_test, p0, epochs=100, lr=0.01, batch_size=25, seed=SEED)

c10_lines = []
c10_lines.append(f"Dataset: {len(x_train)} train / {len(x_test)} test samples | Network parameters: {n_params}")
c10_lines.append(f"Initial State -> Train Loss: {init_tr_loss:.2f} | Test Accuracy: {init_te_acc:.1f}%\n")

c10_lines.append("=" * 102)
c10_lines.append(" PART 2: FINAL NEURAL NETWORK COMPARISON (100 EPOCHS)")
c10_lines.append("=" * 102)
c10_lines.append(f"{'Optimizer':<24} | {'Backprop':<9} | {'Train Loss':<10} | {'Train Acc':<9} | {'Test Acc':<8} | {'Full Evals':<10} | {'Time (s)'}")
c10_lines.append("-" * 102)
results = [
    ("1. Holistic SPSA",   "No",  hist_hol),
    ("2. Layer-Wise SPSA", "No",  hist_lw),
    ("3. GD (Full-Batch)", "Yes", hist_gd),
    ("4. SGD (Mini-Batch)","Yes", hist_sgd),
]
for name, bp, h in results:
    c10_lines.append(
        f"{name:<24} | {bp:<9} | {h['train_loss'][-1]:<10.4f} | "
        f"{h['train_acc'][-1]:<8.1f}% | {h['test_acc'][-1]:<7.1f}% | "
        f"{h['evals'][-1]:<10.0f} | {h['runtime']:.2f}"
    )
c10_lines.append("=" * 102)

c10_lines.append("\n" + "=" * 96)
c10_lines.append(" TABLE 2A: EARLY EPOCH DYNAMICS (EPOCHS 0 TO 20 — TRAIN LOSS & TEST ACCURACY)")
c10_lines.append("=" * 96)
c10_lines.append(f"{'Epoch':<6} | {'Holistic SPSA':<20} | {'Layer-Wise SPSA':<20} | {'GD (Full-Batch)':<20} | {'SGD (Mini-Batch)'}")
c10_lines.append("-" * 96)
for ep in [0, 1, 2, 3, 5, 10, 15, 20]:
    c10_lines.append(
        f"{ep:<6} | "
        f"{hist_hol['train_loss'][ep]:6.2f} ({hist_hol['test_acc'][ep]:5.1f}%)     | "
        f"{hist_lw['train_loss'][ep]:6.2f} ({hist_lw['test_acc'][ep]:5.1f}%)     | "
        f"{hist_gd['train_loss'][ep]:6.2f} ({hist_gd['test_acc'][ep]:5.1f}%)     | "
        f"{hist_sgd['train_loss'][ep]:6.2f} ({hist_sgd['test_acc'][ep]:5.1f}%)"
    )
c10_lines.append("=" * 96)

c10_lines.append("\n" + "=" * 96)
c10_lines.append(" TABLE 2B: FULL TRAINING PROGRESSION (EPOCHS 0 TO 100 — TRAIN LOSS & TEST ACCURACY)")
c10_lines.append("=" * 96)
c10_lines.append(f"{'Epoch':<6} | {'Holistic SPSA':<20} | {'Layer-Wise SPSA':<20} | {'GD (Full-Batch)':<20} | {'SGD (Mini-Batch)'}")
c10_lines.append("-" * 96)
for ep in [0, 20, 40, 60, 80, 100]:
    c10_lines.append(
        f"{ep:<6} | "
        f"{hist_hol['train_loss'][ep]:6.2f} ({hist_hol['test_acc'][ep]:5.1f}%)     | "
        f"{hist_lw['train_loss'][ep]:6.2f} ({hist_lw['test_acc'][ep]:5.1f}%)     | "
        f"{hist_gd['train_loss'][ep]:6.2f} ({hist_gd['test_acc'][ep]:5.1f}%)     | "
        f"{hist_sgd['train_loss'][ep]:6.2f} ({hist_sgd['test_acc'][ep]:5.1f}%)"
    )
c10_lines.append("=" * 96)
c10_stdout = "\n".join(c10_lines) + "\n"
print(c10_stdout)

nb["cells"][10]["execution_count"] = 7
nb["cells"][10]["outputs"] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [line + "\n" for line in c10_lines]
}]

# Save Part 2 CSVs
with open(os.path.join(output_dir, "part2_final_comparison.csv"), "w", encoding="utf-8") as f:
    f.write("Optimizer,Backprop,Final_Train_Loss,Final_Train_Acc,Final_Test_Acc,Total_Evals,Runtime_Sec\n")
    for name, bp, h in results:
        f.write(f'"{name}","{bp}",{h["train_loss"][-1]:.4f},{h["train_acc"][-1]:.2f},{h["test_acc"][-1]:.2f},{h["evals"][-1]:.0f},{h["runtime"]:.2f}\n')

with open(os.path.join(output_dir, "part2_early_epochs_2a.csv"), "w", encoding="utf-8") as f:
    f.write("Epoch,Holistic_Loss,Holistic_TestAcc,LayerWise_Loss,LayerWise_TestAcc,GD_Loss,GD_TestAcc,SGD_Loss,SGD_TestAcc\n")
    for ep in [0, 1, 2, 3, 5, 10, 15, 20]:
        f.write(f"{ep},{hist_hol['train_loss'][ep]:.4f},{hist_hol['test_acc'][ep]:.2f},{hist_lw['train_loss'][ep]:.4f},{hist_lw['test_acc'][ep]:.2f},{hist_gd['train_loss'][ep]:.4f},{hist_gd['test_acc'][ep]:.2f},{hist_sgd['train_loss'][ep]:.4f},{hist_sgd['test_acc'][ep]:.2f}\n")

with open(os.path.join(output_dir, "part2_full_progression_2b.csv"), "w", encoding="utf-8") as f:
    f.write("Epoch,Holistic_Loss,Holistic_TestAcc,LayerWise_Loss,LayerWise_TestAcc,GD_Loss,GD_TestAcc,SGD_Loss,SGD_TestAcc\n")
    for ep in [0, 20, 40, 60, 80, 100]:
        f.write(f"{ep},{hist_hol['train_loss'][ep]:.4f},{hist_hol['test_acc'][ep]:.2f},{hist_lw['train_loss'][ep]:.4f},{hist_lw['test_acc'][ep]:.2f},{hist_gd['train_loss'][ep]:.4f},{hist_gd['test_acc'][ep]:.2f},{hist_sgd['train_loss'][ep]:.4f},{hist_sgd['test_acc'][ep]:.2f}\n")

# -------------------------------------------------------------
# CELL 11: Part 2 Visualizations (Loss vs Epochs, Evals, Accuracy)
# -------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.0))
epochs_axis = np.arange(len(hist_lw["train_loss"]))

curves = [
    ("Holistic SPSA (0 bp)",   hist_hol, "#d62728", "-"),
    ("Layer-Wise SPSA (0 bp)", hist_lw,  "#ff7f0e", "-"),
    ("GD (Full-Batch)",        hist_gd,  "#1f77b4", "--"),
    ("SGD (Mini-Batch)",       hist_sgd, "#2ca02c", "-."),
]

# Subplot 1: Training Loss vs. Epochs
ax = axes2[0]
for label, h, color, style in curves:
    ax.semilogy(epochs_axis, h["train_loss"], linestyle=style, linewidth=2.0, color=color, label=label)
ax.set_title("Training Loss vs. Epochs", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss (log scale)")
ax.legend(fontsize=8.5)
ax.grid(True, which="both", alpha=0.25)

# Subplot 2: Training Loss vs. Function Evaluations
ax = axes2[1]
for label, h, color, style in curves:
    ax.loglog(np.array(h["evals"]) + 1.0, h["train_loss"], linestyle=style, linewidth=2.0, color=color, label=label)
ax.set_title("Training Loss vs. Function Evaluations", fontsize=12, fontweight="bold")
ax.set_xlabel("Full-Dataset Eval Equivalents (+1, log scale)")
ax.set_ylabel("Training Loss (log scale)")
ax.legend(fontsize=8.5)
ax.grid(True, which="both", alpha=0.25)

# Subplot 3: Test Accuracy vs. Epochs
ax = axes2[2]
for label, h, color, style in curves:
    ax.plot(epochs_axis, h["test_acc"], linestyle=style, linewidth=2.0, color=color, label=label)
ax.set_title("Held-Out Test Accuracy vs. Epochs", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Test Accuracy (%)")
ax.set_ylim(30, 105)
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(True, alpha=0.25)

plt.tight_layout()
fig2_path = os.path.join(figures_dir, "part2_loss_evals_accuracy.png")
fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
b64_fig2 = fig_to_base64(fig2)
plt.close(fig2)

nb["cells"][11]["execution_count"] = 8
nb["cells"][11]["outputs"] = [{
    "data": {
        "image/png": b64_fig2,
        "text/plain": ["<Figure size 1800x500 with 3 Axes>"]
    },
    "metadata": {},
    "output_type": "display_data"
}]

# -------------------------------------------------------------
# CELL 12: Part 2 Decision Boundaries
# -------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 5, figsize=(21, 4.3))

x_min, x_max = x_raw[:, 0].min() - 0.3, x_raw[:, 0].max() + 0.3
y_min, y_max = x_raw[:, 1].min() - 0.3, x_raw[:, 1].max() + 0.3
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180))
grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

models = [
    ("Initial (Untrained)",    p0),
    ("Holistic SPSA (0 bp)",   p_hol),
    ("Layer-Wise SPSA (0 bp)", p_lw),
    ("GD (Full-Batch)",        p_gd),
    ("SGD (Mini-Batch)",       p_sgd),
]

for ax, (title, params) in zip(axes3, models):
    with torch.no_grad():
        zz = forward_pass(grid_tensor, params).numpy().reshape(xx.shape)
        te_acc = compute_accuracy(forward_pass(x_test, params), y_test)

    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.45)
    ax.contour(xx, yy, zz, levels=[0.0], colors="black", linewidths=2.0)

    ax.scatter(x_tr_np[y_tr_np == -1, 0], x_tr_np[y_tr_np == -1, 1], c="#b2182b", edgecolor="white", s=25, label="Train (-1)")
    ax.scatter(x_tr_np[y_tr_np ==  1, 0], x_tr_np[y_tr_np ==  1, 1], c="#2166ac", edgecolor="white", s=25, label="Train (+1)")
    ax.scatter(x_te_np[y_te_np == -1, 0], x_te_np[y_te_np == -1, 1], c="#b2182b", marker="^", edgecolor="black", s=45, label="Test (-1)")
    ax.scatter(x_te_np[y_te_np ==  1, 0], x_te_np[y_te_np ==  1, 1], c="#2166ac", marker="^", edgecolor="black", s=45, label="Test (+1)")

    ax.set_title(f"{title}\nTest Acc: {te_acc:.1f}%", fontsize=10.5, fontweight="bold")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])

axes3[0].legend(fontsize=7.5, loc="lower left")
plt.tight_layout()
fig3_path = os.path.join(figures_dir, "part2_decision_boundaries.png")
fig3.savefig(fig3_path, dpi=200, bbox_inches="tight")
b64_fig3 = fig_to_base64(fig3)
plt.close(fig3)

nb["cells"][12]["execution_count"] = 9
nb["cells"][12]["outputs"] = [{
    "data": {
        "image/png": b64_fig3,
        "text/plain": ["<Figure size 2100x430 with 5 Axes>"]
    },
    "metadata": {},
    "output_type": "display_data"
}]

# -------------------------------------------------------------
# Save Executed Notebook
# -------------------------------------------------------------
executed_nb_path = os.path.join(output_dir, "mahan.ipynb")
with open(executed_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"\n[Saved] Executed notebook saved to: {executed_nb_path}")

# -------------------------------------------------------------
# Extract and Save Pure Python Source Code
# -------------------------------------------------------------
code_path = os.path.join(output_dir, "mahan_code.py")
with open(code_path, "w", encoding="utf-8") as f:
    f.write('"""\n')
    f.write('Mahan SPSA Implementation & Empirical Study\n')
    f.write('Extracted from mahan.ipynb\n')
    f.write('Reference Paper: B. Wulff, J. Schuecker, C. Bauckhage (ICANN 2018)\n')
    f.write('"""\n\n')
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            f.write("# " + "=" * 70 + "\n")
            f.write("".join(cell["source"]))
            f.write("\n\n")

print(f"[Saved] Python code script saved to: {code_path}")

# -------------------------------------------------------------
# Generate Comprehensive README for mahan/
# -------------------------------------------------------------
readme_path = os.path.join(output_dir, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(f"""# Mahan's SPSA Implementation & Benchmark Suite

This directory contains the executed Jupyter Notebook, standalone Python source code, high-resolution figures, and CSV data logs for **Mahan's SPSA Implementation**.

---

## 1. Directory Contents

```text
mahan/
├── mahan.ipynb                        # Fully executed Jupyter Notebook with embedded plots and outputs
├── mahan_code.py                      # Standalone, runnable Python script extracted from the notebook
├── README.md                          # Documentation of experiments and results (this file)
├── part1_optimization_summary.csv     # Part 1 final summary metrics
├── part1_convergence_checkpoints.csv  # Part 1 iteration checkpoints (every 15 steps)
├── part2_final_comparison.csv         # Part 2 neural network 100-epoch results
├── part2_early_epochs_2a.csv          # Table 2A: Early epoch dynamics (epochs 0 to 20)
├── part2_full_progression_2b.csv      # Table 2B: Full progression (every 20 epochs to 100)
└── figures/
    ├── part1_trajectories_convergence.png  # Part 1 2D contour paths & log-loss convergence
    ├── part2_loss_evals_accuracy.png       # Part 2 training loss, evals, and held-out test accuracy
    └── part2_decision_boundaries.png       # Part 2 2D decision boundaries across all 5 models
```

---

## 2. Experimental Summary & Key Findings

### Part 1: Cubic Polynomial Optimization ($f(x, y) = x^3 + y^3 - 3xy + 1$)
* **Initial State**: $\\mathbf{{x}}_0 = (2.0, 0.3)$, Loss $= 7.227000$
* **Target Minimum**: $\\mathbf{{x}}^* = (1.0, 1.0)$, $f(\\mathbf{{x}}^*) = 0.0$

| Method | Evals/Step | Final Loss | Final Coordinates $(x, y)$ |
|:---|:---:|:---:|:---:|
| **1. Holistic SPSA** | 2 forward | {loss_spsa[-1]:.6f} | ({th_spsa[0]:.4f}, {th_spsa[1]:.4f}) |
| **2. Coordinate-Wise FDSA** | 4 forward | {loss_fdsa[-1]:.6f} | ({th_fdsa[0]:.4f}, {th_fdsa[1]:.4f}) |
| **3. GD (Exact Analytical)** | Exact Grad | {loss_gd[-1]:.6f} | ({th_gd[0]:.4f}, {th_gd[1]:.4f}) |
| **4. SGD (Noisy Gradient)** | Noisy Grad | {loss_sgd[-1]:.6f} | ({th_sgd[0]:.4f}, {th_sgd[1]:.4f}) |

---

### Part 2: Neural Network Classification on Concentric Circles
* **Dataset**: $N=250$ concentric circles ($200$ train / $50$ held-out test samples).
* **Architecture**: $2 \\to 5 \\to 5 \\to 5 \\to 1$ ($3$ tanh hidden layers + $1$ linear output layer, $81$ parameters).

#### Final Performance Comparison (Epoch 100):
| Optimizer | Backprop? | Train Loss | Train Acc | Test Acc | Full Evals | Runtime (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Holistic SPSA** | **No** | {hist_hol['train_loss'][-1]:.4f} | {hist_hol['train_acc'][-1]:.1f}% | {hist_hol['test_acc'][-1]:.1f}% | {hist_hol['evals'][-1]:.0f} | {hist_hol['runtime']:.2f} s |
| **2. Layer-Wise SPSA** | **No** | {hist_lw['train_loss'][-1]:.4f} | {hist_lw['train_acc'][-1]:.1f}% | {hist_lw['test_acc'][-1]:.1f}% | {hist_lw['evals'][-1]:.0f} | {hist_lw['runtime']:.2f} s |
| **3. GD (Full-Batch)** | Yes | {hist_gd['train_loss'][-1]:.4f} | {hist_gd['train_acc'][-1]:.1f}% | {hist_gd['test_acc'][-1]:.1f}% | {hist_gd['evals'][-1]:.0f} | {hist_gd['runtime']:.2f} s |
| **4. SGD (Mini-Batch)** | Yes | {hist_sgd['train_loss'][-1]:.4f} | {hist_sgd['train_acc'][-1]:.1f}% | {hist_sgd['test_acc'][-1]:.1f}% | {hist_sgd['evals'][-1]:.0f} | {hist_sgd['runtime']:.2f} s |

---

## 3. How to Run

1. **Jupyter Notebook**:
   ```bash
   jupyter notebook mahan.ipynb
   ```
2. **Python Code**:
   ```bash
   python mahan_code.py
   ```
""")

print(f"[Saved] Documentation saved to: {readme_path}")
print("\nExecution and export to 'mahan/' completed successfully!")
