# SPSA Paper Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)

An implementation of **SPSA** and **layer-wise SPSA** for optimising mathematical functions and for training neural networks **without backpropagation**, compared against `torch.optim.SGD`. It is based on:

> **B. Wulff, J. Schuecker, C. Bauckhage.** *SPSA for Layer-Wise Training of Deep Networks.* ICANN 2018, LNCS 11141, pp. 564–573. DOI: [10.1007/978-3-030-01424-7_55](https://doi.org/10.1007/978-3-030-01424-7_55)

**Implementation by**: Ananth Shyam, Rama Narasimha and Mahan Bhimireddy

## Repository layout

```text
.
├── demo/
│   └── implementation demonstration/
│       ├── spsa_demonstration.ipynb   # the full demonstration (code, tables, figures, takeaways)
│       ├── README.md                  # detailed results
│       └── figures/                   # figures exported from the executed notebook
├── CITATION.cff
├── LICENSE
├── requirements.txt
└── README.md
```

## What the notebook does

| Part | Task | Methods | Paper |
|---|---|---|---|
| 1A | Minimise f(x, y) = x³ + y³ − 3xy + 1 from (2, 0.3) | SPSA (Algorithm 1) vs. `torch.optim.SGD` | Algorithm 1 |
| 1B | Minimise ‖W₃W₂W₁ − M‖²_F (75 parameters in 3 "layers") | layer-wise vs. holistic SPSA vs. SGD | Algorithm 2 |
| 2 | Classify XOR and moons with the 2×5×5×5×1 tanh network (81 parameters) | layer-wise SPSA (bottom-up, top-down, random order) vs. holistic SPSA vs. SGD | Sect. 4.1, Fig. 1a/b/d/e |

Every experiment is repeated over 100 trials. The notebook reports:
- loss vs. iterations/epochs and loss vs. function evaluations;
- held-out test accuracy, precision, recall, F1, confusion matrices and decision boundaries.

The paper's 3-class circles task and MNIST experiment are not included. [Full results and details are in the demo README](demo/implementation%20demonstration/README.md).

## Headline results (100 trials, 25 epochs)

**Part 2, mean training error E (lower is better):**

| Task | Layer-wise SPSA (bottom-up) | Holistic SPSA, same gains | SGD, 25 epochs | SGD, 300 epochs |
|---|---|---|---|---|
| XOR | **3.03** (test acc 99.3%) | 50.75 (93.0%) | 5.28 (99.2%) | 0.91 (99.2%) |
| Moons | **20.41** (test acc 96.0%) | 74.27 (88.7%) | 46.71 (89.3%) | 0.05 (100%) |

**What this shows:**
- **Layer-wise SPSA beats holistic SPSA, and the layer order hardly matters,** as the paper reports.
- **SGD needs far less compute.** It uses 3 dataset passes per epoch against SPSA's 400, so given 900 passes it beats SPSA's 10,000-pass result.

**Part 1A:** SPSA reaches the minimum (1, 1) in all 100 trials, with a median final f of 3.7·10⁻⁸ after 300 function evaluations.

## Run it

```bash
pip install -r requirements.txt
cd "demo/implementation demonstration"
jupyter notebook spsa_demonstration.ipynb      # Run All; about 7 min on a CUDA GPU, also runs on a CPU
```

## Citation

```bibtex
@inproceedings{wulff2018spsa,
  title     = {SPSA for Layer-Wise Training of Deep Networks},
  author    = {Wulff, Benjamin and Schuecker, Jannis and Bauckhage, Christian},
  booktitle = {Artificial Neural Networks and Machine Learning -- ICANN 2018},
  series    = {LNCS},
  volume    = {11141},
  pages     = {564--573},
  publisher = {Springer},
  year      = {2018},
  doi       = {10.1007/978-3-030-01424-7_55}
}
```
