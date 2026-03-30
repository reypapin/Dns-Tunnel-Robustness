# DNS Tunnel Robustness

Code for the paper **"The Problem of Positional Overfitting in Machine Learning Detection of DNS Tunneling"**
by Reynier Leyva La O and Carlos A. Catania.

---

## The problem

DNS tunnel detectors trained and evaluated on a static tool set learn where protocol metadata appears in a packet, not what makes traffic malicious. We call this positional overfitting. Models that reach 98%+ recall on the CIC Bell DNS 2021 benchmark fail completely when the tunnel structure changes — even with simple modifications that require no model access and no adversarial optimization.

We tested this with four structural perturbation dimensions: metadata position, encoding scheme, chunking strategy, and query timing. All are implementable in roughly 50 lines of Python.

| Model | CIC 2021 recall | After structural perturbation |
|---|---|---|
| RandomForest | 98.88% | 0.00% |
| XGBoost | 98.89% | 0.00% |
| LightGBM | 98.90% | 0.00% |
| LSTM | 98.86% | 1.80% |
| CNN | 98.83% | 29.85% |
| LogisticRegression | 98.52% | 45.18% |

Tree-based models fail completely (0% recall). The evasion requires no access to the detector — only basic DNS protocol knowledge.

## The fix

Retraining with a small amount of structurally diverse samples — 3.02% of the total training set — restores 100% recall on the known perturbation space while keeping CIC 2021 performance within 0.15 percentage points.

Hardened models also generalize to tunneling tools never seen during training (dns2tcp, dnspot, cobalstrike, tuns, and others from the GraphTunnel dataset):

| Model | Recall on unknown tools (baseline) | Recall on unknown tools (hardened) | FPR (hardened) |
|---|---|---|---|
| RandomForest | 1.34% | 89.75% | 15.83% |
| XGBoost | 0.00% | 91.08% | 16.02% |
| LightGBM | 0.00% | 64.95% | 14.94% |
| LSTM | 44.90% | 100.00% | 21.54% |

However, hardening comes with a robustness–precision trade-off: false positive rates of 15–22% on traffic outside the training distribution. This cost is invisible from held-out evaluation on the same perturbation distribution.

---

## Repo structure

```
notebooks/      main experiment notebook (runs on Google Colab)
src/            standalone Python scripts for figures and stats
data/           instructions for getting and organizing the datasets
models/         where trained models go after running the notebook
results/        CSV outputs from the experiments
figures/        paper figures
```

---

## Setup

The notebook runs on Google Colab with data on Google Drive. You need three things:

**CIC Bell DNS 2021** — from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/). We use the pre-extracted CSV features, not the raw pcaps. See `data/README.md` for the expected folder layout.

**Perturbed Payload** — generated with [perturbed-dns](https://github.com/reypapin/perturbed-dns), a DNS tunnel tool with 4-dimensional structural perturbations. We ran it with 9 random seeds (10, 21, 35, 42, 55, 60, 75, 82, 99), producing 7,548 attack flows. See `data/README.md` for the expected folder layout.

**GraphTunnel** — only needed for Experiment 4. The notebook clones it automatically from [DNS-Datasets/GraphTunnel](https://github.com/DNS-Datasets/GraphTunnel).

Once you have the data on Drive, open the notebook and run the cells in order. Experiment 1 trains and saves the baseline models, which the later experiments load.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reypapin/Dns-Tunnel-Robustness/blob/main/notebooks/experiments.ipynb)

---

## Features

23 features per flow, split into two groups:

**Stateless (11)** — extracted per DNS query: `FQDN_count`, `subdomain_length`, `upper`, `lower`, `numeric`, `entropy`, `special`, `labels`, `labels_max`, `labels_average`, `len`

**Stateful (12)** — extracted per session: `rr`, `A_frequency`, `AAAA_frequency`, `CNAME_frequency`, `TXT_frequency`, `MX_frequency`, `NS_frequency`, `NULL_frequency`, `rr_count`, `distinct_ip`, `unique_ttl`, `total_queries`

---

## Citation

```bibtex
@article{leyva2026positional,
  title={The Problem of Positional Overfitting in Machine Learning Detection of DNS Tunneling},
  author={Leyva La O, Reynier and Catania, Carlos A.},
  year={2026}
}
```
