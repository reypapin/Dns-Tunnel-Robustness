# DNS Tunnel Robustness

Code for the paper **"Simple Payload Mutations Break Machine Learning Based DNS Tunneling Detection"**
by Reynier Leyva La O and Carlos A. Catania.

---

## The problem

DNS tunnel detectors are trained on specific captures and then tested on captures from the same session. That works fine on paper, but it breaks as soon as the attacker changes what they put inside the tunnel.

We tested this with iodine. We kept the tool, the protocol, and the setup exactly the same. We only changed the payload type — audio, image, text, video, executable, compressed. Models that got 98% recall on the original captures dropped to zero.

| Model | CIC-2021 recall | After payload mutation |
|---|---|---|
| RandomForest | 98.5% | 0.0% |
| XGBoost | 98.5% | 0.0% |
| LightGBM | 98.5% | 0.0% |
| LSTM | 93.5% | 1.8% |
| CNN | 93.5% | 29.9% |
| LogisticRegression | 94.7% | 45.2% |

We then looked at why. The benign traffic distribution barely matters (changing it drops recall by at most 0.24 percentage points). The attack mutation is the issue.

## The fix

We retrained each model with a small amount of mutated samples added to the training set — 3.47% of the total data. That was enough to get 100% recall on mutated payloads while keeping performance on the original test set.

The hardened models also worked on tunneling tools they had never seen before (dns2tcp, dnspot, cobalstrike, tuns, and others from the GraphTunnel dataset). The models were trained on iodine and dnscat2. The fact that they transfer suggests the mutation training captures something general about tunneling behavior, not just iodine's specific patterns.

| Model | Recall on unknown tools (baseline) | Recall on unknown tools (hardened) |
|---|---|---|
| RandomForest | 1.3% | 89.8% |
| XGBoost | 0.0% | 91.1% |
| LightGBM | 0.0% | 65.0% |
| LSTM | 44.9% | 100.0% |

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

**Mutant Payload** — generated with [ARAGAT](https://github.com/DNS-Datasets/ARAGAT). We ran it with 9 random seeds (10, 21, 35, 42, 55, 60, 75, 82, 99), producing 7,548 attack flows.

**GraphTunnel** — only needed for Experiment 5. The notebook clones it automatically from [DNS-Datasets/GraphTunnel](https://github.com/DNS-Datasets/GraphTunnel).

Once you have the data on Drive, open the notebook and run the cells in order. Experiment 1 trains and saves the baseline models, which the later experiments load.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rleyva-la/Dns-Tunnel-Robustness/blob/main/notebooks/experiments.ipynb)

---

## Features

23 features per flow, split into two groups:

**Stateless (11)** — extracted per DNS query: `FQDN_count`, `subdomain_length`, `upper`, `lower`, `numeric`, `entropy`, `special`, `labels`, `labels_max`, `labels_average`, `len`

**Stateful (12)** — extracted per session: `rr`, `A_frequency`, `AAAA_frequency`, `CNAME_frequency`, `TXT_frequency`, `MX_frequency`, `NS_frequency`, `NULL_frequency`, `rr_count`, `distinct_ip`, `unique_ttl`, `total_queries`

---

## Citation

```bibtex
@article{leyva2025dns,
  title={Simple Payload Mutations Break Machine Learning Based DNS Tunneling Detection},
  author={Leyva La O, Reynier and Catania, Carlos A.},
  year={2025}
}
```

BibTeX will be updated once the paper is published.
