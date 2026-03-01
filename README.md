# DNS Tunnel Robustness

Code and experiments for the paper:

> **Simple Payload Mutations Break Machine Learning Based DNS Tunneling Detection**
> Reynier Leyva La O, Carlos A. Catania

---

## What this is about

DNS tunneling detectors are usually trained and tested on the same type of traffic.
The question we asked is: *what happens when the attacker changes the payload?*

Not changing the tool, not changing the protocol — just changing what gets encoded inside the DNS queries (audio vs image vs text vs compressed data).

The answer: most models break completely.

Then we fixed it.

---

## Key results

**Experiment 2 — What happens when the payload mutates**

| Model | Baseline recall | After mutation |
|---|---|---|
| RandomForest | 98.5% | **0.0%** |
| XGBoost | 98.5% | **0.0%** |
| LightGBM | 98.5% | **0.0%** |
| LSTM | 93.5% | **1.8%** |
| CNN | 93.5% | 29.9% |
| LogisticRegression | 94.7% | 45.2% |

Tree-based models, which usually dominate benchmarks, fail entirely. LR does better — not because it's smarter, but because its linear boundary happens to be more stable.

**Experiment 3 — Why it breaks**

We split the problem into two parts: does the model fail because the *benign traffic changed*, or because the *attack changed*?

Answer: the attack mutation is the cause. Changing the benign traffic alone barely affects recall (≤0.24 pp drop). Mutating the attack payload drops recall by up to 99 pp.

**Experiment 4 — Hardening**

We retrained each model adding a small amount of mutant payload samples to the training set (3.47% of the total). That was enough to recover 100% recall on mutant payloads while keeping performance on the original CIC-2021 test set.

**Experiment 5 — Transfer to unknown tools**

We then tested the hardened models on 8 DNS tunneling tools they had never seen — not iodine, not dnscat2, but dns2tcp, dnspot, cobalstrike, ozymandns, tuns, tcp-over-dns-CNAME, tcp-over-dns-TXT, and DNS-shell.

| Model | Baseline recall | Hardened recall |
|---|---|---|
| RandomForest | 1.3% | 89.8% |
| XGBoost | 0.0% | 91.1% |
| LightGBM | 0.0% | 65.0% |
| LSTM | 44.9% | 100.0% |
| CNN | 0.0% | 100.0%* |

*CNN hardened reaches 100% recall but also 100% FPR — it flags everything as attack. LSTM hardened is the best balance (100% recall, 21.5% FPR).

---

## What's in this repo

```
├── experiments.ipynb   — all experiments end to end (Colab)
├── requirements.txt    — Python dependencies
└── data/
    └── README.md       — dataset structure and how to get the data
```

The notebook runs on Google Colab with data on Google Drive.
Trained models are not included because of size, but the notebook trains everything from scratch.

---

## How to run

**1. Get the data**

You need two datasets:

- **CIC Bell DNS 2021** — available from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/). We use the pre-extracted CSV features (stateless + stateful), not the raw pcaps.
- **Mutant Payload (ARAGAT-generated)** — generated using the ARAGAT tool. See `data/README.md` for the expected folder structure.
- **GraphTunnel** — used in Experiment 5. The notebook clones it automatically from [DNS-Datasets/GraphTunnel](https://github.com/DNS-Datasets/GraphTunnel).

**2. Set up Google Drive**

Put the data in your Google Drive at:
```
MyDrive/Tunnel/
├── CSV_CIC21/              ← CIC-2021 features
│   ├── Attack_Light_Benign/
│   ├── Attack_heavy_Benign/
│   ├── Models_SOTA_Hybrid/ ← saved baseline models (created by Exp 1)
│   └── Models_Hardened/    ← saved hardened models (created by Exp 4)
└── CSV_Generated/          ← ARAGAT mutant payload CSVs
```

**3. Open the notebook in Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rleyva-la/Dns-Tunnel-Robustness/blob/main/experiments.ipynb)

Run the cells in order. Experiment 1 trains and saves the baseline models, so it needs to run before Experiments 2–5.

---

## Features used

23 features total, extracted per DNS flow:

**Stateless (11)** — computed per query:
`FQDN_count`, `subdomain_length`, `upper`, `lower`, `numeric`, `entropy`, `special`, `labels`, `labels_max`, `labels_average`, `len`

**Stateful (12)** — computed over the session:
`rr`, `A_frequency`, `AAAA_frequency`, `CNAME_frequency`, `TXT_frequency`, `MX_frequency`, `NS_frequency`, `NULL_frequency`, `rr_count`, `distinct_ip`, `unique_ttl`, `total_queries`

---

## Citation

If you use this code or find the results useful, please cite:

```bibtex
@article{leyva2025dns,
  title={Simple Payload Mutations Break Machine Learning Based DNS Tunneling Detection},
  author={Leyva La O, Reynier and Catania, Carlos A.},
  journal={},
  year={2025}
}
```

*(BibTeX will be updated once the paper is published.)*
