# Data

The datasets are not included in this repo due to size. This file explains where to get them and how to organize them.

---

## CIC Bell DNS 2021

Download from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/).

We use the pre-extracted CSV features, not the raw pcaps. Each capture produces two CSV files:

- `stateless_*.csv` — per-query features (one row per DNS query)
- `stateful_*.csv` — per-session features (one row per session, aligned with stateless)

The notebook expects this folder structure on Google Drive:

```
MyDrive/Tunnel/CSV_CIC21/
├── Attack_Light_Benign/
│   ├── Attacks/
│   │   ├── stateless_features-light_audio.pcap.csv
│   │   ├── stateful_features-light_audio.pcap.csv
│   │   ├── stateless_features-light_compressed.pcap.csv
│   │   ├── stateful_features-light_compressed.pcap.csv
│   │   ├── stateless_features-light_exe.pcap.csv
│   │   ├── stateful_features-light_exe.pcap.csv
│   │   ├── stateless_features-light_image.pcap.csv
│   │   ├── stateful_features-light_image.pcap.csv
│   │   ├── stateless_features-light_text.pcap.csv
│   │   ├── stateful_features-light_text.pcap.csv
│   │   ├── stateless_features-light_video.pcap.csv
│   │   └── stateful_features-light_video.pcap.csv
│   └── Benign/
│       ├── stateless_features-_light_benign.pcap.csv
│       └── stateful_features-_light_benign.pcap.csv
└── Attack_heavy_Benign/
    ├── Attacks/
    │   ├── stateless_features-heavy_audio.pcap.csv
    │   ├── stateful_features-heavy_audio.pcap.csv
    │   ├── stateless_features-heavy_compressed.pcap.csv
    │   ├── stateful_features-heavy_compressed.pcap.csv
    │   ├── stateless_features-heavy_exe.pcap.csv
    │   ├── stateful_features-heavy_exe.pcap.csv
    │   ├── stateless_features-heavy_image.pcap.csv
    │   ├── stateful_features-heavy_image.pcap.csv
    │   ├── stateless_features-heavy_text.pcap.csv
    │   ├── stateful_features-heavy_text.pcap.csv
    │   ├── stateless_features-heavy_video.pcap.csv
    │   └── stateful_features-heavy_video.pcap.csv
    └── Benign/
        ├── stateless_features-benign_heavy_1.pcap.csv
        ├── stateful_features-benign_heavy_1.pcap.csv
        ├── stateless_features-benign_heavy_2.pcap.csv
        ├── stateful_features-benign_heavy_2.pcap.csv
        ├── stateless_features-benign_heavy_3.pcap.csv
        └── stateful_features-benign_heavy_3.pcap.csv
```

Total: ~140,000 flows (attacks + benign, light + heavy load).

---

## Mutant Payload (ARAGAT)

Generated using the [ARAGAT](https://github.com/DNS-Datasets/ARAGAT) tool. ARAGAT replays iodine tunnel sessions with different payload types (audio, image, text, video, exe, compressed) across multiple random seeds.

The notebook expects the generated CSVs here:

```
MyDrive/Tunnel/CSV_CIC21/CSV_Generated/
├── stateless_features-bridge.pcap_10.csv   ← seed 10
├── stateful_features-bridge.pcap_10.csv
├── stateless_features-bridge.pcap_21.csv   ← seed 21
├── stateful_features-bridge.pcap_21.csv
├── stateless_features-bridge.pcap_35.csv
├── stateful_features-bridge.pcap_35.csv
├── stateless_features-bridge.pcap_42.csv
├── stateful_features-bridge.pcap_42.csv
├── stateless_features-bridge.pcap_55.csv
├── stateful_features-bridge.pcap_55.csv
├── stateless_features-bridge.pcap_60.csv
├── stateful_features-bridge.pcap_60.csv
├── stateless_features-bridge.pcap_75.csv
├── stateful_features-bridge.pcap_75.csv
├── stateless_features-bridge.pcap_82.csv
├── stateful_features-bridge.pcap_82.csv
├── stateless_features-bridge.pcap_99.csv
└── stateful_features-bridge.pcap_99.csv
```

Seeds used: 10, 21, 35, 42, 55, 60, 75, 82, 99 (9 seeds total, 7,548 attack flows).

---

## GraphTunnel (Experiment 5 only)

The notebook clones this automatically. No manual setup needed.

Source: [https://github.com/DNS-Datasets/GraphTunnel](https://github.com/DNS-Datasets/GraphTunnel)

Tools used from this dataset: dns2tcp, dnspot, tcp-over-dns-CNAME, tcp-over-dns-TXT, DNS-shell, cobalstrike, ozymandns, tuns.

Tools excluded (already in CIC-2021): iodine, dnscat2, AndIodine.

---

## Saved models

After running Experiment 1, the notebook saves trained models to:

```
MyDrive/Tunnel/CSV_CIC21/
├── Models_SOTA_Hybrid/
│   ├── LogisticRegression_sota.joblib
│   ├── RandomForest_sota.joblib
│   ├── XGBoost_sota.joblib
│   ├── LightGBM_sota.joblib
│   ├── CNN_sota.keras
│   ├── LSTM_sota.keras
│   └── scaler_sota.joblib
└── Models_Hardened/
    ├── LogisticRegression_hardened.joblib
    ├── RandomForest_hardened.joblib
    ├── XGBoost_hardened.joblib
    ├── LightGBM_hardened.joblib
    ├── CNN_hardened.keras
    └── LSTM_hardened.keras
```

Experiments 2–5 load models from these folders, so Experiments 1 and 4 must run first.
