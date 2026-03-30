# data/

The datasets are not in the repo. This file explains where to get them and where to put them.

---

## CIC Bell DNS 2021

Download from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/).

We use the pre-extracted CSV files, not the raw pcaps. Each capture produces two files: one with per-query features (`stateless_*.csv`) and one with per-session features (`stateful_*.csv`).

Put the data on Google Drive at `MyDrive/Tunnel/CSV_CIC21/` with this layout:

```
CSV_CIC21/
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

Total: ~140,000 flows.

---

## Perturbed Payload

Generated with [perturbed-dns](https://github.com/reypapin/perturbed-dns), a DNS tunnel tool with 4-dimensional structural perturbations (metadata position, encoding, chunking, and timing).

Put the generated CSVs at `MyDrive/Tunnel/CSV_CIC21/CSV_Generated/`:

```
CSV_Generated/
├── stateless_features-bridge.pcap_10.csv
├── stateful_features-bridge.pcap_10.csv
├── stateless_features-bridge.pcap_21.csv
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

Seeds used: 10, 21, 35, 42, 55, 60, 75, 82, 99. Total: 7,548 attack flows across nine independent runs.

---

## GraphTunnel

Only needed for Experiment 4. The notebook clones it automatically — no manual setup.

Source: [https://github.com/DNS-Datasets/GraphTunnel](https://github.com/DNS-Datasets/GraphTunnel)

Tools we evaluate from this dataset: dns2tcp, dnspot, tcp-over-dns-CNAME, tcp-over-dns-TXT, DNS-shell, cobalstrike, ozymandns, tuns.

Tools we skip (already in CIC-2021): iodine, dnscat2, AndIodine.
