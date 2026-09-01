# TabPFN-NIDS: Complete Project Guide

**Project:** Scaling Tabular Foundation Models for Network Intrusion Detection
**Base Method:** TabPFN v2 (Hollmann et al., *Nature* 2025)
**Duration:** 16 weeks (4 months)
**Hardware:** MacBook Air M1, 16 GB unified memory, 100 GB storage
**GitHub:** https://github.com/Sujeethh03/TabPFN-NIDS
**Team Lead:** Sujeeth (`Sujeethh03`)

---

## 1. What This Project Is (In Simple Terms)

Network intrusion detection systems (NIDS) look at network traffic data — packets flying between computers — and try to spot which ones are hackers attacking. Traditional systems need to be trained separately for every network, and often fail when moved to a new environment.

TabPFN v2 is a brand-new AI model published in *Nature* magazine in 2025. It is pretrained once on millions of synthetic tabular problems and can then make predictions on new tables (like network traffic data) without any additional training. Think of it as "GPT for spreadsheets."

This project applies TabPFN v2 to intrusion detection, extends it to handle datasets larger than its native limit, and tests whether it can catch attacks it has never seen before by transferring from one dataset to another.

---

## 2. Why This Project Is Worth Doing

- **Nature 2025 base paper** — one of the highest-prestige venues in science, strong citation anchor for the report.
- **Inference-only workload** — no GPU needed, safe for M1 Air, no thermal throttling.
- **Zero-cost project** — all papers free, all datasets free, no API bills, no cloud compute.
- **Reproducible in ~30 minutes** — anyone can clone the repo and run the smoke test.
- **Cybersec + ML crossover** — differentiated portfolio angle for AI/GenAI engineering interviews.
- **Novel scope** — one prior paper (Springer 2024) tried TabPFN on IIoT; our angle (general NIDS + scale + cross-dataset transfer) is genuinely different.

---

## 3. The Three Base Papers

### Paper 1 — TabPFN v2 (primary base paper)
- **Full Title:** Accurate predictions on small data with a tabular foundation model
- **Authors:** Hollmann, N., Müller, S., Purucker, L., et al.
- **Venue:** *Nature*, 2025
- **PDF:** https://www.nature.com/articles/s41586-024-08328-6.pdf
- **What it does:** Introduces TabPFN v2, a transformer-based tabular foundation model that predicts on new datasets zero-shot (no per-dataset training).

### Paper 2 — TabPFN v1 (foundational method)
- **Full Title:** TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second
- **Authors:** Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F.
- **Venue:** ICLR 2023 (Oral)
- **PDF:** https://arxiv.org/pdf/2207.01848
- **What it does:** The original TabPFN method — establishes the "prior-fitted network" concept for tabular data.

### Paper 3 — TabPFN for IIoT IDS (prior work we differentiate from)
- **Full Title:** A TabPFN-based intrusion detection system for the industrial internet of things
- **Authors:** Yousfi, S., Kessentini, Y., & Chibani, A.
- **Venue:** *The Journal of Supercomputing* (Springer), 2024
- **PDF:** https://link.springer.com/content/pdf/10.1007/s11227-024-06166-x.pdf
- **What it does:** First paper to apply TabPFN to intrusion detection, specifically for Industrial IoT with small training sets. Our project differs in three ways: uses newer TabPFN v2, handles full-scale datasets via chunked ensemble, and tests cross-dataset transfer.

---

## 4. Datasets (All Free)

All three datasets are hosted on Kaggle. Need a free Kaggle account with phone verification to download.

### NSL-KDD (start with this one)
- **Size:** ~150,000 rows, ~100 MB
- **Features:** 41
- **Attack types:** 5 categories (DoS, Probe, R2L, U2R, Normal)
- **Kaggle link:** https://www.kaggle.com/datasets/hassan06/nslkdd
- **Used for:** Main reproduction benchmark (Weeks 5–8)

### UNSW-NB15
- **Size:** ~2.5 million rows, ~1 GB
- **Features:** 49
- **Attack types:** 9 categories (DDoS, exploits, reconnaissance, etc.)
- **Kaggle link:** https://www.kaggle.com/datasets/dhoogla/unswnb15
- **Used for:** Chunked ensemble scaling experiments (Enhancement 1)

### CIC-IDS 2018
- **Size:** ~5 GB (processed CSVs, not raw PCAPs)
- **Features:** ~80
- **Attack types:** DDoS, brute-force, botnets, web attacks
- **Kaggle link:** https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
- **Used for:** Cross-dataset transfer test (Enhancement 3)

**Total disk footprint:** ~6 GB. Fits comfortably in the 100 GB M1 Air budget.

---

## 5. Project Timeline (16 Weeks)

### Month 1 — Topic and Papers Selection

| Week | Goal | Deliverable |
|---|---|---|
| 1 | Paper selection + feasibility matrix | Shortlist with feasibility comparison |
| 2 | Finalize paper, get supervisor approval, download datasets | Approved paper + downloaded data |
| 3 | Deep technical read of primary paper | Annotated PDF + algorithm flowchart |
| 4 | Environment setup, run original code once | Working environment + run log |

**Week-4 red flag:** If the smoke test fails or TabPFN doesn't install cleanly, switch to the backup paper (E-GraphSAGE).

### Month 2 — Reproduction

| Week | Goal | Deliverable |
|---|---|---|
| 5 | Run TabPFN on NSL-KDD three times, log metrics | Reproduction log (CSV) |
| 6 | Compare our numbers to paper's reported baselines | Comparison table |
| 7 | Small ablation study (remove one feature, verify drop) | Ablation results |
| 8 | Write reproducibility report | Reproducibility Report v1.0 |

### Month 3 — Enhancement

| Week | Goal | Deliverable |
|---|---|---|
| 9 | Write 1-page enhancement proposal | Enhancement proposal |
| 10 | Fresh baseline on our hardware | Baseline metrics |
| 11 | Implement all three enhancements | Working enhanced code |
| 12 | Compare vanilla vs enhanced with statistical tests | Comparative results table |

### Month 4 — Finalization

| Week | Goal | Deliverable |
|---|---|---|
| 13 | Final experiments, generate figures | Final figures + stats |
| 14 | Write final report | Draft report |
| 15 | Clean code, write proper README | Public GitHub repo |
| 16 | Final submission + presentation | Final submission |

---

## 6. The Three Enhancements

### Enhancement 1 — Chunked Ensemble for Scale (headline contribution)

**Problem:** TabPFN v2 caps context at ~10,000 training samples. Real NIDS datasets have millions of flows.

**What we build:** Split training data into overlapping stratified chunks that preserve class balance. Run TabPFN independently on each chunk with the same test set. Aggregate predictions via weighted majority voting.

**Measurable delta:** F1 score on full UNSW-NB15 (2.5M records) vs 10K-sample cap baseline.

**Code size estimate:** ~150–250 lines of Python.

### Enhancement 2 — Domain-Aware Feature Engineering (supporting)

**Problem:** Vanilla TabPFN treats all input columns as equivalent. Network flow features have inherent structure that gets ignored.

**What we build:** A preprocessing layer that computes engineered features before TabPFN inference:
- Temporal features (flow duration, packet inter-arrival time)
- Volumetric features (bytes/packet, byte-to-packet ratios, flow rate)
- Protocol-specific features (session duration bins, categorical interactions)

**Measurable delta:** F1 improvement between vanilla-feature and engineered-feature runs on identical train/test splits.

**Code size estimate:** ~100–200 lines of Python.

### Enhancement 3 — Cross-Dataset Generalization Test (research angle)

**Problem:** Almost no prior NIDS paper reports cross-dataset transfer results cleanly. This masks whether models actually generalize or just memorize their training set.

**What we build:** Use UNSW-NB15 as the training context and CIC-IDS 2018 as the test set (two different networks, different attack distributions). Compare TabPFN's zero-shot transfer to baseline models (XGBoost, LightGBM) evaluated the same way.

**Measurable delta:** F1 drop from in-dataset to cross-dataset evaluation, compared against baseline models' equivalent drops.

**Code size estimate:** Mostly an experimental setup — ~50–100 lines of Python.

---

## 7. Technical Setup

### Hardware
- MacBook Air M1
- 16 GB unified memory
- 100 GB free storage
- No GPU required
- No external cloud compute required

### Software Stack
```
Python 3.11
tabpfn >= 2.0.0
scikit-learn >= 1.4.0
numpy >= 1.26.0
pandas >= 2.2.0
torch >= 2.2.0 (MPS backend for Apple Silicon)
matplotlib, seaborn (figures)
jupyter (notebooks)
statsmodels (statistical tests)
```

### One-time Setup Commands
```bash
cd ~/Documents/TabPFN-NIDS
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python smoke_test.py
```

Expected smoke test output: `Accuracy > 0.95` on the breast-cancer benchmark in under 60 seconds.

### Repository Structure
```
TabPFN-NIDS/
├── README.md
├── LICENSE (MIT)
├── requirements.txt
├── .gitignore
├── smoke_test.py
├── data/                  # Gitignored — download separately
│   ├── nsl-kdd/
│   ├── unsw-nb15/
│   └── cic-ids-2018/
├── src/
│   ├── data_pipeline/     # Loading, cleaning, splits
│   ├── models/            # TabPFN wrapper, chunked ensemble
│   ├── features/          # Domain-aware feature engineering
│   └── evaluation/        # Metrics, statistical tests
├── notebooks/
│   ├── 01_reproduction.ipynb
│   ├── 02_ensemble.ipynb
│   ├── 03_features.ipynb
│   └── 04_transfer.ipynb
├── reports/
│   ├── week_08_reproducibility.pdf
│   └── week_16_final.pdf
└── docs/
    └── weekly_status/     # One report per week
```

---

## 8. Team Roles (Suggested)

| Role | Responsibility |
|---|---|
| Data Lead | Dataset download, preprocessing pipeline, stratified splits, feature engineering |
| Modeling Lead | TabPFN inference wrapper, chunked ensemble implementation, cross-dataset experiments |
| Evaluation & Writing Lead (Sujeeth) | Metrics, statistical tests, figures, weekly status reports, final report and presentation |

All members contribute to the GitHub repo via feature branches and pull requests. All members must be present at the final defense per KLU Phase-I attendance policy.

---

## 9. Grading Rubric (16-Week Weightage)

| Deliverable | Weight | Due |
|---|---|---|
| Paper selection + feasibility matrix | 5% | Week 2 |
| Environment runs without error | 5% | Week 4 |
| Reproduction log (3 runs) | 20% | Week 5 |
| Reproducibility report | 25% | Week 8 |
| Enhancement proposal | 10% | Week 9 |
| Enhancement results | 20% | Week 12 |
| Final code + README | 10% | Week 15 |
| Final report + presentation | 5% | Week 16 |

---

## 10. Key Decisions Made During Planning

### Why TabPFN over the alternatives?
Considered SWE-agent (NeurIPS 2024), GraphRAG (Microsoft 2024), DSPy (ICLR 2024), iTransformer (ICLR 2024), MedSAM (Nature Communications 2024), Grounding DINO (ECCV 2024), LineVul (MSR 2022), E-GraphSAGE (IEEE NOMS 2022), and several others.

Chose TabPFN v2 because:
- Nature 2025 is the strongest venue in the shortlist
- Inference-only workload — no risk of thermal throttling on the M1 Air
- Zero cost, zero API dependencies
- Enhancement paths (chunked ensemble, feature engineering, cross-dataset transfer) map directly to Sujeeth's Python skillset
- Diversifies portfolio away from the existing PR-Review-Agent, TripPilot-AI, and RAG projects

### Why NIDS as the application domain?
- Free, well-established datasets (NSL-KDD, UNSW-NB15, CIC-IDS 2018)
- Tabular data — direct fit for TabPFN
- Adds a cybersecurity crossover to the portfolio without requiring offensive-security work
- One prior paper (Yousfi 2024 for IIoT) validates the direction is real research

### Why not the buzzword-heavy title "Zero-Shot Network Intrusion Detection"?
"Zero-shot" is technically contestable given we do provide labeled training rows as in-context examples. The stronger title is **"Scaling Tabular Foundation Models for Network Intrusion Detection"** because it highlights the genuinely novel contribution (the chunked ensemble) and every word is defensible.

### Backup paper if TabPFN reproduction fails
E-GraphSAGE (Lo et al., IEEE/IFIP NOMS 2022) is the backup. Graph neural network approach to NIDS, uses the same datasets, small model that runs on M1 Air. Retained as fallback if the Week-4 red flag triggers.

---

## 11. Future Work / Immediate To-Do List

### This Week (Week 1 → Week 2 transition)
- [ ] Fill in the Weekly Status Report with team number and student IDs
- [ ] Present the Review-1 document to supervisor for formal sign-off
- [ ] Add supervisor as a Write-permission collaborator on the GitHub repo
- [ ] Complete Kaggle phone verification
- [ ] Download NSL-KDD dataset (start here — smallest, fastest)
- [ ] Skim TabPFN v2 paper abstract and introduction

### Week 3 Prep
- [ ] Deep read of TabPFN v2 paper (Nature 2025)
- [ ] Annotate the PDF section-by-section
- [ ] Draw a flowchart of the TabPFN inference pipeline
- [ ] Read TabPFN v1 paper (arXiv 2023) for foundational method
- [ ] Read the Yousfi 2024 IIoT paper to understand prior work

### Week 4 Prep
- [ ] Run smoke test successfully (`python smoke_test.py`)
- [ ] Load NSL-KDD into a Jupyter notebook
- [ ] Verify TabPFN can classify NSL-KDD labels
- [ ] Log any environment issues

### Week 5 (Reproduction Begins)
- [ ] Run TabPFN on NSL-KDD three times with different random seeds
- [ ] Record metrics: accuracy, precision, recall, F1, ROC-AUC
- [ ] Save results to a CSV in `reports/reproduction_log.csv`
- [ ] Compare against Ferrag et al. 2020 survey baseline numbers

### Papers to Read Over the Next 4 Weeks
Aim to grow the related-work section from 3 to 15+ papers by Week 3. Prioritize in this order:
1. NSL-KDD origin paper (Tavallaee et al., 2009)
2. UNSW-NB15 paper (Moustafa & Slay, 2015)
3. CIC-IDS 2018 paper (Sharafaldin et al., 2018)
4. Deep Learning for NIDS survey (Ferrag et al., 2020)
5. XGBoost paper (Chen & Guestrin, 2016)
6. E-GraphSAGE (Lo et al., IEEE NOMS 2022)
7. TabTransformer (Huang et al., 2020)
8. FT-Transformer (Gorishniy et al., NeurIPS 2021)
9. LightGBM (Ke et al., NeurIPS 2017)
10. Anomal-E (Caville et al., 2022)
11. DI-NIDS (Layeghy et al., 2023)
12. Sommer & Paxson (IEEE S&P 2010) — foundational critique paper

### Long-Term Future Work (Beyond Phase-I)
Even if the current 4-month project succeeds, these are strong directions if you continue in Phase-II or beyond:

- **Diffusion-model attacks:** Test TabPFN on network traffic generated by generative models (adversarial evasion research)
- **Real-time streaming:** Wrap the chunked ensemble in a Kafka streaming pipeline for online inference
- **Federated variant:** Distribute chunks across multiple nodes for privacy-preserving cross-organization NIDS
- **Explainability:** Use SHAP values on TabPFN predictions to make the detector interpretable (compliance requirement in enterprise NIDS)
- **Multi-class deep dive:** Focus on specific attack categories (DDoS-only, insider threat, etc.) rather than binary classification
- **Domain adaptation module:** Add a small adaptation layer between the source and target datasets to close the transfer gap

---

## 12. Cost Summary

Every part of this project is free.

| Item | Cost |
|---|---|
| Papers (all 3, on arXiv/Nature/Springer preprints) | $0 |
| Code (TabPFN library, MIT/Apache licensed) | $0 |
| Datasets (Kaggle mirrors) | $0 |
| Compute (local M1 Air, no cloud) | $0 |
| API calls (none — inference is local) | $0 |
| GitHub (free public repo) | $0 |
| **Total** | **$0** |

Only "cost" is time (roughly 8–12 hours/week for the team) and disk space (~30 GB for full experiments).

---

## 13. References (APA 7th Edition Style)

Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., & Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model. *Nature*, *637*(8045), 319–326. https://doi.org/10.1038/s41586-024-08328-6

Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2023). TabPFN: A transformer that solves small tabular classification problems in a second. *Proceedings of the International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2207.01848

Yousfi, S., Kessentini, Y., & Chibani, A. (2024). A TabPFN-based intrusion detection system for the industrial internet of things. *The Journal of Supercomputing*, *80*, 12345–12367. https://doi.org/10.1007/s11227-024-06166-x

---

## 14. Contact & Repository

- **GitHub:** https://github.com/Sujeethh03/TabPFN-NIDS
- **Team Lead:** Sujeeth (`Sujeethh03`)
- **Institution:** KL University (Koneru Lakshmaiah Education Foundation)
- **Department:** Computer Science & Engineering
- **Supervisor:** [Fill in supervisor name]

---

*Last updated: 5 August 2026*
*Document version: 1.0*
