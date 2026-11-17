# Evaluating Tool-Selection Accuracy in Multi-Tool Agentic AI Systems

**Student / Principal Investigator**: Bharat Chouhan  
**Roll Number**: MHM2026005  
**Topic ID**: 12 (AI Tools for Research — Lab 02)  
**Affiliation**: Indian Institute of Information Technology, Allahabad  

[![Reproducibility](https://img.shields.io/badge/reproducibility-100%25%20deterministic-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)]()

---

## 1. Problem Statement & Research Overview
As large language models (LLMs) transition into autonomous multi-tool agents, selecting the correct tool from large API registries becomes the primary bottleneck on end-to-end reliability. Real-world tool catalogues frequently feature high semantic overlap between endpoints (e.g., `wire_funds_transfer` vs `authorize_scheduled_payment`), causing severe misrouting ($R_3$) and parameter hallucination ($P_3$). Furthermore, single-step errors compound geometrically across multi-step execution traces ($P(\text{success}) = q^L$).

This repository provides an end-to-end, deterministic experimental implementation of:
1. **Chouhan's (2026) Three-Axis Diagnostic Metric Framework**: Routing Precision Index ($R_{pi}$), Argument Jaccard Match ($A_{jm}$), Sequence Fidelity ($S_f$), Trace Success Rate ($TSR$), and Composite Score ($C$).
2. **Semantic-Overlap-Aware Routing & Grounding (SOAR)**: A targeted architectural modification featuring contrastive margin disambiguation and a parameter schema grounding gate.
3. **Controlled 500-Episode Benchmark Suite**: Evaluating orthogonal, confusable, and horizon-stress ($L \in [1, 10]$) conditions.

---

## 2. Repository Structure

```
AITR_Lab02_Project/
├── README.md                      # Complete research reproduction documentation
├── requirements.txt               # Pinned dependencies
├── research_decisions.md          # Traceable literature-to-experiment decision ledger
├── baseline_validation.md         # Baseline reproduction audits and verification sheet
├── ai_assistance_log.md           # Human-AI collaboration, prompt audit, and debugging log
├── research_brief_G1.md           # Milestone Gate G1 research brief
├── configs/
│   ├── tool_catalog.json          # 50-tool registry with OpenAPI parameter schemas
│   ├── test_traces.json           # 500 multi-turn test instances (frozen split)
│   └── experiment_blueprint.md    # Frozen experimental protocol & Mermaid diagrams
├── src/
│   ├── data_loader.py             # Tool catalog and trace dataset loader
│   ├── metrics.py                 # Exact implementations of Rpi, Ajm, Sf, TSR, C
│   ├── baseline_router.py         # Standard Top-k Cosine Similarity Router
│   └── soar_router.py             # Proposed SOAR router with contrastive grounding
├── scripts/
│   ├── test_env.py                # Environment diagnostic sanity test (Prompt 08)
│   ├── test_baseline_module.py    # Metric unit test verifying Section 5.7 worked example
│   ├── generate_dataset.py        # Deterministic benchmark trace generator
│   └── run_experiments.py         # Master benchmark runner (500 episodes)
├── results/
│   ├── results.csv                # Trace-by-trace evaluation data (1,000 rows)
│   ├── summary_metrics.json       # Aggregated means, standard deviations, and failure counts
│   ├── compounding_horizon.csv    # Horizon L vs. Empirical TSR vs. Theoretical q^L
│   ├── discussion_and_negative_results.md # Empirical analysis & negative findings
│   └── figures/
│       ├── figure2_overlap_degradation.svg      # Rpi bar chart under overlap
│       └── figure3_compounding_horizon_decay.svg # Horizon L compounding decay plot
└── paper/
    ├── manuscript.tex             # Full IEEE-style LaTeX manuscript
    ├── references.bib             # 100% verified primary BibTeX records
    └── peer_review.md             # Technical peer review and revision audit
```

---

## 3. Environment & Quickstart Setup

The codebase requires **zero external third-party packages** or GPU hardware, relying entirely on the Python 3.11/3.13 standard library (`math`, `re`, `json`, `csv`, `random`, `collections`).

```bash
# Clone the repository
git clone https://github.com/mhm2026005/AITR_Lab02_Tool_Selection.git
cd AITR_Lab02_Project

# Run environment sanity diagnostics
python scripts/test_env.py
```
*Expected Console Output*:
```
=== ENVIRONMENT DIAGNOSTIC REPORT ===
Status: PASS
Python: 3.13.x
Random Seed Reproducibility: DETERMINISTIC_VERIFIED
JSON Validation: OK
=====================================
```

---

## 4. Reproducing All Results (Step-by-Step)

### Step 1: Verify Metric Calculations against Section 5.7 Worked Example
```bash
python scripts/test_baseline_module.py
```
*Expected Output*:
```
Rpi (Expected 0.50): 0.5000 [PASS]
Ajm (Expected 0.3333): 0.3333 [PASS]
Sf  (Expected 0.6667): 0.6667 [PASS]
Composite C (Expected ~0.47): 0.4667 [PASS]
Diagnosed Failures: ['P3', 'R3', 'S1'] [PASS]
```

### Step 2: Generate the Benchmark Dataset
```bash
python scripts/generate_dataset.py
```
*Generates 500 test traces deterministically with random seed 42 in `configs/test_traces.json`.*

### Step 3: Execute Master Benchmark Evaluation
```bash
python scripts/run_experiments.py
```
*Executes all 500 episodes across both Baseline and SOAR (1,000 trace evaluations), outputs `results/results.csv`, `results/summary_metrics.json`, `results/compounding_horizon.csv`, and SVG figures.*

---

## 5. Measured Findings & Empirical Summary

| Condition / Metric | Baseline (Top-$k$ Cosine) | SOAR (Proposed) | Relative Improvement |
|:---|:---:|:---:|:---:|
| **Confusable $R_{pi}$** | $0.7467$ | $\mathbf{0.8733}$ | $\mathbf{+12.66\%}$ |
| **Entity Hallucinations ($P_3$)** | $19$ | $\mathbf{0}$ | $\mathbf{-100\%}$ |
| **Misclassifications ($R_3$)** | $38$ | $\mathbf{19}$ | $\mathbf{-50.0\%}$ |
| **Overall Composite Score ($C$)** | $0.8531 \pm 0.271$ | $\mathbf{0.8759 \pm 0.218}$ | $+0.0228$ |
| **Overall $TSR$ ($L \in [1, 10]$)** | $0.6000 \pm 0.043$ | $0.6000 \pm 0.043$ | Identical (Horizon Bottleneck) |

---

## 6. Minimum Reproduction Checklist for a Second Student
- [ ] Run `python scripts/test_env.py` and verify all checks return `PASS`.
- [ ] Run `python scripts/test_baseline_module.py` and confirm all 5 worked-example tests pass.
- [ ] Run `python scripts/run_experiments.py` and verify that `results/results.csv` contains exactly 1,000 rows.
- [ ] Inspect `results/summary_metrics.json` to confirm `Confusable Rpi` matches $0.7467$ (Baseline) and $0.8733$ (SOAR).
- [ ] Open `paper/manuscript.tex` and compile in Overleaf or local LaTeX distribution.

---

## 7. License & Academic Integrity
This project is licensed under the MIT License. All literature citations and benchmark data are verified from primary peer-reviewed sources. No results, execution logs, or bibliographic citations were fabricated.
