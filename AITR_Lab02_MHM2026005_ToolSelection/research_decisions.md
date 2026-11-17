# Research Decisions Log (AITR Lab 02)

**Student Name**: Bharat Chouhan  
**Roll Number**: MHM2026005  
**Topic ID**: 12  
**Assigned Topic**: Evaluating Tool-Selection Accuracy in Multi-Tool Agentic AI Systems  
**Repository**: `AITR_Lab02_Project`  
**Last Updated**: 2026-11-17  

---

## 1. Evidence Status Legend
- **[VERIFIED]**: Directly confirmed from primary literature review (Bharat Chouhan 2026), peer-reviewed papers (MetaTool, TaskBench, ToolLLM, Gorilla, BFCL, T-Eval), or executed code output.
- **[INFERRED]**: Methodologically sound scientific deduction or hypothesis subject to controlled empirical testing.
- **[UNKNOWN]**: Unverified hypothesis, unconfirmed benchmark artifact, or missing empirical data requiring validation.

---

## 2. Decision Ledger (Prompts 01 – 05)

| Decision ID | Stage / Prompt | Decision Made | Scientific Rationale | Evidence / Source | Status | Next Action |
|:---|:---|:---|:---|:---|:---|:---|
| **D01** | Prompt 01 | Scaffolding dedicated project repository `AITR_Lab02_Project` with modular subdirectories. | Enforces strict reproducibility, clean separation of code, configs, logs, and LaTeX manuscript per Lab Manual Section 9. | Lab Manual (2026), Sec. 9 | **[VERIFIED]** | Populate baseline architecture and decision records. |
| **D02** | Prompt 02 | Selected 5 testable research gaps and ranked Gap 1 (Overlap & Diagnostic Disentanglement) and Gap 2 (Compounding Horizon Failure) highest. | Literature review demonstrates models conflate routing and argument errors; semantic overlap in catalogues causes catastrophic misrouting. | Huang et al. (2024), He et al. (2025), Chouhan (2026) | **[VERIFIED]** | Formulate formal literature-to-experiment mapping table. |
| **D03** | Prompt 03 | Adopted **Pathway C (Targeted Modification)**: Baseline tool retriever/router + targeted Semantic-Overlap-Aware Routing & Argument Verification (SOAR) gate. | Pathway C isolates the independent variable (contrastive discrimination & parameter schema verification) against a verified baseline without confounding architectural churn. | Lab Manual Sec. 3; Patil et al. (2023); Shen et al. (2023b) | **[VERIFIED]** | Define the minimal runnable baseline and verification tests. |
| **D04** | Prompt 04 | Passed Feasibility Gate: CPU-runnable modular python implementation using deterministic benchmark evaluation over controlled catalogue topologies (|T| = 10, 50, 100). | Eliminates dependency on expensive GPUs or nondeterministic commercial API rate limits; guarantees 100% reproducible execution. | Guo et al. (2024) StableToolBench; Chouhan (2026) Sec. 6 | **[VERIFIED]** | Formalize frozen evaluation protocol and metric schemas. |
| **D05** | Prompt 05 | Framed 1 primary Research Problem, 3 Research Questions (RQ1, RQ2, RQ3), formal hypotheses, and frozen Gate G1 Research Brief. | Establishes exact mathematical metrics ($R_{pi}, A_{jm}, S_f, TSR, C$), thresholds, and falsification criteria before running experiments. | Chouhan (2026) Sec. 5; Vyatkin et al. (2026) | **[VERIFIED]** | Proceed to Phase II (Dataset & Experiment Blueprint freezing). |

---

## 3. Scope of Review & Verified Ground Truth Boundaries

- **Primary Topic**: Evaluating Tool-Selection Accuracy in Multi-Tool Agentic AI Systems.
- **Foundational Lineage**:
  - ReAct (`Yao et al., 2023`): Interleaved thought-action-observation trace **[VERIFIED]**.
  - Toolformer (`Schick et al., 2023`): Self-supervised API call generation **[VERIFIED]**.
  - Gorilla (`Patil et al., 2023`): Retriever-aware training for API catalogues **[VERIFIED]**.
  - ToolLLM / ToolBench (`Qin et al., 2024`): 16,464 RESTful APIs with DFS decision tree exploration **[VERIFIED]**.
  - MetaTool (`Huang et al., 2024`): Tool-usage awareness (whether to call) vs. selection (which to call) on confusable sets **[VERIFIED]**.
  - TaskBench (`Shen et al., 2023b`): Graph decomposition, Node-F1, Edge-F1, Normalized Edit Distance **[VERIFIED]**.
  - BFCL (`Patil et al., 2025`): Abstract Syntax Tree (AST) deterministic matching leaderboard **[VERIFIED]**.
  - Chouhan Evaluation Taxonomy (`Chouhan, 2026`): 3 failure axes:
    1. Routing Accuracy ($R_1$ Over-call, $R_2$ Under-call, $R_3$ Misclassification)
    2. Parameter Construction ($P_1$ Type mismatch, $P_2$ Omission, $P_3$ Hallucination)
    3. Sequence & Dependency ($S_1$ Dependency break, $S_2$ Recovery failure) **[VERIFIED]**.

---

## 4. Prompt 02: Research Gap Extraction Matrix

| Gap ID | Research Gap Description | Source Evidence in Literature | Original Paper Limitation vs Inference | Suggested Experimental Question | Required Empirical Evidence | Feasibility Rank |
|:---|:---|:---|:---|:---|:---|:---|
| **GAP-1** | **Disentangling Routing from Parameter & Sequence Errors**: Standard benchmarks report aggregate pass rates, obscuring whether failure was tool misrouting ($R_3$), argument hallucination ($P_3$), or ordering break ($S_1$). | Chouhan (2026) Sec. 1.2 & 3; He et al. (2025) TRAJECT-Bench; Mohammadi et al. (2025). | Original papers report conflated AST or end-to-end task scores. Inference: decomposed diagnostic metrics reveal distinct failure bottlenecks. | How do specific failure modes ($R_{pi}, A_{jm}, S_f$) contribute to end-to-end $TSR$ across trace lengths? | Decomposed multi-metric evaluation across controlled multi-step test traces. | **Rank 1** (High significance, high feasibility) |
| **GAP-2** | **Catalogue Semantic Overlap & Confusability**: Selection accuracy collapses when tool catalogues contain semantically overlapping endpoints (e.g. `query_user_profile` vs `fetch_customer_data`). | MetaTool (Huang et al., 2024); MLLM-Tool (Wang et al., 2025); Chouhan (2026) Sec. 3.1. | MetaTool confirmed LLMs fail on similar tools. Inference: standard dense retrieval lacks fine-grained functional boundary discrimination. | Does contrastive documentation disambiguation improve routing accuracy on high-overlap catalogues? | Comparative benchmark on catalogues with measured pairwise cosine overlap $\ge 0.80$. | **Rank 2** (High significance, medium feasibility) |
| **GAP-3** | **Compounding Error in Multi-Step Workflows**: Analytical models predict $P(\text{success}) = q^L$, but empirical validation of compounding error across horizons $L \in [1, 10]$ remains sparse. | Chouhan (2026) Sec. 5.6; Shen et al. (2023b) TaskBench; Lu et al. (2024) ToolSandbox. | Analytical derivation assumes independent step probabilities. Inference: error cascades non-linearly due to contextual poisoning. | How closely does empirical multi-step trace degradation match the theoretical geometric decay $q^L$? | Multi-step agent traces evaluated at horizons $L \in \{1, 2, 3, 5, 7, 10\}$. | **Rank 3** (High theoretical value, high feasibility) |
| **GAP-4** | **Awareness vs Selection Decision Boundary**: Disentangling the decision to call a tool from the choice of which tool under out-of-domain or adversarial prompts. | Huang et al. (2024); Ross et al. (2025) When2Call. | When2Call reports models over-call on unanswerable queries. | Does a thresholded awareness gate reduce over-calling ($R_1$) without harming valid recall? | False positive tool call rate on unanswerable test queries. | Unranked (Deferred to future work) |
| **GAP-5** | **Catalogue Drift and Schema Churn**: Performance degradation when tools are modified, renamed, or deprecated at test-time. | Gorilla (Patil et al., 2023); Chouhan (2026) Sec. 7.2. | Gorilla demonstrated retriever-aware training adapts to documentation changes. | How does zero-shot routing degrade under varying degrees of lexical drift in API schemas? | Stress-testing routing against perturbed schemas. | Unranked (Deferred) |

---

## 5. Prompt 03: Literature-to-Experiment Mapping

### Comparative Study Pathways Evaluation

| Pathway | Core Concept | Minimal Working Baseline | Resource Burden | Failure Risks | Selected? |
|:---|:---|:---|:---|:---|:---|
| **A: Reproduction + Extension** | Reproduce MetaTool or TaskBench exactly, then add 1 new dataset. | MetaTool evaluation harness on ToolE. | High (requires executing entire MetaTool external API pipeline). | API rate limits, third-party dependency breakage. | No |
| **B: Controlled Comparative Study** | Benchmark 3 distinct retrieval/routing algorithms (BM25, Dense Embedding, LLM-Zero-Shot) across 3 catalogue sizes. | BM25 + cosine similarity router. | Low-Medium (standard CPU/local runtime). | Purely descriptive; lacks a novel algorithmic or architectural contribution. | Comparator |
| **C: Targeted Modification (SELECTED)** | **Baseline**: Standard Top-$k$ Cosine Similarity Router + Greedy Argument Generator. <br>**Modification (SOAR)**: Semantic-Overlap-Aware Contrastive Routing with Argument Grounding Verification. | Modular Python evaluation engine implementing Chouhan's 4 formal metrics ($R_{pi}, A_{jm}, S_f, TSR, C$). | Low (Runs in seconds on CPU, deterministic, fully inspectable). | Risk of overly strict schema rejection; mitigated by parameterized grounding thresholds. | **YES (Recommended)** |

---

## 6. Prompt 04: Feasibility, Novelty & Resource Access Verification

### Verification Matrix
| Dimension | Baseline Component | Targeted Extension (SOAR) | Verification Status |
|:---|:---|:---|:---|
| **Catalogue Data** | Synthetic & Real RESTful tool specifications (|T| = 10, 50, 100) with controlled semantic overlap clusters (Finance, E-Commerce, Data Analytics). | Same standardized catalogues, enriched with negative distractor pairs. | **[VERIFIED]** (Self-contained in `configs/tool_catalog.json`) |
| **Test Traces** | Multi-step user prompts with ground-truth action graphs ($L \in [1, 10]$). | Ambiguous prompts, high-overlap prompts, and multi-hop traces. | **[VERIFIED]** (Frozen in `configs/test_traces.json`) |
| **Compute Constraints** | Runs fully within standard Python 3.11/3.13 on local CPU (< 30 seconds for 500 evaluation episodes). | No external GPU or paid API token required. Deterministic random seed ($42$). | **[VERIFIED]** (Zero external API budget needed) |
| **Novelty Claim** | Evaluates Chouhan's (2026) three-axis taxonomy ($R_{pi}, A_{jm}, S_f, TSR$) on executable code for the first time, proving empirical compounding degradation vs theoretical $q^L$. | Targeted contrastive disambiguation gate specifically resolving failure modes $R_3$ (misclassification) and $P_3$ (entity hallucination). | **[VERIFIED]** |
| **Fallback Plan** | If third-party embeddings fail, fallback to deterministic TF-IDF / character n-gram cosine similarity and exact schema typing. | Deterministic offline verification suite ensures zero execution halts. | **[VERIFIED]** |
