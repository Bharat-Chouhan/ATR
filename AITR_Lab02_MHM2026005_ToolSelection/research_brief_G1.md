# Gate G1: Research Problem Brief

**Student Name**: Bharat Chouhan  
**Roll Number**: MHM2026005  
**Topic ID**: 12  
**Date**: 2026-11-17  
**Gate**: G1 (Research Brief Approval)  

---

## 1. Review Topic & Key Papers (Verified Sources)
- **Topic**: Evaluating Tool-Selection Accuracy in Multi-Tool Agentic AI Systems.
- **Key Verified Papers**:
  1. *MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use* (Huang et al., 2024, arXiv:2310.03128) **[VERIFIED]**
  2. *TaskBench: Benchmarking Large Language Models for Task Automation* (Shen et al., 2023b, NeurIPS 2024, arXiv:2311.18760) **[VERIFIED]**
  3. *Gorilla: Large Language Model Connected with Massive APIs* (Patil et al., 2023, arXiv:2305.15334) **[VERIFIED]**
  4. *Evaluating Tool-Selection Accuracy in Multi-Tool Agentic AI Systems: A Taxonomy, Metric Framework, and Evaluation Protocol* (Chouhan, 2026) **[VERIFIED]**
  5. *When2Call: When (not) to Call Tools* (Ross et al., 2025, arXiv:2504.18851) **[VERIFIED]**

---

## 2. Research Problem Statement
In multi-tool agentic AI systems, tool selection accuracy degrades precipitously when tool catalogues contain high semantic overlap, and single-step routing errors compound multiplicatively across long-horizon execution traces. Current benchmark frameworks conflate routing errors, argument hallucinations, and sequencing violations into monolithic task success metrics. This study investigates: **How can multi-tool agents mitigate routing misclassification and parameter hallucination under semantically overlapping tool catalogues, and how does empirical failure compounding correlate with analytical error-chain models across trace horizons?**

---

## 3. Testable Research Questions & Hypotheses

### **RQ1 (Catalogue Overlap Sensitivity)**
- **Question**: To what extent does pairwise semantic overlap between candidate tool descriptions in a registry degrade the Routing Precision Index ($R_{pi}$)?
- **Hypothesis $H_1$**: As mean pairwise cosine similarity among tool descriptions increases from $\le 0.30$ (orthogonal) to $\ge 0.80$ (confusable clusters), baseline routing precision ($R_{pi}$) drops by $> 25\%$, with misclassification ($R_3$) accounting for $> 70\%$ of routing errors.
- **Falsification Condition**: $R_{pi}$ remains stable ($< 5\%$ variance) across orthogonal and confusable catalogue clusters.

### **RQ2 (Targeted Intervention Effectiveness - SOAR)**
- **Question**: Does introducing a Semantic-Overlap-Aware Contrastive Routing and Schema Grounding Gate (SOAR) improve the Composite Tool-Selection Score ($C$) over standard Top-$k$ semantic similarity routing?
- **Hypothesis $H_2$**: SOAR significantly improves Composite Score $C$ (threshold improvement $\Delta C \ge 0.15$) by actively penalizing description ambiguity and verifying argument grounding prior to dispatch.
- **Falsification Condition**: SOAR produces $\Delta C \le 0.05$ or increases over-calling / under-calling errors.

### **RQ3 (Compounding Horizon Degradation)**
- **Question**: How accurately does the analytical geometric error model $P(\text{success}) = q^L$ predict empirical Trace Success Rate ($TSR$) across trace lengths $L \in \{1, 2, 3, 5, 7, 10\}$?
- **Hypothesis $H_3$**: Empirical $TSR(L)$ exhibits faster-than-geometric decay ($TSR_{\text{empirical}}(L) < q^L$) due to error cascades where early misroutes corrupt downstream execution state.
- **Falsification Condition**: Empirical $TSR$ decays strictly at or slower than $q^L$.

---

## 4. Study Pathway & Experimental Variables
- **Study Pathway**: **Pathway C: Targeted Modification**.
- **Independent Variables**:
  1. Catalogue Overlap Condition: Orthogonal ($\text{sim} < 0.35$) vs. Confusable ($\text{sim} \ge 0.75$).
  2. Routing Architecture: Baseline (Top-$k$ Cosine Matching) vs. SOAR (Contrastive Disambiguation + Grounding Gate).
  3. Trace Length: $L \in \{1, 2, 3, 5, 7, 10\}$.
- **Dependent Variables (Chouhan Metric Framework)**:
  1. Routing Precision Index ($R_{pi} = \frac{|A \cap P|}{|A \cup P|}$)
  2. Argument Jaccard Match ($A_{jm} = \frac{\sum I[v_{\text{act}} = v_{\text{pred}}]}{|K_a \cup K_p|}$)
  3. Sequence Fidelity ($S_f = 1 - \frac{\text{Lev}(A, P)}{\max(|A|, |P|)}$)
  4. Trace Success Rate ($TSR = \frac{1}{N}\sum I[\tau_i \models \phi]$)
  5. Composite Score ($C = 0.40 R_{pi} + 0.40 A_{jm} + 0.20 S_f$)

---

## 5. Computing, Data, and Risk Management
- **Environment**: Standalone Python 3.11/3.13 standard library (zero external API cost; zero network dependencies; fully deterministic with seed 42).
- **Execution Budget**: 500 multi-step test episodes; execution time < 10 seconds.
- **Data Partitions**: 60% Train/Calibration, 20% Validation, 20% Frozen Test. Zero data leakage across episode seeds.
- **First Executable Milestone**: Automated environment sanity check and baseline single-step routing test.
