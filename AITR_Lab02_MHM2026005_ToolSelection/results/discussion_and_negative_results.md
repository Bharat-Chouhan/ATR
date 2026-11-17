# Experimental Results Interpretation & Negative Findings (Prompt 14)

**Paper**: Evaluating Tool-Selection Accuracy in Multi-Tool Agentic AI Systems  
**Author**: Bharat Chouhan (MHM2026005)  
**Date**: 2026-11-17  

---

## 1. Summary of Measured Findings

| Evaluation Metric | Baseline (Top-$k$ Cosine) | Proposed SOAR (Contrastive + Grounding) | Relative Delta ($\Delta$) | Empirical Interpretation |
|:---|:---:|:---:|:---:|:---|
| **Overall Composite Score ($C$)** | $0.8531 \pm 0.2712$ | $\mathbf{0.8759 \pm 0.2187}$ | $+0.0228$ | Lower variance indicates superior system stability. |
| **Orthogonal $R_{pi}$** | $1.0000$ | $1.0000$ | $0.0000$ | Both methods achieve ceiling routing on distinct toolsets. |
| **Confusable Overlap $R_{pi}$** | $0.7467$ | $\mathbf{0.8733}$ | $\mathbf{+12.66\%}$ | Contrastive disambiguation resolves ambiguous lexical overlap. |
| **Trace Success Rate ($TSR$)** | $0.6000 \pm 0.0429$ | $0.6000 \pm 0.0429$ | $0.0000$ | Constrained by long-horizon compounding failure ($L \ge 7$). |
| **Entity Hallucinations ($P_3$)** | $19$ occurrences | $\mathbf{0}$ occurrences | $\mathbf{-100\%}$ | Schema Grounding Gate completely halts fabricated entities. |
| **Misclassifications ($R_3$)** | $38$ occurrences | $\mathbf{19}$ occurrences | $\mathbf{-50.0\%}$ | Misroutes between confusable tools halved. |
| **Argument Omissions ($P_2$)** | $162$ occurrences | $181$ occurrences | $+11.7\%$ | **Negative Result**: Hallucinations converted to omissions. |

---

## 2. In-Depth Analysis of Research Questions

### **RQ1: Sensitivity to Catalogue Semantic Overlap**
- **Observation**: When tested on orthogonal tools, both the baseline and SOAR achieved an $R_{pi}$ of $1.0000$. However, when candidate tools exhibited high pairwise semantic overlap ($\ge 0.75$), baseline $R_{pi}$ dropped steeply to $0.7467$ (a $25.33\%$ degradation).
- **Mechanism**: Dense and term-frequency retrievers weight shared domain vocabulary ("account", "balance", "transfer", "order") uniformly. Without contrastive differentiation, slight variations in prompt phrasing cause the router to pick the wrong tool ($R_3$ misclassification).
- **Hypothesis Confirmation**: $H_1$ is **confirmed** ($> 25\%$ degradation observed).

### **RQ2: Intervention via SOAR Contrastive Disambiguation**
- **Observation**: SOAR boosted confusable routing precision to $0.8733$ ($+12.66\%$), reducing misclassifications from $38$ down to $19$, and eliminated all $19$ entity hallucinations ($P_3 = 0$).
- **Mechanism**: Contrastive weighting over distinguishing operational tokens (e.g. "immediate wire" vs "scheduled recurring", "available liquid cash" vs "long-term credit profile") breaks score ties when top candidates have narrow margins.
- **Hypothesis Confirmation**: $H_2$ is **partially confirmed**; while routing and entity hallucination improved significantly, the overall Composite Score gain was $+0.0228$ across the aggregate 500 episodes because orthogonal traces were already at ceiling.

### **RQ3: Compounding Error Across Trace Horizons**
- **Observation**: Empirical $TSR$ remained $1.0$ for $L \in \{1, 2, 3, 5\}$, but dropped to $0.0000$ at $L = 7$ and $L = 10$.
- **Mechanism**: In longer horizons, tasks introduce multi-parameter tool calls (e.g., `send_email_notification` requiring recipient, subject, and body). When prompts contain incomplete contextual arguments, single-step failure cascades, driving end-to-end trace success to zero.
- **Hypothesis Confirmation**: $H_3$ is **confirmed**. Empirical degradation exhibits a cliff-like collapse rather than gradual geometric decay, illustrating that cascading failures in agentic systems are even more unforgiving than analytical independence models predict.

---

## 3. Honest Report of Negative Results & Trade-Offs

> [!WARNING] Negative Finding: The Hallucination-Omission Conservation Law
> The Schema Grounding Gate successfully eliminated entity hallucinations ($P_3: 19 \to 0$). However, this caused Argument Omissions to rise ($P_2: 162 \to 181$). In existing benchmarks with binary evaluation ($TSR$), an omitted required argument produces a failed trace just like a hallucinated argument does.
> 
> **Why this matters for production agents**: In production, an omitted parameter produces a safe HTTP 400 validation error that can prompt a clarification question, whereas a hallucinated parameter (e.g., executing a bank transfer to a fabricated destination account) produces catastrophic, irreversible damage. Traditional binary benchmark metrics fail to credit this safety improvement.

---

## 4. Top Three Recommended Follow-Up Experiments
1. **Interactive Disambiguation Clarification (Low Cost)**: Implement a user-in-the-loop fallback when $P_2$ is triggered to test whether $TSR$ recovers to $> 0.85$ at $L = 7$.
2. **Dense Neural Bi-Encoder vs Contrastive SOAR (Medium Cost)**: Benchmark SOAR against BGE-large and Contriever embeddings to test whether neural embeddings resolve $R_3$ without rule-based contrastive terms.
3. **Adversarial Tool Injection (Medium Cost)**: Introduce malicious decoy tools with identical descriptions but conflicting parameters to stress-test the schema grounding boundary.
