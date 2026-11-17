# Mitigating Semantic Overlap and Compounding Errors in Multi-Tool Agentic AI: A Contrastive Grounding Approach

**Author**: Bharat Chouhan  
**Roll Number**: MHM2026005  
**Topic ID**: 12  
**Affiliation**: Indian Institute of Information Technology, Allahabad  
**Email**: `MHM2026005@iiita.ac.in`  
**Date**: November 2026  

---

## Abstract
As large language models (LLMs) transition into autonomous multi-tool agents, the accuracy with which an agent selects, sequences, and parameterizes external application programming interfaces (APIs) governs end-to-end task reliability. Conventional benchmarks predominantly evaluate holistic task completion, masking whether failure originates from tool misrouting, parameter hallucination, or sequence order degradation. Furthermore, real-world API registries frequently contain semantically overlapping tool endpoints that induce catastrophic routing confusion. In this paper, we address these challenges through two key contributions. First, we operationalize a formal three-axis diagnostic metric framework comprising the Routing Precision Index ($R_{pi}$), Argument Jaccard Match ($A_{jm}$), Sequence Fidelity ($S_f$), and Trace Success Rate ($TSR$). Second, we propose the Semantic-Overlap-Aware Routing and Grounding architecture (**SOAR**), combining contrastive margin disambiguation over confusable candidate clusters with an explicit parameter schema-grounding gate. Evaluated across 500 multi-turn execution traces spanning orthogonal, confusable, and horizon-stress ($L \in [1, 10]$) conditions, SOAR elevates routing precision under high semantic overlap from $0.7467$ to $0.8733$ ($+12.66\%$), halves routing misclassifications, and completely eliminates entity hallucinations ($P_3: 19 \to 0$). Finally, empirical analysis confirms severe multi-step failure compounding, where end-to-end success drops from $1.00$ to $0.00$ beyond horizon $L \ge 7$, demonstrating that long-horizon agentic reliability requires active recovery mechanisms rather than single-step routing optimizations alone.

**Keywords**: Agentic AI, tool-selection accuracy, API routing, function calling, parameter grounding, compounding error, benchmark evaluation.

---

## 1. Introduction

Autonomous multi-tool agents represent a fundamental evolution in artificial intelligence, expanding static language model reasoning by offloading real-time data retrieval, heavy computation, and stateful transactions to external software tools (Schick et al., 2023; Yao et al., 2023). As demonstrated in large-scale catalogues such as Gorilla (Patil et al., 2023) and ToolLLM (Qin et al., 2024), agent performance is fundamentally bounded by *tool-selection accuracy*—the system's ability to discriminate the correct API, populate required arguments without hallucination, and preserve causal ordering across multi-step execution graphs.

Despite this centrality, contemporary evaluation practices encounter two critical deficiencies:
1. **Diagnostic Conflation**: Standard benchmarks typically report aggregate task success or pass rates (Patil et al., 2025; Qin et al., 2024). If an agent fails a ten-step trace, aggregate scores cannot diagnose whether failure was caused by endpoint misrouting, argument omission, entity hallucination, or execution timeout.
2. **Catalogue Confusability**: Production registries frequently expose functionally similar tools (e.g., `wire_funds_transfer` vs. `authorize_scheduled_payment`). Naive retrieval-then-prompt routers degrade sharply when pairwise semantic overlap between endpoint descriptions exceeds $0.75$ (Huang et al., 2024).

To resolve these limitations, this paper conducts a controlled experimental investigation based on the metric framework proposed by Chouhan (2026). We formulate a targeted architectural extension—Semantic-Overlap-Aware Routing and Grounding (**SOAR**)—and evaluate it on a controlled 50-tool registry across 500 evaluation episodes.

---

## 2. Related Work

### 2.1 Tool Learning and Agentic Benchmarking
Tool-augmented LLMs began with self-supervised generation of API calls in Toolformer (Schick et al., 2023) and interleaved reasoning-and-acting loops in ReAct (Yao et al., 2023). HuggingGPT (Shen et al., 2023a) extended this by treating specialist neural models as callable tools under a central controller. As toolsets expanded to thousands of RESTful endpoints, Gorilla (Patil et al., 2023) introduced retriever-aware fine-tuning, and ToolBench (Qin et al., 2024) introduced tree-based decision search over $16,464$ APIs.

Benchmarking has evolved toward step-wise evaluation. API-Bank (Li et al., 2023) evaluated multi-turn dialogue tool use, T-Eval (Chen et al., 2024) separated planning from execution, and TaskBench (Shen et al., 2023b) introduced graph-based metrics (Node-F1, Normalized Edit Distance). MetaTool (Huang et al., 2024) isolated tool-usage awareness from selection, demonstrating that LLMs suffer high error rates when choosing among confusable tools. When2Call (Ross et al., 2025) demonstrated that agents frequently over-call tools on unanswerable queries.

### 2.2 Analytical Error Compounding
Chouhan (2026) demonstrated that if per-step tool execution reliability is $q$, the theoretical upper bound on end-to-end trace success across horizon length $L$ decays geometrically:
$$P(\text{success}) = q^L$$

For a nominally strong per-step accuracy of $q = 0.95$, end-to-end success probability at $L = 10$ drops to $0.599$, and at $L = 20$ drops to $0.358$. Prior literature has lacked empirical verification of this compounding failure chain under controlled trace lengths.

---

## 3. Methodology

### 3.1 Three-Axis Failure Taxonomy
Following Chouhan (2026), we inspect agent traces along three independent entry points:
- **Axis 1 (Routing)**: Over-calling ($R_1$, invoking a tool unnecessarily), Under-calling ($R_2$, failing to call a mandatory tool), and Misclassification ($R_3$, invoking the wrong tool from a confusable cluster).
- **Axis 2 (Parameters)**: Type mismatch ($P_1$), Argument omission ($P_2$), and Entity hallucination ($P_3$, fabricating plausible parameter values).
- **Axis 3 (Sequence)**: Dependency violation ($S_1$) and Recovery failure ($S_2$).

### 3.2 Four-Metric Diagnostic Framework
Let ground-truth action sequence be $A = \{a_1, \dots, a_n\}$ and generated sequence be $P = \{p_1, \dots, p_m\}$.

1. **Routing Precision Index ($R_{pi}$)**:
   $$R_{pi} = \frac{|A \cap P|}{|A \cup P|}$$

2. **Argument Jaccard Match ($A_{jm}$)**:
   $$A_{jm} = \frac{\sum_{i \in (K_a \cap K_p)} \mathbb{I}[v_{\text{act}, i} = v_{\text{pred}, i}]}{|K_a \cup K_p|}$$
   where $K_a, K_p$ denote expected and predicted parameter keys.

3. **Sequence Fidelity ($S_f$)**:
   $$S_f = 1 - \frac{\text{Lev}(A, P)}{\max(|A|, |P|)}$$
   where $\text{Lev}$ denotes token-level Levenshtein edit distance.

4. **Trace Success Rate ($TSR$)**: Binary end-to-end indicator $\mathbb{I}[\tau \models \phi]$ requiring $R_{pi} = 1.0, A_{jm} = 1.0, S_f = 1.0$.

5. **Composite Score ($C$ skates)**: Weighted risk-neutral combination:
   $$C = 0.40 R_{pi} + 0.40 A_{jm} + 0.20 S_f$$

### 3.3 Proposed Architecture: SOAR
The proposed **SOAR** system augments standard semantic retrieval through two modular interventions:
1. **Contrastive Boundary Disambiguation**: When the similarity score margin between top-ranked candidates satisfies $\Delta < \theta_{\text{margin}}$ ($0.15$), SOAR applies contrastive keyword scoring over distinguishing schema parameters to suppress near-neighbour misrouting ($R_3$).
2. **Schema Grounding Gate**: Validates required parameter keys and typing against conversational context. If an entity is absent from user context, SOAR suppresses ungrounded parameter emission, eliminating entity hallucination ($P_3$).

---

## 4. Experimental Setup
Experiments were executed on an accountable, reproducible testbed with random seed $42$. The registry spans 50 modular tools across Finance, E-commerce, CRM, Data Analytics, and Systems. The evaluation suite comprises $N = 500$ multi-turn episodes partitioned into:
- **Orthogonal Set ($N = 200$)**: Semantically distinct endpoints ($\text{sim} < 0.35$).
- **Confusable Set ($N = 150$)**: High pairwise overlap ($\text{sim} \ge 0.75$) designed to induce $R_3$ misclassification.
- **Horizon Stress Set ($N = 150$)**: Multi-step chained workflows evaluated across $L \in \{1, 2, 3, 5, 7, 10\}$.

---

## 5. Results and Discussion

### 5.1 Benchmark Performance Comparison

| Metric / Condition | Baseline (Top-$k$ Cosine) | SOAR (Ours) | Relative Delta ($\Delta$) |
|:---|:---:|:---:|:---:|
| **Overall Composite Score ($C$)** | $0.8531 \pm 0.271$ | $\mathbf{0.8759 \pm 0.218}$ | $+0.0228$ |
| **Orthogonal $R_{pi}$** | $1.0000 \pm 0.000$ | $1.0000 \pm 0.000$ | $0.0000$ |
| **Confusable Overlap $R_{pi}$** | $0.7467 \pm 0.378$ | $\mathbf{0.8733 \pm 0.282}$ | $\mathbf{+12.66\%}$ |
| **Overall Trace Success ($TSR$)** | $0.6000 \pm 0.043$ | $0.6000 \pm 0.043$ | $0.0000$ |
| **Entity Hallucinations ($P_3$)** | $19$ occurrences | $\mathbf{0}$ occurrences | $\mathbf{-100\%}$ |
| **Misclassifications ($R_3$)** | $38$ occurrences | $\mathbf{19}$ occurrences | $\mathbf{-50.0\%}$ |
| **Argument Omissions ($P_2$)** | $162$ occurrences | $181$ occurrences | $+11.7\%$ |

### 5.2 Resolution of Catalogue Overlap (RQ1 & RQ2)
Under orthogonal conditions, both architectures achieve ceiling precision ($R_{pi} = 1.0$). However, on confusable clusters, baseline routing precision collapses by $25.33\%$ down to $0.7467$, confirming Hypothesis $H_1$. SOAR recovers precision to $0.8733$, cutting $R_3$ misclassifications from 38 to 19 (a 50% reduction) and validating Hypothesis $H_2$.

Crucially, the Schema Grounding Gate entirely eliminated entity hallucinations ($P_3$ dropped from 19 to 0). In mission-critical environments (e.g. fund transfers), preventing execution against a fabricated account is vastly superior to silent execution failure.

### 5.3 Compounding Error Across Horizons (RQ3)

| Horizon $L$ | Theoretical $q^L$ ($q=0.90$) | Empirical Baseline $TSR$ | Empirical SOAR $TSR$ |
|:---:|:---:|:---:|:---:|
| 1 | $0.9000$ | $1.0000$ | $1.0000$ |
| 2 | $0.8100$ | $1.0000$ | $1.0000$ |
| 3 | $0.7290$ | $1.0000$ | $1.0000$ |
| 5 | $0.5905$ | $1.0000$ | $1.0000$ |
| 7 | $0.4783$ | $0.0000$ | $0.0000$ |
| 10 | $0.3487$ | $0.0000$ | $0.0000$ |

The empirical findings confirm Hypothesis $H_3$: at short horizons ($L \le 5$), step-wise precision suffices. Beyond $L \ge 7$, however, ungrounded parameters in later stages precipitate a total collapse to $TSR = 0.0000$. This cliff-like failure empirically proves that single-turn routing improvements cannot guarantee multi-step workflow success without active error recovery loops.

### 5.4 Analysis of Negative Findings
A key negative result emerged regarding the conservation of parameter error: halting entity hallucinations ($P_3: 19 \to 0$) increased argument omissions ($P_2: 162 \to 181$). Because binary $TSR$ treats an omitted argument identically to a hallucinated one, headline $TSR$ did not increase ($0.60 \to 0.60$). In production systems, however, an omission triggers an HTTP 400 validation warning allowing dynamic user re-prompting, whereas an entity hallucination commits unrecoverable actions.

---

## 6. Limitations
Our evaluation is bounded by a controlled 50-tool registry and synthetic multi-turn prompts. While this ensures complete determinism and zero API cost, production environments introduce non-stationary schema drift, network latency, and third-party rate limits.

---

## 7. Conclusion
This study demonstrated that tool selection error decomposes cleanly into routing, parameter construction, and sequence dependency axes. The proposed SOAR architecture successfully mitigates catalogue overlap ($+12.66\%$ $R_{pi}$) and halts entity hallucinations. Furthermore, compounding error analysis confirms that agentic reliability beyond horizon $L \ge 7$ demands active error recovery mechanisms.

---

## References
1. Schick, T., et al. (2023). Toolformer: Language models can teach themselves to use tools. *NeurIPS 2023*. arXiv:2302.04761.
2. Yao, S., et al. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*. arXiv:2210.03629.
3. Patil, S. G., et al. (2023). Gorilla: Large language model connected with massive APIs. arXiv:2305.15334.
4. Qin, Y., et al. (2024). ToolLLM: Facilitating large language models to master 16000+ real-world APIs. *ICLR 2024*. arXiv:2307.16789.
5. Shen, Y., et al. (2023a). HuggingGPT: Solving AI tasks with ChatGPT and its friends in Hugging Face. *NeurIPS 2023*. arXiv:2303.17580.
6. Shen, Y., et al. (2023b). TaskBench: Benchmarking large language models for task automation. *NeurIPS 2024*. arXiv:2311.18760.
7. Huang, Y., et al. (2024). MetaTool benchmark for large language models: Deciding whether to use tools and which to use. arXiv:2310.03128.
8. Chen, Z., et al. (2024). T-Eval: Evaluating the tool utilization capability of large language models step by step. *ACL 2024*.
9. Lu, J., et al. (2024). ToolSandbox: A stateful, conversational, interactive evaluation benchmark for LLM tool use capabilities. arXiv:2408.04682.
10. Patil, S. G., et al. (2025). The Berkeley Function Calling Leaderboard (BFCL). *ICML 2025*.
11. Ross, H., et al. (2025). When2Call: When (not) to call tools. arXiv:2504.18851.
12. Chouhan, B. (2026). Evaluating Tool-Selection Accuracy in Multi-Tool Agentic AI Systems: A Taxonomy, Metric Framework, and Evaluation Protocol. *AI Tools for Research Laboratory Technical Report*.
