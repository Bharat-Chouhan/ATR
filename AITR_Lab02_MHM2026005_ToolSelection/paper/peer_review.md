# Technical Peer-Review Report & Revision Log (Prompt 16)

**Manuscript Title**: Mitigating Semantic Overlap and Compounding Errors in Multi-Tool Agentic AI: A Contrastive Grounding Approach  
**Author**: Bharat Chouhan (MHM2026005)  
**Reviewer**: Critical Technical Reviewer (Automated Rigorous Audit)  
**Recommendation**: **Accept with Minor Revisions**  

---

## 1. Overall Evaluation & Summary of Manuscript
The manuscript investigates tool-selection accuracy in multi-tool agentic AI, targeting two core challenges: catalogue semantic overlap and multi-step compounding error. It formulates a four-metric diagnostic framework ($R_{pi}, A_{jm}, S_f, TSR, C$) and proposes the SOAR router (Contrastive Disambiguation + Grounding Gate). Experiments across 500 multi-turn episodes validate that SOAR improves confusable routing precision by $+12.66\%$ and eliminates entity hallucinations ($P_3: 19 \to 0$). The manuscript honestly reports a key negative result (conservation of hallucination to omission) and demonstrates the collapse of $TSR$ at $L \ge 7$.

---

## 2. Review Assessment Grid

| Evaluation Dimension | Rating (1–5) | Reviewer Critique |
|:---|:---:|:---|
| **Significance & Novelty** | 4.5 / 5 | Clear, practical contribution addressing genuine failure modes in agent routing. |
| **Literature Coverage** | 5.0 / 5 | Comprehensive coverage (ReAct, Toolformer, Gorilla, ToolLLM, MetaTool, TaskBench, BFCL). |
| **Technical Correctness** | 4.8 / 5 | Mathematical metric formulation is sound and verified against worked examples. |
| **Data Leakage & Fairness** | 5.0 / 5 | Partitioning strictly preserved; test queries avoid verbatim copying of schema descriptions. |
| **Experimental Design** | 4.5 / 5 | 500 episodes across 3 conditions; random seed pinned to 42. |
| **Reproducibility** | 5.0 / 5 | Pure standard library implementation; zero external API dependencies or GPU requirements. |

---

## 3. Major Concerns and Required Remedies

### **Major Concern 1: Binary TSR Insensitivity to Grounding Improvement (Section 5.4)**
- **Issue**: The authors report that overall $TSR$ remains unchanged ($0.6000$ for both Baseline and SOAR), despite SOAR eliminating all 19 entity hallucinations ($P_3: 19 \to 0$). In the current manuscript, this is presented as a neutral outcome.
- **Why It Matters**: To a casual reader, identical $TSR$ suggests the proposed intervention provides zero deployment benefit.
- **Remedy**: The authors must explicitly contrast the real-world operational cost of an unhandled exception (HTTP 400 from omission) versus an irreversible corrupted transaction (executing a bank transfer to a hallucinated destination account).
- **Author Response**: Clarified in Section 5.4 that $P_3$ failures represent destructive errors, whereas $P_2$ errors represent recoverable validation halts that can trigger user clarification.

### **Major Concern 2: Generalizability Beyond Controlled Lexical Overlap (Section 6)**
- **Issue**: The contrastive disambiguation rules rely on keyword discriminators between known confusable pairs. How will this scale to 10,000 uncurated APIs?
- **Why It Matters**: Rule-based contrastive terms risk being brittle if the registry grows combinatorially.
- **Remedy**: Acknowledge this boundary condition in the Limitations section and propose automated LLM-driven contrastive extraction for large-scale registries.
- **Author Response**: Added explicit discussion in Section 6 recommending automated embedding clustering and dynamic LLM contrastive synthesis for registries exceeding $|T| > 1,000$.

---

## 4. Minor Concerns
1. **Section 4**: State the exact random seed ($42$) and Python version (3.13) in the text for absolute auditability.
2. **Table 1**: Include standard deviation for the delta column or specify that deltas reflect paired-sample means.
3. **Figure Reference**: Ensure references to Figure 2 (Overlap Degradation) and Figure 3 (Compounding Horizon Decay) match generated SVG assets in `results/figures/`.

---

## 5. Prioritized Author Revision Table

| Issue ID | Concern Type | Location | Author Revision & Action Taken | Status |
|:---:|:---:|:---|:---|:---:|
| **REV-01** | Major | Sec. 5.4 | Added discussion on operational safety: distinguishing destructive entity hallucinations ($P_3$) from recoverable omissions ($P_2$). | **RESOLVED** |
| **REV-02** | Major | Sec. 6 | Added limitation regarding automated scaling of contrastive rules to 10,000+ API registries. | **RESOLVED** |
| **REV-03** | Minor | Sec. 4 | Pinned random seed ($42$) and pure Python runtime environment details in experimental setup. | **RESOLVED** |
| **REV-04** | Minor | Sec. 5.1 | Clarified Table 1 confidence intervals and sample size ($N = 500$). | **RESOLVED** |
