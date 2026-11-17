# Baseline Validation & Dataset Verification Sheet (Prompt 06)

**Student**: Bharat Chouhan (MHM2026005)  
**Task / Domain**: Evaluating Tool-Selection Accuracy in Multi-Tool Agentic AI Systems  
**Status**: VERIFIED & AUDITED  

---

## 1. Dataset Verification & Access Audit

| Dataset / Repository | Official Source | File Structure & Format | Licensing / Terms | Compute Burden | Common Pitfalls & Leakage Risks | Verification Status |
|:---|:---|:---|:---|:---|:---|:---|
| **Synthesized ToolBench / MetaTool Subset** | Derived from MetaTool (Huang et al., 2024, arXiv:2310.03128) and ToolBench (Qin et al., 2024, ICLR). | JSON format: `tool_catalog.json` (50 tools with full parameter schemas), `test_traces.json` (500 multi-turn test instances). | Open research license (Apache 2.0 / MIT compatible). | Pure CPU; deterministic; < 5 MB RAM; executes in seconds. | **Leakage Risk**: Test query verbatim matching tool documentation. **Mitigation**: Queries paraphrase intents without copying tool names or argument keys verbatim. | **[VERIFIED]** |

---

## 2. Comparators & Controls

1. **Control / Baseline 1: Standard Semantic Retriever (Top-$k$ Cosine)**
   - Encodes query and tool descriptions in term-frequency feature space.
   - Selects candidate based on highest raw cosine similarity.
   - Parameter extractor assigns values based on loose keyword extraction.
   - Known limitation: Highly susceptible to failure $R_3$ (misclassifying confusable pairs) and $P_3$ (hallucinating missing parameters).
2. **Comparator 2: Rule-Based Keyword Matcher**
   - Exact substring and token intersection matching.
   - High precision on verbatim queries; near-zero recall on paraphrased or indirect queries.
3. **Proposed System: SOAR (Semantic-Overlap-Aware Routing & Grounding)**
   - Implements contrastive boundary discrimination between confusable pairs.
   - Explicit parameter typing and presence verification prior to dispatch.

---

## 3. Independent Verification Checklist
- [x] Tool definitions contain strictly valid JSON schemas matching OpenAPI / REST standards.
- [x] Zero data leakage: Tool documentation is strictly partitioned from evaluation test query generation.
- [x] Random seed pinned to 42 for all stochastic sampling.
- [x] Execution requires zero network connections or private tokens, satisfying complete auditability.
