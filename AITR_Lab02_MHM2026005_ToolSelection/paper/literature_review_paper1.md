# Evaluating Tool-Selection Accuracy in Multi-Tool Agentic AI Systems: A Literature Review

**Student Name**: Bharat Chouhan  
**Roll Number**: MHM2026005  
**Topic ID**: 12  
**Laboratory**: AI Tools for Research — Laboratory 1: AI-Assisted Citation Integrity Audit  
**Date**: 2026-08-21  

---

## Abstract
Large language models (LLMs) are increasingly deployed as agents that must choose among many external tools, APIs, or specialist models to complete a task. As the number of available tools grows into the thousands, the accuracy with which an agent selects, sequences, and parameterizes the correct tool has become a distinct research problem, separate from the reasoning quality of the underlying model. This review synthesizes literature gathered through four AI-assisted research tools—ResearchRabbit, Litmaps, Semantic Scholar, and Elicit—covering tool-augmented LLM frameworks, large-scale API-calling benchmarks, the decision boundary between calling and not calling a tool, multi-agent orchestration, granular and stateful evaluation benchmarks, and recent (2025–2026) work on retrieval-based, planning-based, adaptive, and bias-aware tool selection. The review finds that the field has moved from prompted, single-tool reasoning frameworks toward dedicated retrieval, planning, and routing mechanisms, standardized step-wise and category-based evaluation, and an emerging concern with systematic bias and personalization in tool choice. Persistent gaps remain in evaluating tool selection under long-horizon, stateful, multi-tool workflows, and in disentangling selection accuracy from downstream task success.

**Keywords**: tool-augmented LLMs, tool selection, agentic AI, API calling, benchmark evaluation, multi-tool systems, LLM agents, tool retrieval.

---

## 1. Introduction
The ability of a language model to decide *whether* to call an external tool, *which* tool to invoke, *when* to invoke it, and *with what arguments*, has become central to agentic AI research. Early work established that LLMs could be taught to interleave reasoning with external actions (Yao et al., 2023) and to teach themselves when an API call was necessary (Schick et al., 2023). As the number of available tools has scaled from a handful of hand-picked functions to benchmarks containing over sixteen thousand real-world APIs (Qin et al., 2024), tool selection has shifted from a largely solved sub-problem into an open research question with its own benchmarks, failure modes, and mitigation strategies. This review asks: **how accurately can multi-tool agentic AI systems select the correct tool, and how has the literature evaluated and improved that accuracy?**

---

## 2. Background / Conceptual Foundation
Tool-augmented LLMs extend a language model's native capabilities by allowing it to call external functions, APIs, retrieval systems, or other models mid-generation. Two conceptual lineages converge in this space:
1. **The reasoning-and-acting lineage**: The model produces an interleaved trace of thoughts, actions, and observations rather than a single tool call (Yao et al., 2023).
2. **The tool-teaching lineage**: The model is trained or prompted to recognize the specific point in a task where an external call is needed and to generate a well-formed call (Schick et al., 2023).

Huang et al. (2023) sharpen this further by separating two distinct sub-decisions:
- **Tool-usage awareness**: Should a tool be called at all?
- **Tool selection**: Which tool, among several plausible candidates, is correct?

---

## 3. Thematic Literature Review

### 3.1 Foundational Reasoning-and-Acting and Self-Taught Tool Use
- **ReAct** (Yao et al., 2023): Introduced a prompting paradigm alternating explicit reasoning with actions, reducing hallucinations.
- **Toolformer** (Schick et al., 2023): Self-supervised filtering of candidate API calls during pretraining; demonstrated tool competence without full human supervision.

### 3.2 Large-Scale API Calling and Tool Retrieval
- **Gorilla / APIBench** (Patil et al., 2023): Retriever-aware training (RAT) over 1,645 APIs, mitigating hallucinated arguments.
- **ToolLLM / ToolBench** (Qin et al., 2024): Scaled to 16,464 REST APIs with DFS decision-tree trajectory search.
- **Dynamic Relation Modeling** (Nie & Zhang, 2025): Graph-based relation modeling for context provision.
- **Dynamic Tool Dependency Retrieval** (Patel et al., 2025): Retrieval conditioned on evolving plans.
- **Lookahead-R** (Wu, Guo & Li, 2026): Budget-aware execution-centric planning and world models.

### 3.3 The Decision Boundary: Whether to Call a Tool
- **MetaTool** (Huang et al., 2023/2024): Demonstrated that models struggle to decide tool necessity and suffer high error on confusable tools.
- **When2Call** (Ross et al., 2025): Evaluated when NOT to call tools, showing models frequently over-call on unanswerable queries.

### 3.4 Multi-Model Orchestration
- **HuggingGPT** (Shen et al., 2023): Controller LLM dispatching subtasks to specialized HuggingFace models, exposing compounding delegation errors.

### 3.5 Granular, Stateful, and Category-Based Evaluation
- **T-Eval** (Chen et al., 2024): Step-wise capability decomposition scoring.
- **ToolSandbox** (Lu et al., 2024): Stateful conversational environment mutating in response to actions.
- **MultiCAT-Bench** (Vyatkin et al., 2026): 10 difficulty categories showing tool recall (72–90%) heavily lags precision (88–99%).

---

## 4. Comparison of Existing Benchmarks

| Benchmark | Ref | Scale | Multi-turn? | Stateful? | Dominant Scoring Mode |
|:---|:---|:---|:---:|:---:|:---|
| **API-Bank** | Li et al. (2023) | 73 APIs | Partial | No | Exact match against reference |
| **T-Eval** | Chen et al. (2024) | Diverse | Yes | No | Per-sub-process capability scoring |
| **ToolLLM / ToolBench** | Qin et al. (2024) | 16,464 APIs | Yes | Partial | Pass rate and win rate |
| **ToolSandbox** | Lu et al. (2024) | Conversational | Yes | Yes | Milestone / state assertion |
| **MetaTool** | Huang et al. (2024) | 199 tools (ToolE) | No | No | Awareness & Selection accuracy |
| **TaskBench** | Shen et al. (2023b) | Multi-domain | Yes | Yes | Node-F1, Edge-F1, Normalized Edit Distance |
| **BFCL (v1–v4)** | Patil et al. (2025) | Enterprise APIs | Yes | Yes | Deterministic AST exact match |

---

## 5. Identified Research Gaps
1. **Conflation of Failure Modes**: Most benchmarks report aggregate pass rates, failing to separate routing ($R$), parameter ($P$), and sequence ($S$) failures.
2. **Catalogue Confusability**: High error rates when registries contain near-duplicate, overlapping tools.
3. **Multi-Step Compounding Error**: Lack of empirical verification for the theoretical decay $P(\text{success}) = q^L$.
4. **Tool Churn and Registry Drift**: Limited evaluation of zero-shot routing when tools are modified or deprecated.
