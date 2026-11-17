# AI Assistance & Research Integrity Report (Prompt 18)

**Student Name**: Bharat Chouhan  
**Roll Number**: MHM2026005  
**Topic ID**: 12 (Evaluating Tool-Selection Accuracy in Multi-Tool Agentic AI Systems)  
**Date**: 2026-11-17  
**Submission Gate**: G4 Research Provenance & Final Audit  

---

## 1. Executive Summary & Provenance Statement
This study was conducted with an AI research assistant functioning as an interactive pair programmer, scaffolding generator, and audit partner. In accordance with laboratory integrity guidelines, all generated code was executed in a sandboxed environment, all mathematical metrics were verified against published analytical worked examples, and all citations were checked against primary index databases (arXiv, ACM Digital Library, IEEE Xplore, ACL Anthology). No experimental outputs, p-values, or bibliographic records were fabricated.

---

## 2. Comprehensive 18-Prompt Traceability Log

| Prompt # | Lifecycle Stage | Primary Task | AI Proposal / Scaffold | Human Decision & Verification Method | Observed Result & Artifact |
|:---:|:---|:---|:---|:---|:---|
| **01** | Phase I: Review to Question | Set up accountable research workspace | Proposed modular repo directory tree and evidence status schema (`[VERIFIED]`, `[INFERRED]`, `[UNKNOWN]`). | Audited against Lab Manual Section 9; approved directory structure. | Scaffold created at `AITR_Lab02_Project/`. |
| **02** | Phase I: Review to Question | Extract evidence-based research gaps | Identified 5 gaps across routing, arguments, and sequence dependencies. | Selected Gap 1 (disentangling failure axes) and Gap 2 (catalogue overlap). | `research_decisions.md` (Sec. 4). |
| **03** | Phase I: Review to Question | Map literature to experiment | Evaluated Pathways A, B, and C. Proposed Pathway C (Targeted Modification: SOAR). | Approved Pathway C; defined minimal runnable baseline and intervention. | `research_decisions.md` (Sec. 5). |
| **04** | Phase I: Review to Question | Check feasibility & novelty | Proposed Python implementation. Tested environment dependencies. | Confirmed local CPU feasibility without external API tokens or GPU costs. | `research_decisions.md` (Sec. 6). |
| **05** | Phase I: Review to Question | Frame RQs and success criteria | Drafted 3 formal RQs, mathematical hypotheses ($H_1, H_2, H_3$), and falsification rules. | Verified alignment with Bharat Chouhan (2026) paper; approved Gate G1 brief. | `research_brief_G1.md`. |
| **06** | Phase II: Experiment Plan | Verify datasets & comparators | Proposed 50-tool catalogue schema and baseline comparators. | Verified schema alignment with OpenAPI/REST standards; checked data leakage risks. | `baseline_validation.md`. |
| **07** | Phase II: Experiment Plan | Freeze experiment blueprint | Authored Mermaid system architecture, metric formulas, and pre-experiment checklist. | Locked metric weights ($w_1=0.4, w_2=0.4, w_3=0.2$) and random seed ($42$) prior to runs. | `configs/experiment_blueprint.md`. |
| **08** | Phase III: Coding & Validation | Configure & test coding env | Created `scripts/test_env.py` to check Python runtime, random seed repeatability, and JSON. | Executed script in console; confirmed all checks returned `PASS`. | `scripts/test_env.py`. |
| **09** | Phase III: Coding & Validation | Modular vibe-coding of baseline | Implemented `src/metrics.py` ($R_{pi}, A_{jm}, S_f, TSR, C$). | Verified against Section 5.7 worked numerical example to 4 decimal places. | `src/metrics.py`. |
| **10** | Phase III: Coding & Validation | Verify & reproduce baseline | Created `scripts/test_baseline_module.py` unit test suite. | Executed test suite; confirmed exact match: $R_{pi}=0.50, A_{jm}=0.3333, S_f=0.6667, C=0.4667$. | `scripts/test_baseline_module.py`. |
| **11** | Phase III: Coding & Validation | Debug using actual evidence | Resolved runtime exceptions and tokenization issues using console tracebacks. | Applied root-cause fixes (path anchoring, token splitting, entity template replacement). | Documented below in Section 3. |
| **12** | Phase IV: Contribution & Evidence | Choose & evaluate contribution | Implemented `src/soar_router.py` with contrastive disambiguation & schema grounding. | Verified that baseline implementation remained preserved for clean A/B comparison. | `src/soar_router.py`. |
| **13** | Phase IV: Contribution & Evidence | Evaluate using real outputs only | Authored `scripts/run_experiments.py` executing 500 episodes across both methods. | Exported 1,000 trace rows to `results/results.csv` and summary JSON. | `results/results.csv`, `results/summary_metrics.json`. |
| **14** | Phase IV: Contribution & Evidence | Interpret outcomes & negative findings | Analyzed results: $+12.66\%$ $R_{pi}$ gain; $-100\%$ $P_3$ hallucinations; horizon collapse at $L \ge 7$. | Evaluated trade-off: hallucination elimination converted into argument omissions ($P_2$). | `results/discussion_and_negative_results.md`. |
| **15** | Phase V: Write & Publish | Draft manuscript in LaTeX | Authored complete IEEEtran manuscript in `paper/manuscript.tex` with verified `.bib`. | Checked cross-references, table numbers, equations, and author affiliations. | `paper/manuscript.tex`, `paper/references.bib`. |
| **16** | Phase V: Write & Publish | Conduct technical peer review | Acted as critical technical reviewer; evaluated significance, correctness, and validity. | Produced prioritized revision matrix addressing $TSR$ sensitivity and scalability limits. | `paper/peer_review.md`. |
| **17** | Phase V: Write & Publish | Prepare GitHub documentation | Authored comprehensive `README.md` with problem statement, quickstart, and reproduction steps. | Verified reproduction checklist for independent students. | `README.md`. |
| **18** | Phase VI: Accountability | Audit human-AI collaboration | Synthesized complete provenance audit and reflective account of human supervision. | Approved final submission package and verified all submission gates G1--G4. | `ai_assistance_log.md`. |

---

## 3. Real Examples of AI Limitations, Manual Corrections, and Verification

### **Example 1: Working Directory and Relative Path Assumptions**
- **AI Proposal**: AI generated `generate_dataset.py` opening files via bare relative paths: `open("configs/tool_catalog.json")`.
- **Failure / Traceback**: Execution failed with `FileNotFoundError: [Errno 2] No such file or directory: 'configs/tool_catalog.json'` when the terminal session ran outside the project folder.
- **Human Verification & Fix**: Diagnosed that scripts must remain location-agnostic. Refactored all path resolutions to use `os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))`. Verified execution succeeded from arbitrary directories.

### **Example 2: Tokenizer Flaw with Underscore Identifiers**
- **AI Proposal**: AI implemented standard regex word tokenization `\b[a-zA-Z0-9_]+\b` over tool names.
- **Failure / Discrepancy**: When evaluating `"query order status for ORD_4002"`, the baseline router ranked `cancel_pending_order` ahead of `query_order_status`. Inspection revealed that `query_order_status` was stored as a single unbroken token (`query_order_status`), giving it zero lexical overlap with the query's individual words `query`, `order`, `status`.
- **Human Verification & Fix**: Modified `BaselineRouter.__init__` to explicitly split tool names and parameter keys by underscores (`t['name'].replace('_', ' ')`). Re-testing immediately corrected `query_order_status` to rank #1 ($0.4819$ similarity).

### **Example 3: Disconnected Prompt-Action Entity IDs in Synthetic Data**
- **AI Proposal**: AI initially generated randomized entity IDs in the action arguments (`ACC_1000`) without replacing the corresponding placeholder in the prompt string (`ACC_9021`).
- **Failure / Discrepancy**: The model accurately extracted `ACC_9021` from the prompt, but $A_{jm}$ failed because the ground truth expected `ACC_1000`.
- **Human Verification & Fix**: Refactored the generator to use unified `{id}` template strings that simultaneously resolve prompt text and expected action arguments, restoring ground-truth integrity.

### **Example 4: Critical Interpretation of the Negative Result**
- **AI Proposal**: AI initially treated identical baseline and SOAR $TSR$ ($0.60 \to 0.60$) as a failure of the intervention.
- **Human Scientific Reasoning**: Human reasoning recognized the "Hallucination-Omission Conservation" trade-off: SOAR eliminated destructive entity hallucinations ($P_3: 19 \to 0$), converting them into argument omissions ($P_2: 162 \to 181$). While a simple binary metric ($TSR$) treats both as identical failures, an operational enterprise system can gracefully recover from an omission via user re-prompting, whereas an entity hallucination executes destructive state mutations. This critical distinction was incorporated into the manuscript's discussion.

---

## 4. Reflections on "Vibe Coding" vs. Supervised Engineering
1. **The Risk of Plausible Illusions**: LLMs readily generate code that runs without errors but calculates incorrect quantities (e.g., treating snake_case identifiers as monolithic tokens). Rigorous unit testing against known numerical baselines (such as Section 5.7) is indispensable.
2. **Deterministic Reproducibility**: Implementing self-contained standard-library modules eliminated environment drift and package mismatch issues, guaranteeing that results can be verified on any machine in seconds.
3. **Accountability of Claims**: AI provides rapid prototyping velocity, but only human supervision ensures that empirical evidence aligns strictly with scientific claims.
