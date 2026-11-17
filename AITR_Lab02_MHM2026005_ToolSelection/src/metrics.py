"""
Diagnostic Evaluation Metrics for Multi-Tool Agentic AI Systems
Implements the 4-metric framework, composite score, and 3-axis failure taxonomy
defined in Chouhan (2026), Section 5.
"""
from typing import List, Dict, Any, Tuple

def compute_levenshtein(seq1: List[str], seq2: List[str]) -> int:
    """Compute Levenshtein edit distance between two sequences of tokens."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def compute_rpi(expected_tools: List[str], predicted_tools: List[str]) -> float:
    """
    Routing Precision Index (Eq. 1 in Chouhan 2026):
    Rpi = |A ∩ P| / |A ∪ P|
    """
    set_a = set(expected_tools)
    set_p = set(predicted_tools)
    if not set_a and not set_p:
        return 1.0
    intersection = len(set_a.intersection(set_p))
    union = len(set_a.union(set_p))
    return intersection / union if union > 0 else 0.0

def compute_ajm(expected_args: Dict[str, Any], predicted_args: Dict[str, Any]) -> float:
    """
    Argument Jaccard Match (Eq. 2 in Chouhan 2026):
    Ajm = sum_{i in Ka ∩ Kp} I[v_act == v_pred] / |Ka ∪ Kp|
    """
    ka = set(expected_args.keys())
    kp = set(predicted_args.keys())
    if not ka and not kp:
        return 1.0
    union_len = len(ka.union(kp))
    if union_len == 0:
        return 1.0
    
    matches = 0
    for k in ka.intersection(kp):
        # Value agreement check
        v_act = expected_args[k]
        v_pred = predicted_args[k]
        if str(v_act).strip().lower() == str(v_pred).strip().lower():
            matches += 1
            
    return matches / union_len

def compute_sf(expected_tools: List[str], predicted_tools: List[str]) -> float:
    """
    Sequence Fidelity (Eq. 3 in Chouhan 2026):
    Sf = 1 - Lev(A, P) / max(|A|, |P|)
    """
    len_a = len(expected_tools)
    len_p = len(predicted_tools)
    if len_a == 0 and len_p == 0:
        return 1.0
    max_len = max(len_a, len_p)
    lev = compute_levenshtein(expected_tools, predicted_tools)
    return max(0.0, 1.0 - (lev / max_len))

def compute_composite(rpi: float, ajm: float, sf: float, w1: float = 0.4, w2: float = 0.4, w3: float = 0.2) -> float:
    """Composite Tool-Selection Score (Eq. 5 in Chouhan 2026): C = w1*Rpi + w2*Ajm + w3*Sf"""
    return (w1 * rpi) + (w2 * ajm) + (w3 * sf)

def classify_failures(expected_trace: List[Dict[str, Any]], predicted_trace: List[Dict[str, Any]]) -> List[str]:
    """
    Classify observed trace failures according to Chouhan's 3-axis taxonomy:
    Axis 1 (Routing): R1 (Over-call), R2 (Under-call), R3 (Misclassification)
    Axis 2 (Parameters): P1 (Type mismatch), P2 (Omission), P3 (Hallucination)
    Axis 3 (Sequence): S1 (Dependency break)
    """
    failures = []
    exp_tools = [x["tool"] for x in expected_trace]
    pred_tools = [x["tool"] for x in predicted_trace]
    
    # Axis 1
    if len(exp_tools) == 0 and len(pred_tools) > 0:
        failures.append("R1") # Over-calling
    elif len(exp_tools) > 0 and len(pred_tools) == 0:
        failures.append("R2") # Under-calling
    elif set(exp_tools) != set(pred_tools):
        failures.append("R3") # Misclassification
        
    # Axis 2 & 3 per matched tool
    for i in range(min(len(expected_trace), len(predicted_trace))):
        exp_call = expected_trace[i]
        pred_call = predicted_trace[i]
        if exp_call["tool"] == pred_call["tool"]:
            ka = set(exp_call.get("arguments", {}).keys())
            kp = set(pred_call.get("arguments", {}).keys())
            # Omission
            if not ka.issubset(kp):
                failures.append("P2")
            # Hallucination or Type mismatch
            for k in ka.intersection(kp):
                v_act = exp_call["arguments"][k]
                v_pred = pred_call["arguments"][k]
                if type(v_act) != type(v_pred):
                    failures.append("P1")
                elif str(v_act).strip().lower() != str(v_pred).strip().lower():
                    failures.append("P3")
        else:
            failures.append("S1") # Dependency / step mismatch
            
    return list(set(failures))
