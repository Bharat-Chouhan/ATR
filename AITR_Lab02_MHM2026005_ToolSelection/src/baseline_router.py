"""
Baseline Router: Top-k Cosine Similarity Tool Router with Naive Parameter Extraction.
Represents standard dense/sparse retrieval-then-generate pipeline (Patil et al. 2023, Qin et al. 2024).
"""
import re
import math
from collections import Counter
from typing import List, Dict, Any, Tuple

def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text string."""
    return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())

def compute_cosine_similarity(vec1: Counter, vec2: Counter) -> float:
    """Compute cosine similarity between two term-frequency frequency vectors."""
    intersection = set(vec1.keys()).intersection(set(vec2.keys()))
    dot_product = sum(vec1[x] * vec2[x] for x in intersection)
    
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

class BaselineRouter:
    def __init__(self, tools: List[Dict[str, Any]]):
        self.tools = tools
        self.tool_map = {t["name"]: t for t in tools}
        self.tool_vectors = {}
        for t in tools:
            # Document representation: split tool name + description + parameter names
            name_tokens = t['name'].replace('_', ' ')
            param_tokens = ' '.join(k.replace('_', ' ') for k in t.get('parameters', {}).keys())
            doc_text = f"{t['name']} {name_tokens} {t['description']} {param_tokens}"
            self.tool_vectors[t["name"]] = Counter(tokenize(doc_text))
            
    def route_query(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """Rank tools by cosine similarity to query."""
        query_vec = Counter(tokenize(query))
        scores = []
        for name, tool_vec in self.tool_vectors.items():
            sim = compute_cosine_similarity(query_vec, tool_vec)
            scores.append((name, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
        
    def extract_arguments(self, tool_name: str, query: str) -> Dict[str, Any]:
        """
        Naive argument extractor:
        Extracts tokens following key patterns; if not found, may emit hallucinated defaults
        or omit parameters.
        """
        tool_spec = self.tool_map.get(tool_name, {})
        params = tool_spec.get("parameters", {})
        extracted = {}
        tokens = tokenize(query)
        
        for p_name, p_spec in params.items():
            # Check for patterns like ACC_123, ORD_456, USD, etc.
            if p_name in ["account_id", "source_account"]:
                match = re.search(r'\b(acc_\d+)\b', query, re.IGNORECASE)
                if match:
                    extracted[p_name] = match.group(1).upper()
            elif p_name in ["destination_account"]:
                matches = re.findall(r'\b(acc_\d+)\b', query, re.IGNORECASE)
                if len(matches) > 1:
                    extracted[p_name] = matches[1].upper()
            elif p_name in ["order_id"]:
                match = re.search(r'\b(ord_\d+)\b', query, re.IGNORECASE)
                if match:
                    extracted[p_name] = match.group(1).upper()
            elif p_name in ["tracking_number"]:
                match = re.search(r'\b(trk_\d+)\b', query, re.IGNORECASE)
                if match:
                    extracted[p_name] = match.group(1).upper()
            elif p_name in ["user_id"]:
                match = re.search(r'\b(usr_\d+)\b', query, re.IGNORECASE)
                if match:
                    extracted[p_name] = match.group(1).upper()
            elif p_name in ["currency"]:
                match = re.search(r'\b(usd|eur|gbp|jpy)\b', query, re.IGNORECASE)
                if match:
                    extracted[p_name] = match.group(1).upper()
                elif p_spec.get("required", False):
                    # Naive model hallucinates default entity when required
                    extracted[p_name] = "USD_DEFAULT_HALLUCINATED"
            elif p_name in ["amount"]:
                match = re.search(r'\b(\d+(\.\d+)?)\b', query)
                if match:
                    extracted[p_name] = float(match.group(1))
            elif p_name in ["max_price"]:
                match = re.search(r'(?:under|\$)\s*(\d+(?:\.\d+)?)', query, re.IGNORECASE)
                if not match:
                    match = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:dollars|\$)?', query, re.IGNORECASE)
                if match:
                    extracted[p_name] = float(match.group(1))
            elif p_name in ["city"]:
                match = re.search(r'\bin\s+([a-zA-Z]+)\b', query, re.IGNORECASE)
                if match:
                    extracted[p_name] = match.group(1).capitalize()
            elif p_name in ["days"]:
                match = re.search(r'\b(\d+)\s+days\b', query, re.IGNORECASE)
                if match:
                    extracted[p_name] = int(match.group(1))
            elif p_name in ["instance_id"]:
                match = re.search(r'\b(inst_\d+|srv_\d+)\b', query, re.IGNORECASE)
                if match:
                    extracted[p_name] = match.group(1).upper()
            elif p_name in ["query"]:
                match = re.search(r'for\s+(.*?)(?:\s+under|\s+dollars|\s*$)', query, re.IGNORECASE)
                if match:
                    extracted[p_name] = match.group(1).strip()
            elif p_name in ["sku"]:
                match = re.search(r'\b(sku_\d+)\b', query, re.IGNORECASE)
                if match:
                    extracted[p_name] = match.group(1).upper()
                    
        return extracted

    def generate_trace(self, prompt: str, horizon_L: int = 1) -> List[Dict[str, Any]]:
        """Generate full execution trace for an episode."""
        subqueries = [s.strip() for s in re.split(r';\s*then|;|\bthen\b', prompt) if len(s.strip()) > 5]
        if not subqueries:
            subqueries = [prompt]
            
        trace = []
        for i in range(min(horizon_L, len(subqueries))):
            sq = subqueries[i]
            top_routes = self.route_query(sq, top_k=1)
            if top_routes:
                tool_name = top_routes[0][0]
                args = self.extract_arguments(tool_name, sq)
                trace.append({"tool": tool_name, "arguments": args})
                
        # If horizon requires more steps, fill remaining with secondary subqueries
        while len(trace) < horizon_L:
            idx = len(trace)
            sq = subqueries[idx % len(subqueries)]
            top_routes = self.route_query(sq, top_k=1)
            if top_routes:
                t_name = top_routes[0][0]
                args = self.extract_arguments(t_name, sq)
                trace.append({"tool": t_name, "arguments": args})
            else:
                break
                
        return trace
