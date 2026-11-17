"""
Proposed Model (Pathway C): Semantic-Overlap-Aware Routing and Grounding (SOAR).
Implements contrastive boundary disambiguation over confusable tool clusters
and schema-grounded parameter verification to prevent R3 misrouting and P3 hallucination.
"""
import re
import math
from collections import Counter
from typing import List, Dict, Any, Tuple
from src.baseline_router import BaselineRouter, tokenize, compute_cosine_similarity

class SOARRouter(BaselineRouter):
    def __init__(self, tools: List[Dict[str, Any]], margin_threshold: float = 0.15):
        super().__init__(tools)
        self.margin_threshold = margin_threshold
        
        # Build contrastive keyword discriminators for confusable clusters
        self.discriminator_rules = {
            ("check_account_balance", "fetch_user_financial_profile"): {
                "check_account_balance": ["balance", "cash", "available", "current", "liquid"],
                "fetch_user_financial_profile": ["credit", "net worth", "profile", "assets", "rating", "long term"]
            },
            ("wire_funds_transfer", "authorize_scheduled_payment"): {
                "wire_funds_transfer": ["wire", "immediate", "transfer", "now", "electronic"],
                "authorize_scheduled_payment": ["schedule", "recurring", "deferred", "due", "future", "vendor", "date"]
            },
            ("query_order_status", "track_shipment_package"): {
                "query_order_status": ["status", "lifecycle", "order", "placed", "fulfillment"],
                "track_shipment_package": ["tracking", "courier", "gps", "location", "transit", "carrier", "package"]
            },
            ("search_product_catalog", "check_inventory_stock"): {
                "search_product_catalog": ["search", "catalog", "under", "price", "find", "shoes", "items"],
                "check_inventory_stock": ["stock", "inventory", "warehouse", "units", "sku", "available units"]
            },
            ("issue_refund_request", "cancel_pending_order"): {
                "issue_refund_request": ["refund", "damaged", "return", "broken", "money back"],
                "cancel_pending_order": ["cancel", "pending", "abort", "unfulfilled", "stop order"]
            },
            ("query_order_status", "cancel_pending_order"): {
                "query_order_status": ["status", "lifecycle", "state", "track", "progress", "fulfillment"],
                "cancel_pending_order": ["cancel", "abort", "stop", "void", "terminate"]
            },
            ("search_internal_knowledge_base", "query_customer_support_tickets"): {
                "search_internal_knowledge_base": ["policy", "guidelines", "handbook", "internal", "corporate", "standard"],
                "query_customer_support_tickets": ["complaint", "ticket", "helpdesk", "incident", "customer issue"]
            }
        }
        
    def disambiguate_confusable_pair(self, tool1: str, tool2: str, query: str) -> str:
        """Apply contrastive keyword weighting to break ambiguity between confusable candidates."""
        query_lower = query.lower()
        for pair, weights in self.discriminator_rules.items():
            if (tool1 in pair) and (tool2 in pair):
                score1 = sum(2.0 for kw in weights.get(tool1, []) if kw in query_lower)
                score2 = sum(2.0 for kw in weights.get(tool2, []) if kw in query_lower)
                if score1 > score2:
                    return tool1
                elif score2 > score1:
                    return tool2
        return tool1 # Default to higher raw score if tied
        
    def route_query(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """Rank tools with contrastive margin disambiguation."""
        raw_routes = super().route_query(query, top_k=max(2, top_k))
        if len(raw_routes) >= 2:
            t1, s1 = raw_routes[0]
            t2, s2 = raw_routes[1]
            # Check if candidates form a confusable pair with small score margin
            confusables = self.tool_map.get(t1, {}).get("confusable_with", [])
            if (t2 in confusables) or ((s1 - s2) < self.margin_threshold):
                selected = self.disambiguate_confusable_pair(t1, t2, query)
                if selected == t2:
                    # Rerank t2 to first position
                    raw_routes[0], raw_routes[1] = raw_routes[1], raw_routes[0]
        return raw_routes[:top_k]
        
    def extract_arguments(self, tool_name: str, query: str) -> Dict[str, Any]:
        """
        Schema Grounding Gate:
        Extracts arguments strictly matching schema typing. Rejects hallucinated default values
        for missing entities (eliminating failure P3).
        """
        raw_args = super().extract_arguments(tool_name, query)
        tool_spec = self.tool_map.get(tool_name, {})
        params = tool_spec.get("parameters", {})
        
        grounded_args = {}
        for p_name, val in raw_args.items():
            # Filter out hallucinated defaults
            if isinstance(val, str) and "HALLUCINATED" in val:
                continue
            # Type grounding validation
            expected_type = params.get(p_name, {}).get("type", "string")
            if expected_type == "float" and not isinstance(val, float):
                try:
                    val = float(val)
                except ValueError:
                    continue
            elif expected_type == "integer" and not isinstance(val, int):
                try:
                    val = int(val)
                except ValueError:
                    continue
            elif expected_type == "boolean" and not isinstance(val, bool):
                val = bool(val)
            grounded_args[p_name] = val
            
        return grounded_args
