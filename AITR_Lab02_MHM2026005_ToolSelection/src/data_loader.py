"""
Data Loader Module: Manages Tool Catalog and Benchmark Test Traces.
"""
import os
import json
from typing import List, Dict, Any

class DataLoader:
    def __init__(self, project_root: str = None):
        if project_root is None:
            self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        else:
            self.project_root = project_root
            
        self.catalog_path = os.path.join(self.project_root, "configs", "tool_catalog.json")
        self.traces_path = os.path.join(self.project_root, "configs", "test_traces.json")
        
    def load_catalog(self) -> List[Dict[str, Any]]:
        with open(self.catalog_path, "r") as f:
            data = json.load(f)
        return data["tools"]
        
    def load_traces(self) -> List[Dict[str, Any]]:
        with open(self.traces_path, "r") as f:
            data = json.load(f)
        return data["episodes"]
        
    def get_traces_by_category(self, category: str) -> List[Dict[str, Any]]:
        all_traces = self.load_traces()
        return [t for t in all_traces if t.get("category") == category]
