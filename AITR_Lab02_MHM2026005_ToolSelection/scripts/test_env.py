"""
Environment and System Diagnostic Sanity Test (Prompt 08)
Verifies Python version, runtime environment, file system accessibility,
and deterministic random seed control without external dependencies.
"""
import sys
import os
import json
import random
import platform

def run_environment_diagnostics():
    diagnostics = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "seed_check": None,
        "json_support": False,
        "status": "PASS"
    }
    
    # Check random seed repeatability
    random.seed(42)
    val1 = [random.random() for _ in range(5)]
    random.seed(42)
    val2 = [random.random() for _ in range(5)]
    if val1 == val2:
        diagnostics["seed_check"] = "DETERMINISTIC_VERIFIED"
    else:
        diagnostics["seed_check"] = "NON_DETERMINISTIC_FAILED"
        diagnostics["status"] = "FAIL"
        
    # Check JSON parsing
    test_dict = {"tool": "get_user_balance", "args": {"user_id": 1234}}
    parsed = json.loads(json.dumps(test_dict))
    if parsed == test_dict:
        diagnostics["json_support"] = True
    else:
        diagnostics["status"] = "FAIL"

    print("=== ENVIRONMENT DIAGNOSTIC REPORT ===")
    print(f"Status: {diagnostics['status']}")
    print(f"Python: {diagnostics['python_version'].split()[0]}")
    print(f"Platform: {diagnostics['platform']}")
    print(f"Random Seed Reproducibility: {diagnostics['seed_check']}")
    print(f"JSON Validation: {'OK' if diagnostics['json_support'] else 'ERROR'}")
    print("=====================================")
    return diagnostics["status"] == "PASS"

if __name__ == "__main__":
    success = run_environment_diagnostics()
    sys.exit(0 if success else 1)
