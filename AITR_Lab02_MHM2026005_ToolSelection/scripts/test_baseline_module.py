"""
Unit and Sanity Test for Metric Framework (Section 5.7 Worked Example Verification)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics import compute_rpi, compute_ajm, compute_sf, compute_composite, classify_failures

def test_chouhan_worked_example():
    # Worked example from Section 5.7
    ground_truth_tools = ["get_user_id", "get_balance", "send_report"]
    predicted_tools = ["get_user_id", "get_balance", "send_email"]
    
    # 1. Routing Precision Index
    rpi = compute_rpi(ground_truth_tools, predicted_tools)
    assert abs(rpi - 0.50) < 1e-4, f"Expected Rpi=0.50, got {rpi}"
    
    # 2. Argument Jaccard Match for get_balance
    expected_args = {"user_id": "U123", "currency": "USD"}
    predicted_args = {"user_id": "U123", "currency": "EUR_HALLUCINATED", "region": "US-WEST"}
    ajm = compute_ajm(expected_args, predicted_args)
    assert abs(ajm - (1.0 / 3.0)) < 1e-4, f"Expected Ajm=0.3333, got {ajm}"
    
    # 3. Sequence Fidelity
    sf = compute_sf(ground_truth_tools, predicted_tools)
    assert abs(sf - (2.0 / 3.0)) < 1e-4, f"Expected Sf=0.6667, got {sf}"
    
    # 4. Composite Score with weights (0.4, 0.4, 0.2)
    composite = compute_composite(rpi, ajm, sf, 0.4, 0.4, 0.2)
    assert abs(composite - 0.4666666) < 1e-3, f"Expected Composite ~0.47, got {composite}"
    
    # 5. Diagnostic failure classification
    exp_trace = [
        {"tool": "get_user_id", "arguments": {"username": "alice"}},
        {"tool": "get_balance", "arguments": expected_args},
        {"tool": "send_report", "arguments": {"format": "pdf"}}
    ]
    pred_trace = [
        {"tool": "get_user_id", "arguments": {"username": "alice"}},
        {"tool": "get_balance", "arguments": predicted_args},
        {"tool": "send_email", "arguments": {"format": "pdf"}}
    ]
    failures = classify_failures(exp_trace, pred_trace)
    assert "R3" in failures, "Expected R3 (misclassification)"
    assert "P3" in failures, "Expected P3 (entity hallucination)"
    assert "S1" in failures, "Expected S1 (sequence/step mismatch)"
    
    print("=== WORKED EXAMPLE UNIT TEST RESULTS ===")
    print(f"Rpi (Expected 0.50): {rpi:.4f} [PASS]")
    print(f"Ajm (Expected 0.3333): {ajm:.4f} [PASS]")
    print(f"Sf  (Expected 0.6667): {sf:.4f} [PASS]")
    print(f"Composite C (Expected ~0.47): {composite:.4f} [PASS]")
    print(f"Diagnosed Failures: {sorted(failures)} [PASS]")
    print("=========================================")
    return True

if __name__ == "__main__":
    test_chouhan_worked_example()
