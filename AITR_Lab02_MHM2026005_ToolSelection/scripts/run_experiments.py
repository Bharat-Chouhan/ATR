"""
Master Experiment Runner (Phase IV: Contribution and Evidence)
Executes Baseline vs. SOAR across 500 evaluation episodes.
Exports results.csv, summary_metrics.json, compounding_horizon.csv, and SVG plots.
"""
import os
import sys
import json
import csv
import math
from typing import Dict, List, Any

# Ensure project modules are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import DataLoader
from src.baseline_router import BaselineRouter
from src.soar_router import SOARRouter
from src.metrics import (
    compute_rpi,
    compute_ajm,
    compute_sf,
    compute_composite,
    classify_failures
)

def evaluate_method(method_name: str, router, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for ep in episodes:
        prompt = ep["prompt"]
        horizon_L = ep["horizon_L"]
        expected_trace = ep["expected_trace"]
        
        # Generate predicted trace
        predicted_trace = router.generate_trace(prompt, horizon_L=horizon_L)
        
        exp_tools = [x["tool"] for x in expected_trace]
        pred_tools = [x["tool"] for x in predicted_trace]
        
        # 1. Routing Precision Index
        rpi = compute_rpi(exp_tools, pred_tools)
        
        # 2. Argument Jaccard Match (macro average over steps)
        ajm_scores = []
        for i in range(min(len(expected_trace), len(predicted_trace))):
            ajm_step = compute_ajm(expected_trace[i].get("arguments", {}), predicted_trace[i].get("arguments", {}))
            ajm_scores.append(ajm_step)
        ajm = sum(ajm_scores) / len(ajm_scores) if ajm_scores else 0.0
        
        # 3. Sequence Fidelity
        sf = compute_sf(exp_tools, pred_tools)
        
        # 4. Composite Score
        composite = compute_composite(rpi, ajm, sf, 0.4, 0.4, 0.2)
        
        # 5. Trace Success Rate (Binary: 1 if Rpi==1.0 and Ajm==1.0 and Sf==1.0)
        tsr = 1.0 if (rpi == 1.0 and ajm == 1.0 and sf == 1.0) else 0.0
        
        # 6. Diagnostic Failures
        failures = classify_failures(expected_trace, predicted_trace)
        
        records.append({
            "episode_id": ep["episode_id"],
            "category": ep["category"],
            "horizon_L": horizon_L,
            "method": method_name,
            "rpi": round(rpi, 4),
            "ajm": round(ajm, 4),
            "sf": round(sf, 4),
            "composite": round(composite, 4),
            "tsr": tsr,
            "failures": ";".join(sorted(failures)) if failures else "NONE"
        })
    return records

def calculate_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    
    def mean(key):
        return sum(r[key] for r in records) / n
        
    def std(key, m):
        return math.sqrt(sum((r[key] - m) ** 2 for r in records) / n)
        
    m_rpi = mean("rpi")
    m_ajm = mean("ajm")
    m_sf = mean("sf")
    m_comp = mean("composite")
    m_tsr = mean("tsr")
    
    # 95% Binomial confidence interval for TSR
    ci_tsr = 1.96 * math.sqrt((m_tsr * (1.0 - m_tsr)) / n) if n > 0 else 0.0
    
    # Failure counts
    all_failures = []
    for r in records:
        if r["failures"] != "NONE":
            all_failures.extend(r["failures"].split(";"))
    f_counts = {}
    for f in all_failures:
        f_counts[f] = f_counts.get(f, 0) + 1
        
    return {
        "n": n,
        "mean_rpi": round(m_rpi, 4),
        "std_rpi": round(std("rpi", m_rpi), 4),
        "mean_ajm": round(m_ajm, 4),
        "std_ajm": round(std("ajm", m_ajm), 4),
        "mean_sf": round(m_sf, 4),
        "std_sf": round(std("sf", m_sf), 4),
        "mean_composite": round(m_comp, 4),
        "std_composite": round(std("composite", m_comp), 4),
        "tsr": round(m_tsr, 4),
        "ci_tsr": round(ci_tsr, 4),
        "failure_counts": f_counts
    }

def generate_svg_charts(summary: Dict[str, Any], compounding_data: List[Dict[str, Any]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # Chart 1: Overlap Degradation (Baseline vs SOAR across Orthogonal vs Confusable)
    svg1 = f"""<svg xmlns="http://www.w3.org/2000/svg" width="650" height="400" viewBox="0 0 650 400" style="background:#ffffff; font-family:sans-serif;">
        <text x="325" y="30" font-size="16" font-weight="bold" text-anchor="middle" fill="#1e293b">Routing Precision Index (Rpi) under Overlap Conditions</text>
        <!-- Axes -->
        <line x1="80" y1="320" x2="580" y2="320" stroke="#94a3b8" stroke-width="2"/>
        <line x1="80" y1="70" x2="80" y2="320" stroke="#94a3b8" stroke-width="2"/>
        <!-- Grid lines & Y ticks -->
        <text x="70" y="325" font-size="12" text-anchor="end" fill="#64748b">0.0</text>
        <line x1="80" y1="320" x2="580" y2="320" stroke="#e2e8f0" stroke-dasharray="4"/>
        <text x="70" y="260" font-size="12" text-anchor="end" fill="#64748b">0.25</text>
        <line x1="80" y1="257" x2="580" y2="257" stroke="#e2e8f0" stroke-dasharray="4"/>
        <text x="70" y="195" font-size="12" text-anchor="end" fill="#64748b">0.50</text>
        <line x1="80" y1="195" x2="580" y2="195" stroke="#e2e8f0" stroke-dasharray="4"/>
        <text x="70" y="132" font-size="12" text-anchor="end" fill="#64748b">0.75</text>
        <line x1="80" y1="132" x2="580" y2="132" stroke="#e2e8f0" stroke-dasharray="4"/>
        <text x="70" y="75" font-size="12" text-anchor="end" fill="#64748b">1.0</text>
        <line x1="80" y1="70" x2="580" y2="70" stroke="#e2e8f0" stroke-dasharray="4"/>

        <!-- Bars Category 1: Orthogonal -->
        <text x="210" y="345" font-size="13" font-weight="600" text-anchor="middle" fill="#334155">Orthogonal Catalogue</text>
        <rect x="140" y="{320 - int(summary['baseline']['categories']['orthogonal']['mean_rpi'] * 250)}" width="60" height="{int(summary['baseline']['categories']['orthogonal']['mean_rpi'] * 250)}" fill="#94a3b8" rx="4"/>
        <text x="170" y="{310 - int(summary['baseline']['categories']['orthogonal']['mean_rpi'] * 250)}" font-size="11" text-anchor="middle" fill="#0f172a">{summary['baseline']['categories']['orthogonal']['mean_rpi']}</text>

        <rect x="215" y="{320 - int(summary['soar']['categories']['orthogonal']['mean_rpi'] * 250)}" width="60" height="{int(summary['soar']['categories']['orthogonal']['mean_rpi'] * 250)}" fill="#0284c7" rx="4"/>
        <text x="245" y="{310 - int(summary['soar']['categories']['orthogonal']['mean_rpi'] * 250)}" font-size="11" text-anchor="middle" fill="#0f172a">{summary['soar']['categories']['orthogonal']['mean_rpi']}</text>

        <!-- Bars Category 2: Confusable -->
        <text x="450" y="345" font-size="13" font-weight="600" text-anchor="middle" fill="#334155">Confusable Overlap</text>
        <rect x="380" y="{320 - int(summary['baseline']['categories']['confusable']['mean_rpi'] * 250)}" width="60" height="{int(summary['baseline']['categories']['confusable']['mean_rpi'] * 250)}" fill="#f87171" rx="4"/>
        <text x="410" y="{310 - int(summary['baseline']['categories']['confusable']['mean_rpi'] * 250)}" font-size="11" text-anchor="middle" fill="#0f172a">{summary['baseline']['categories']['confusable']['mean_rpi']}</text>

        <rect x="455" y="{320 - int(summary['soar']['categories']['confusable']['mean_rpi'] * 250)}" width="60" height="{int(summary['soar']['categories']['confusable']['mean_rpi'] * 250)}" fill="#10b981" rx="4"/>
        <text x="485" y="{310 - int(summary['soar']['categories']['confusable']['mean_rpi'] * 250)}" font-size="11" text-anchor="middle" fill="#0f172a">{summary['soar']['categories']['confusable']['mean_rpi']}</text>

        <!-- Legend -->
        <rect x="220" y="375" width="14" height="14" fill="#94a3b8"/>
        <text x="240" y="387" font-size="12" fill="#334155">Baseline Router</text>
        <rect x="360" y="375" width="14" height="14" fill="#10b981"/>
        <text x="380" y="387" font-size="12" fill="#334155">Proposed SOAR Router</text>
    </svg>"""
    with open(os.path.join(output_dir, "figure2_overlap_degradation.svg"), "w") as f:
        f.write(svg1)
        
    # Chart 2: Compounding Horizon Decay (Empirical TSR vs Theoretical q^L)
    svg2 = f"""<svg xmlns="http://www.w3.org/2000/svg" width="650" height="420" viewBox="0 0 650 420" style="background:#ffffff; font-family:sans-serif;">
        <text x="325" y="30" font-size="16" font-weight="bold" text-anchor="middle" fill="#1e293b">Empirical TSR vs Theoretical Analytical Model P(success)=q^L</text>
        <line x1="80" y1="340" x2="580" y2="340" stroke="#94a3b8" stroke-width="2"/>
        <line x1="80" y1="60" x2="80" y2="340" stroke="#94a3b8" stroke-width="2"/>
        
        <text x="70" y="345" font-size="12" text-anchor="end" fill="#64748b">0.0</text>
        <text x="70" y="270" font-size="12" text-anchor="end" fill="#64748b">0.25</text>
        <text x="70" y="200" font-size="12" text-anchor="end" fill="#64748b">0.50</text>
        <text x="70" y="130" font-size="12" text-anchor="end" fill="#64748b">0.75</text>
        <text x="70" y="65" font-size="12" text-anchor="end" fill="#64748b">1.0</text>
        
        <text x="330" y="375" font-size="13" font-weight="bold" text-anchor="middle" fill="#1e293b">Trace Length L (Horizon Steps)</text>
    """
    
    # Points on X: L = 1 (120), 2 (200), 3 (280), 5 (380), 7 (470), 10 (550)
    x_map = {1: 120, 2: 200, 3: 280, 5: 380, 7: 470, 10: 550}
    
    # Draw theoretical polyline
    pts_theory = " ".join([f"{x_map[d['L']]},{340 - int(d['theoretical_ql'] * 280)}" for d in compounding_data])
    svg2 += f'<polyline fill="none" stroke="#6366f1" stroke-width="3" stroke-dasharray="6" points="{pts_theory}"/>\n'
    
    # Draw empirical baseline polyline
    pts_base = " ".join([f"{x_map[d['L']]},{340 - int(d['empirical_baseline_tsr'] * 280)}" for d in compounding_data])
    svg2 += f'<polyline fill="none" stroke="#ef4444" stroke-width="3" points="{pts_base}"/>\n'

    # Draw empirical SOAR polyline
    pts_soar = " ".join([f"{x_map[d['L']]},{340 - int(d['empirical_soar_tsr'] * 280)}" for d in compounding_data])
    svg2 += f'<polyline fill="none" stroke="#10b981" stroke-width="3" points="{pts_soar}"/>\n'
    
    for d in compounding_data:
        x = x_map[d['L']]
        svg2 += f'<text x="{x}" y="358" font-size="12" text-anchor="middle" fill="#334155">{d["L"]}</text>'
        # Dots
        y_th = 340 - int(d['theoretical_ql'] * 280)
        y_bs = 340 - int(d['empirical_baseline_tsr'] * 280)
        y_sr = 340 - int(d['empirical_soar_tsr'] * 280)
        svg2 += f'<circle cx="{x}" cy="{y_th}" r="4" fill="#6366f1"/>'
        svg2 += f'<circle cx="{x}" cy="{y_bs}" r="4" fill="#ef4444"/>'
        svg2 += f'<circle cx="{x}" cy="{y_sr}" r="4" fill="#10b981"/>'
        
    svg2 += """
        <rect x="130" y="395" width="12" height="12" fill="#6366f1"/>
        <text x="148" y="405" font-size="11" fill="#334155">Theoretical q^L (q=0.90)</text>
        <rect x="300" y="395" width="12" height="12" fill="#ef4444"/>
        <text x="318" y="405" font-size="11" fill="#334155">Empirical Baseline TSR</text>
        <rect x="470" y="395" width="12" height="12" fill="#10b981"/>
        <text x="488" y="405" font-size="11" fill="#334155">Empirical SOAR TSR</text>
    </svg>"""
    with open(os.path.join(output_dir, "figure3_compounding_horizon_decay.svg"), "w") as f:
        f.write(svg2)
        
    print(f"Generated publication SVG figures in {output_dir}")

def run_all_experiments():
    loader = DataLoader(PROJECT_ROOT)
    catalog = loader.load_catalog()
    episodes = loader.load_traces()
    
    baseline_router = BaselineRouter(catalog)
    soar_router = SOARRouter(catalog, margin_threshold=0.15)
    
    print(f"Running evaluation over {len(episodes)} episodes...")
    baseline_records = evaluate_method("Baseline_TopK", baseline_router, episodes)
    soar_records = evaluate_method("SOAR_Proposed", soar_router, episodes)
    
    all_records = baseline_records + soar_records
    
    # Save results.csv
    results_csv_path = os.path.join(PROJECT_ROOT, "results", "results.csv")
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    print(f"Saved {len(all_records)} trace rows to {results_csv_path}")
    
    # Aggregate statistics
    summary = {
        "baseline": {
            "overall": calculate_stats(baseline_records),
            "categories": {
                "orthogonal": calculate_stats([r for r in baseline_records if r["category"] == "orthogonal"]),
                "confusable": calculate_stats([r for r in baseline_records if r["category"] == "confusable"]),
                "horizon_stress": calculate_stats([r for r in baseline_records if r["category"] == "horizon_stress"])
            }
        },
        "soar": {
            "overall": calculate_stats(soar_records),
            "categories": {
                "orthogonal": calculate_stats([r for r in soar_records if r["category"] == "orthogonal"]),
                "confusable": calculate_stats([r for r in soar_records if r["category"] == "confusable"]),
                "horizon_stress": calculate_stats([r for r in soar_records if r["category"] == "horizon_stress"])
            }
        }
    }
    
    # Compounding horizon analysis
    compounding_data = []
    horizon_lengths = [1, 2, 3, 5, 7, 10]
    q_est = 0.90 # Single step accuracy estimate
    for L in horizon_lengths:
        b_l = [r for r in baseline_records if r["category"] == "horizon_stress" and r["horizon_L"] == L]
        s_l = [r for r in soar_records if r["category"] == "horizon_stress" and r["horizon_L"] == L]
        
        b_tsr = sum(r["tsr"] for r in b_l) / len(b_l) if b_l else 0.0
        s_tsr = sum(r["tsr"] for r in s_l) / len(s_l) if s_l else 0.0
        theory = round(q_est ** L, 4)
        
        compounding_data.append({
            "L": L,
            "q_per_step": q_est,
            "theoretical_ql": theory,
            "empirical_baseline_tsr": round(b_tsr, 4),
            "empirical_soar_tsr": round(s_tsr, 4)
        })
        
    comp_csv_path = os.path.join(PROJECT_ROOT, "results", "compounding_horizon.csv")
    with open(comp_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(compounding_data[0].keys()))
        writer.writeheader()
        writer.writerows(compounding_data)
        
    summary_json_path = os.path.join(PROJECT_ROOT, "results", "summary_metrics.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Saved summary metrics to {summary_json_path}")
    
    # Generate Figures
    fig_dir = os.path.join(PROJECT_ROOT, "results", "figures")
    generate_svg_charts(summary, compounding_data, fig_dir)
    
    # Print console summary
    print("\n" + "="*50)
    print("           EXPERIMENTAL RESULTS SUMMARY           ")
    print("="*50)
    print(f"Overall Baseline Composite C : {summary['baseline']['overall']['mean_composite']} ± {summary['baseline']['overall']['std_composite']}")
    print(f"Overall SOAR Composite C     : {summary['soar']['overall']['mean_composite']} ± {summary['soar']['overall']['std_composite']}")
    print(f"Confusable Rpi Baseline      : {summary['baseline']['categories']['confusable']['mean_rpi']}")
    print(f"Confusable Rpi SOAR          : {summary['soar']['categories']['confusable']['mean_rpi']}")
    print(f"Baseline Overall TSR         : {summary['baseline']['overall']['tsr']} (CI: ±{summary['baseline']['overall']['ci_tsr']})")
    print(f"SOAR Overall TSR             : {summary['soar']['overall']['tsr']} (CI: ±{summary['soar']['overall']['ci_tsr']})")
    print(f"Baseline Failures            : {summary['baseline']['overall']['failure_counts']}")
    print(f"SOAR Failures                : {summary['soar']['overall']['failure_counts']}")
    print("="*50)

if __name__ == "__main__":
    run_all_experiments()
