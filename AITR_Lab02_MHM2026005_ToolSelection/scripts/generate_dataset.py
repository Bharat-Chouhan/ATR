"""
Realistic Dataset Generator: Synthesizes 500 deterministic multi-step test traces
with aligned entity groundings for Orthogonal, Confusable, and Horizon-Stress conditions.
"""
import json
import random
import os

def generate_benchmark_dataset():
    random.seed(42)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    catalog_path = os.path.join(project_root, "configs", "tool_catalog.json")
    traces_path = os.path.join(project_root, "configs", "test_traces.json")
    
    with open(catalog_path, "r") as f:
        catalog = json.load(f)["tools"]
        
    episodes = []
    
    # 1. 200 Clean / Orthogonal episodes (L in [1, 2])
    clean_templates = [
        ("Check the balance for account ACC_{id} in USD", 
         [("check_account_balance", {"account_id": "ACC_{id}", "currency": "USD"})]),
        ("What is the weather in Paris for 3 days?", 
         [("query_weather_forecast", {"city": "Paris", "days": 3})]),
        ("Check health metrics for instance INST_{id}", 
         [("get_server_health_metrics", {"instance_id": "INST_{id}"})]),
        ("Search inventory for running shoes under 100 dollars", 
         [("search_product_catalog", {"query": "running shoes", "max_price": 100.0})]),
        ("Check warehouse unit inventory for SKU_{id}", 
         [("check_inventory_stock", {"sku": "SKU_{id}"})]),
        ("Query order ORD_{id} status ; then send email to client@test.com", 
         [("query_order_status", {"order_id": "ORD_{id}"}),
          ("send_email_notification", {"recipient": "client@test.com", "subject": "Order Status", "body": "Order confirmed"})]),
        ("Execute Python script to plot revenue distribution", 
         [("run_python_code_interpreter", {"script": "import matplotlib.pyplot as plt\nplt.hist(rev)"})]),
        ("Check balance ACC_{id} ; then send slack alert to #ops", 
         [("check_account_balance", {"account_id": "ACC_{id}"}),
          ("send_slack_alert", {"channel": "#ops", "message": "Balance check complete"})])
    ]
    
    for i in range(200):
        idx = 1000 + i
        prompt_tmpl, action_tmpl = clean_templates[i % len(clean_templates)]
        prompt = prompt_tmpl.replace("{id}", str(idx))
        actions = []
        for tool_name, args in action_tmpl:
            resolved_args = {}
            for k, v in args.items():
                if isinstance(v, str) and "{id}" in v:
                    resolved_args[k] = v.replace("{id}", str(idx))
                else:
                    resolved_args[k] = v
            actions.append({"tool": tool_name, "arguments": resolved_args})
            
        episodes.append({
            "episode_id": i + 1,
            "category": "orthogonal",
            "prompt": prompt,
            "horizon_L": len(actions),
            "expected_trace": actions
        })
        
    # 2. 150 Confusable / Overlapping prompts designed to test R3 and P3 failure modes
    # In confusable cases, descriptions overlap heavily.
    # We test whether the agent picks the exact tool or the confusable distractor.
    confusable_templates = [
        # Overlapping / Ambiguous phrasing where naive TF-IDF router confuses candidates
        ("Check customer account balance and financial profile for ACC_{id}",
         [("check_account_balance", {"account_id": "ACC_{id}"})],
         "fetch_user_financial_profile"),
        ("Send electronic payment transfer to account ACC_{id}",
         [("wire_funds_transfer", {"source_account": "ACC_{id}", "destination_account": "ACC_{dest}", "amount": 250.0, "currency": "USD"})],
         "authorize_scheduled_payment"),
        ("Check shipping order delivery updates for package TRK_{id}",
         [("track_shipment_package", {"tracking_number": "TRK_{id}"})],
         "query_order_status"),
        ("Search customer problem resolution guidelines and corporate policy",
         [("search_internal_knowledge_base", {"query": "corporate policy"})],
         "query_customer_support_tickets"),
        # Explicit distinction phrasing
        ("Fetch the total long term financial profile credit rating and assets for user USR_{id}",
         [("fetch_user_financial_profile", {"user_id": "USR_{id}"})],
         "check_account_balance"),
        ("Query liquid available cash balance for bank account ACC_{id}",
         [("check_account_balance", {"account_id": "ACC_{id}"})],
         "fetch_user_financial_profile"),
        ("Schedule future recurring authorization payment of 120.0 to vendor VEN_{id} due on 2026-12-01",
         [("authorize_scheduled_payment", {"vendor_id": "VEN_{id}", "amount": 120.0, "due_date": "2026-12-01"})],
         "wire_funds_transfer"),
        ("Track real time courier GPS carrier transit milestones for package tracking TRK_{id}",
         [("track_shipment_package", {"tracking_number": "TRK_{id}"})],
         "query_order_status")
    ]
    
    for i in range(150):
        idx = 2000 + i
        dest_idx = 3000 + i
        prompt_tmpl, action_tmpl, distractor = confusable_templates[i % len(confusable_templates)]
        prompt = prompt_tmpl.replace("{id}", str(idx)).replace("{dest}", str(dest_idx))
        actions = []
        for tool_name, args in action_tmpl:
            resolved_args = {}
            for k, v in args.items():
                if isinstance(v, str):
                    resolved_args[k] = v.replace("{id}", str(idx)).replace("{dest}", str(dest_idx))
                else:
                    resolved_args[k] = v
            actions.append({"tool": tool_name, "arguments": resolved_args})
            
        episodes.append({
            "episode_id": 201 + i,
            "category": "confusable",
            "prompt": prompt,
            "distractor_tool": distractor,
            "horizon_L": len(actions),
            "expected_trace": actions
        })
        
    # 3. 150 Multi-Step Horizon Stress Test episodes (L in {1, 2, 3, 5, 7, 10})
    subtask_templates = [
        ("check account balance for ACC_{id}", ("check_account_balance", {"account_id": "ACC_{id}"})),
        ("query order status for ORD_{id}", ("query_order_status", {"order_id": "ORD_{id}"})),
        ("query weather in Tokyo for 3 days", ("query_weather_forecast", {"city": "Tokyo", "days": 3})),
        ("check health metrics for instance INST_{id}", ("get_server_health_metrics", {"instance_id": "INST_{id}"})),
        ("search product catalog for laptop under 800", ("search_product_catalog", {"query": "laptop", "max_price": 800.0})),
        ("check inventory stock for SKU_{id}", ("check_inventory_stock", {"sku": "SKU_{id}"})),
        ("send email to dev@corp.com", ("send_email_notification", {"recipient": "dev@corp.com", "subject": "Alert", "body": "Audit report"})),
        ("generate pdf invoice for ORD_{id}", ("generate_pdf_invoice", {"order_id": "ORD_{id}", "recipient_name": "Acme Corp"})),
        ("send slack alert to #general", ("send_slack_alert", {"channel": "#general", "message": "Pipeline completed"})),
        ("run python script to compute summary", ("run_python_code_interpreter", {"script": "print('ok')"}))
    ]
    
    horizon_lengths = [1, 2, 3, 5, 7, 10]
    for i in range(150):
        idx = 4000 + i
        target_L = horizon_lengths[i % len(horizon_lengths)]
        selected_st = [subtask_templates[j % len(subtask_templates)] for j in range(target_L)]
        
        prompt_parts = []
        actions = []
        for j, (st_text, (tool_name, base_args)) in enumerate(selected_st):
            p_text = st_text.replace("{id}", str(idx + j))
            prompt_parts.append(p_text)
            
            resolved_args = {}
            for k, v in base_args.items():
                if isinstance(v, str) and "{id}" in v:
                    resolved_args[k] = v.replace("{id}", str(idx + j))
                else:
                    resolved_args[k] = v
            actions.append({"tool": tool_name, "arguments": resolved_args})
            
        chained_prompt = " ; then ".join(prompt_parts)
        episodes.append({
            "episode_id": 351 + i,
            "category": "horizon_stress",
            "prompt": chained_prompt,
            "horizon_L": target_L,
            "expected_trace": actions
        })
        
    with open(traces_path, "w") as f:
        json.dump({"episodes": episodes}, f, indent=2)
        
    print(f"Generated {len(episodes)} test traces saved to {traces_path}")

if __name__ == "__main__":
    generate_benchmark_dataset()
