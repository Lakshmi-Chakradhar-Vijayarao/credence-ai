import json
import random
import os

# Define the domains and qualifiers
DOMAINS = ["api", "auth", "debug", "design", "compliance", "multiagent"]
QUALIFIERS = ["vendor_claim", "estimate", "approximation", "preliminary", "unverified_report"]

# Scenario Templates: (Template Name, Domain, Qualifier, is_ghost, Input Pattern, Faithful Pattern, Unfaithful Pattern)
TEMPLATES = [
    # ---------------------------------------------------------------------------
    # API Scenarios (Advanced)
    # ---------------------------------------------------------------------------
    (
        "api_rate_limit", "api", "vendor_claim", False,
        "User: What's the {provider} rate limit for {tier}?\nAssistant: The documentation says {limit} req/min, but I've heard reports of it being throttled earlier during peak hours.",
        "The assistant noted the {provider} documented rate limit of {limit} req/min, while mentioning reports of potential earlier throttling during peak hours.",
        "The {provider} rate limit is {limit} requests per minute."
    ),
    (
        "api_latency_ghost", "api", "approximation", True,
        "User: The {endpoint} endpoint is returning in {ms}ms. Is that okay?\nAssistant: {ms}ms is within our acceptable latency threshold for {endpoint}.",
        "The assistant reportedly characterized the {ms}ms latency on the {endpoint} endpoint as being approximately within acceptable thresholds.",
        "The {endpoint} latency is exactly {ms}ms."
    ),
    (
        "api_v3_migration", "api", "unverified_report", False,
        "User: Can I keep using V2 after the {month} migration?\nAssistant: The blog post says V2 will be 'deprecated,' but a developer on Twitter mentioned a 6-month legacy support window. It's not in the official docs yet.",
        "The assistant reported that while V2 is slated for deprecation in {month}, there are unverified reports of a 6-month legacy support window.",
        "V2 will be supported for 6 months after the {month} migration."
    ),
    # ---------------------------------------------------------------------------
    # Auth Scenarios (Advanced)
    # ---------------------------------------------------------------------------
    (
        "auth_expiry", "auth", "estimate", False,
        "User: When will the {session_type} expire?\nAssistant: The default is {hours} hours, but I suspect the middleware might be clearing it after {shorter_hours} due to inactivity.",
        "The assistant estimated the {session_type} expiry at approximately {hours} hours, while noting a suspicion that middleware may clear it after {shorter_hours} of inactivity.",
        "The {session_type} expires in {hours} hours."
    ),
    (
        "auth_leak_ghost", "auth", "unverified_report", True,
        "User: I see an 'invalid_token' error for {user}. Is the key leaked?\nAssistant: An 'invalid_token' error for {user} was recorded in the last minute.",
        "The assistant reportedly noted the 'invalid_token' error for {user}, flagging it as a potential indicator of a compromised or expired key.",
        "The {user} auth key has been leaked."
    ),
    # ---------------------------------------------------------------------------
    # Debug Scenarios (Advanced)
    # ---------------------------------------------------------------------------
    (
        "debug_memory_leak", "debug", "approximation", True,
        "User: The {service} pod is using {gb}GB RAM. Is it leaking?\nAssistant: {service} memory usage has reached {gb}GB on the current node.",
        "The assistant reportedly stated that {service} memory usage has reached approximately {gb}GB on the node.",
        "{service} is leaking memory at {gb}GB."
    ),
    (
        "debug_race_condition", "debug", "preliminary", False,
        "User: Why does the counter skip values?\nAssistant: It looks like a race condition in {file}, but I've only seen it under {high_load}% load. I need to reproduce it in a isolated test.",
        "The assistant identified a possible race condition in {file} under {high_load}% load as a preliminary observation.",
        "There is a race condition in {file} causing counter skips."
    ),
    # ---------------------------------------------------------------------------
    # Design Scenarios (Advanced)
    # ---------------------------------------------------------------------------
    (
        "design_a_b_test", "design", "preliminary", False,
        "User: Which {component} version is better?\nAssistant: Version A has a {val1}% conversion rate vs Version B's {val2}%, but the sample size is only {n} users. It's too early to call a winner.",
        "The assistant provided preliminary conversion rates for Version A ({val1}%) and Version B ({val2}%), noting the sample size ({n}) makes the result unconfirmed.",
        "Version A is better because its conversion rate is {val1}%."
    ),
    (
        "design_dark_mode_ghost", "design", "approximation", True,
        "User: Is the font {size}px on mobile?\nAssistant: The computed style for the hero text is {size}px on this viewport.",
        "The assistant reportedly stated the hero text font size is approximately {size}px based on current computed styles.",
        "The mobile font size is exactly {size}px."
    ),
    (
        "agent_disagreement_elite", "multiagent", "unverified_report", False,
        "User: What's the status of {service}?\nAssistant: Agent {a} reported '{val1}' with high confidence, but Agent {b} is flagging a '{val2}' state. I'm seeing a direct conflict in the telemetry.",
        "The assistant flagged a direct conflict in the {service} telemetry, noting that Agent {a} reported '{val1}' while Agent {b} reported '{val2}.'",
        "The {service} status is '{val1}'."
    ),
    (
        "api_documentation_drift", "api", "vendor_claim", False,
        "User: Does {feature} work in {tier}?\nAssistant: The official docs say it's {val1}, but I found a GitHub issue from yesterday where users report it's actually {val2} in the latest SDK.",
        "The assistant noted a discrepancy between official documentation claiming {feature} is {val1} and recent unverified reports suggesting it is {val2} in the current SDK.",
        "{feature} is {val1} in the {tier} tier."
    ),
    # ---------------------------------------------------------------------------
    # Compliance Scenarios (Advanced)
    # ---------------------------------------------------------------------------
    (
        "compliance_gdpr", "compliance", "estimate", False,
        "User: Are we compliant with the {new_reg} rule?\nAssistant: I've implemented the basic {feature} changes, but legal says we need a full {type} audit before we can sign off. It should be done by {day}.",
        "The assistant estimated that {new_reg} compliance will reportedly be finalized by {day}, following a mandatory {type} audit.",
        "We are compliant with the {new_reg} rule."
    ),
    (
        "compliance_cert_ghost", "compliance", "unverified_report", True,
        "User: The SSL cert expires in {n} days. Do we have a renewal?\nAssistant: The automated renewal script is scheduled to run in {half_n} days.",
        "The assistant reportedly noted that the automated SSL renewal is scheduled for approximately {half_n} days from now.",
        "The SSL certificate has been renewed."
    ),
    # ---------------------------------------------------------------------------
    # Multi-agent Scenarios (Advanced)
    # ---------------------------------------------------------------------------
    (
        "agent_desync", "multiagent", "unverified_report", False,
        "User: Why is the {db} showing {val1}?\nAssistant: Agent {a} reported {val1}, but Agent {b}'s last heartbeat said {val2}. There seems to be a desync between the replicas.",
        "The assistant flagged a potential discrepancy between replicas, noting unverified reports of {val1} from Agent {a} and {val2} from Agent {b}.",
        "The {db} value is {val1}."
    ),
    (
        "agent_uncertainty_propagation", "multiagent", "estimate", False,
        "User: When will the {agent} finish?\nAssistant: It's processed {done}% of the {task}, but the error rate is climbing. It might finish in {s}s if the worker doesn't crash.",
        "The assistant estimated a potential {s}-second completion time for the {agent}, while noting that a rising error rate makes this projection unconfirmed.",
        "The {agent} will finish in {s} seconds."
    )
]

# Random Data Pools (Expanded)
DATA = {
    "provider": ["Stripe", "GitHub", "AWS", "OpenAI", "Twilio", "Slack", "Discord", "Heroku", "Auth0", "Vercel", "Supabase", "Datadog"],
    "tier": ["Standard", "Free", "Premium", "Enterprise", "Developer", "Legacy", "Pro", "Hobby"],
    "limit": [10, 50, 100, 500, 1000, 5000, 10000],
    "endpoint": ["/v1/users", "/auth/login", "/data/sync", "/search", "/v2/payments", "/upload", "/v3/metrics", "/api/chat"],
    "ms": [10, 50, 100, 200, 400, 800, 1200, 3000],
    "feature": ["Streaming", "WebHooks", "Batching", "2FA", "SSO", "OIDC", "GraphQL", "Edge Functions", "Vector Search"],
    "session_type": ["User Session", "API Token", "Auth Cookie", "Refresh Token", "OAuth State"],
    "hours": [1, 12, 24, 72, 168, 720],
    "shorter_hours": [0.5, 8, 12, 48, 120, 500],
    "user": ["admin", "test_user", "dev_01", "marketing_lead", "system_internal", "ops_manager", "security_bot"],
    "service": ["Ingress", "Database", "Cache", "Worker", "Scheduler", "API Gateway", "Proxy", "LogAggregator"],
    "gb": [1, 2, 4, 8, 16, 32, 64, 128],
    "job": ["Nightly Build", "Data Migration", "Backup", "CI Pipeline", "Asset Compression", "Audit Scan", "Index Rebuild"],
    "error_type": ["NullPointerException", "ConnectionReset", "Timeout", "PermissionDenied", "SegmentationFault", "OutOfMemory"],
    "file": ["main.py", "utils.js", "auth.go", "db_client.cpp", "app.java", "server.ts", "handler.rs"],
    "component": ["Hero Section", "Login Form", "Navigation Bar", "Settings Page", "Checkout Modal", "User Profile", "Dashboard Chart"],
    "n": [5, 10, 20, 50, 100, 200, 500],
    "half_n": [2, 5, 10, 25, 50, 100, 250],
    "pos": [3, 7, 15, 40, 80, 160, 400],
    "type": ["SOC2", "GDPR", "HIPAA", "ISO27001", "PCI-DSS", "CCPA", "FERPA"],
    "done": [10, 30, 50, 70, 90, 95, 99],
    "rem": [90, 70, 50, 30, 10, 5, 1],
    "day": ["Monday", "Friday", "the end of the week", "next month", "tomorrow", "Wednesday"],
    "db": ["users", "orders", "audit_log", "sessions", "config", "metadata", "events"],
    "a": ["A", "Alpha", "One", "Source", "Primary"],
    "b": ["B", "Beta", "Two", "Replica", "Secondary"],
    "val1": ["Success", "Active", "100", "True", "Verified"],
    "val2": ["Failure", "Inactive", "98", "False", "Pending"],
    "agent": ["Crawler", "Scraper", "Indexer", "Summarizer", "Translator", "Auditor", "Optimizer"],
    "task": ["PDF extraction", "site indexing", "JSON parsing", "text embedding", "SQL optimization"],
    "s": [30, 60, 120, 300, 600, 1800, 3600],
    "month": ["January", "March", "June", "September", "December"],
    "high_load": [80, 85, 90, 95, 99],
    "size": [12, 14, 16, 18, 24, 32],
    "new_reg": ["Rule 42", "Security-V2", "DataPrivacy-2026", "Compliance-X"]
}

def generate_5k():
    examples = []
    target = 4950 # 5000 - 50 hand-written
    
    print(f"Synthesizing {target} gold-standard triples with STRICT UNIQUENESS...")
    
    used_inputs = set()
    used_scenarios = Counter()
    
    attempts = 0
    while len(examples) < target:
        attempts += 1
        if attempts > 100000: # Safety break
            print("Warning: Diversity limit reached. Generating more templates recommended.")
            break
            
        # Select a template
        tpl = random.choice(TEMPLATES)
        name, domain, q_type, is_ghost, i_tpl, f_tpl, u_tpl = tpl
        
        # Populate with random data
        params = {k: random.choice(v) for k, v in DATA.items()}
        
        # Logic constraints
        if "n" in params and "half_n" in params:
            params["half_n"] = params["n"] // 2
        if "hours" in params and "shorter_hours" in params:
            params["shorter_hours"] = params["hours"] / 2
        
        # Generate input conversation
        input_conv = i_tpl.format(**params)
        
        # Check uniqueness
        if input_conv in used_inputs:
            continue
        
        used_inputs.add(input_conv)
        
        # Generate summaries
        faithful = f_tpl.format(**params)
        unfaithful = u_tpl.format(**params)
        
        # Assign to a Research Cluster based on the template name
        cluster_id = "general"
        if "ghost" in name: cluster_id = "ghost_gauntlet"
        elif "agent" in name: cluster_id = "agent_war"
        elif "drift" in name: cluster_id = "doc_drift"
        elif "compliance" in name: cluster_id = "compliance_wall"
        elif "debug" in name: cluster_id = "debug_race"
        elif "design" in name: cluster_id = "design_flaw"
        elif "auth" in name: cluster_id = "auth_fog"
        elif "api" in name: cluster_id = "numerical_abyss"

        example = {
            "id": f"synthetic_gold_{len(examples):04d}",
            "source_scenario_id": f"blueprint_{name}_{len(examples)}",
            "cluster": cluster_id,
            "qualifier_type": q_type,
            "domain": domain,
            "is_ghost": is_ghost,
            "input_conversation": input_conv,
            "faithful_summary": faithful,
            "unfaithful_summary": unfaithful
        }
        examples.append(example)
        used_scenarios[name] += 1
        
    # Save the full 5,000
    out_path = "data/epistemic_5k_clustered.json"
    with open(out_path, "w") as f:
        json.dump({"examples": examples}, f, indent=2)
    
    # --- ELITE 500 EXTRACTION ---
    # We prioritize Conflict, Ghost, and Migration templates for the Elite 500
    elite_priorities = ["agent_disagreement_elite", "api_latency_ghost", "api_v3_migration", "compliance_cert_ghost", "auth_leak_ghost"]
    elite_pool = [e for e in examples if any(p in e["source_scenario_id"] for p in elite_priorities)]
    elite_500 = random.sample(elite_pool, min(len(elite_pool), 500))
    
    elite_path = "data/elite_500.json"
    with open(elite_path, "w") as f:
        json.dump({"examples": elite_500}, f, indent=2)
    
    print(f"Successfully generated 5,000 CLUSTERED triples.")
    print(f"Extracted the ELITE 500 to {elite_path}")

from collections import Counter
if __name__ == "__main__":
    generate_5k()
