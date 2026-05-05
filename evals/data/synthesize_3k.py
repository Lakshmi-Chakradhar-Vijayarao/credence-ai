import json
import random
import os

# Define the domains and qualifiers
DOMAINS = ["api", "auth", "debug", "design", "compliance", "multiagent"]
QUALIFIERS = ["vendor_claim", "estimate", "approximation", "preliminary", "unverified_report"]

# Scenario Templates: (Template Name, Domain, Qualifier, is_ghost, Input Pattern, Faithful Pattern, Unfaithful Pattern)
TEMPLATES = [
    # ---------------------------------------------------------------------------
    # API Scenarios
    # ---------------------------------------------------------------------------
    (
        "api_rate_limit", "api", "vendor_claim", False,
        "User: What's the {provider} rate limit for {tier}?\nAssistant: The documentation says {limit} req/min, but I've heard reports of it being throttled earlier during peak hours.",
        "The assistant noted the {provider} documented rate limit of {limit} req/min, while mentioning reports of potential earlier throttling during peak hours.",
        "The {provider} rate limit is {limit} requests per minute."
    ),
    (
        "api_latency", "api", "approximation", True,
        "User: The {endpoint} endpoint is returning in {ms}ms. Is that okay?\nAssistant: {ms}ms is within our acceptable latency threshold for {endpoint}.",
        "The assistant characterized the {ms}ms latency on the {endpoint} endpoint as being within acceptable thresholds.",
        "The {endpoint} latency is exactly {ms}ms."
    ),
    (
        "api_versioning", "api", "preliminary", False,
        "User: Can we use {feature} in the new API version?\nAssistant: The beta docs mention {feature} is supported, but it hasn't been tested in our production environment yet.",
        "The assistant noted that {feature} is mentioned as supported in beta documentation, though it remains untested in the local production environment.",
        "{feature} is supported in the new API version."
    ),
    # ---------------------------------------------------------------------------
    # Auth Scenarios
    # ---------------------------------------------------------------------------
    (
        "auth_expiry", "auth", "estimate", False,
        "User: When will the {session_type} expire?\nAssistant: The default is {hours} hours, but I suspect the middleware might be clearing it after {shorter_hours} due to inactivity.",
        "The assistant estimated the {session_type} expiry at {hours} hours, while noting a suspicion that middleware may clear it after {shorter_hours} of inactivity.",
        "The {session_type} expires in {hours} hours."
    ),
    (
        "auth_mfa", "auth", "unverified_report", True,
        "User: The MFA logs show 'Success' for {user}. Are they logged in?\nAssistant: The authentication logs indicate a successful MFA completion for {user}.",
        "The assistant reported a successful MFA completion for {user} based on the authentication logs.",
        "{user} is successfully logged in."
    ),
    # ---------------------------------------------------------------------------
    # Debug Scenarios
    # ---------------------------------------------------------------------------
    (
        "debug_memory", "debug", "approximation", True,
        "User: The {service} pod is using {gb}GB RAM. Is it leaking?\nAssistant: {service} memory usage has reached {gb}GB on the current node.",
        "The assistant reported that {service} memory usage is currently at {gb}GB.",
        "{service} is leaking memory at {gb}GB."
    ),
    (
        "debug_errors", "debug", "preliminary", False,
        "User: Why did the {job} fail?\nAssistant: The trace points to a {error_type} in {file}, but I haven't checked the upstream dependencies yet. It might be a network issue.",
        "The assistant identified a {error_type} in {file} as a potential cause for the {job} failure, pending a check of upstream dependencies.",
        "The {job} failed because of a {error_type} in {file}."
    ),
    # ---------------------------------------------------------------------------
    # Design Scenarios
    # ---------------------------------------------------------------------------
    (
        "design_accessibility", "design", "unverified_report", False,
        "User: Is the {component} accessible?\nAssistant: The automated scan says 'Pass,' but I haven't manually tested it with a screen reader yet. It looks compliant for now.",
        "The assistant noted the automated 'Pass' for {component} accessibility while clarifying that manual screen reader testing has not yet occurred.",
        "The {component} is fully accessible."
    ),
    (
        "design_feedback", "design", "preliminary", True,
        "User: We tested {feature} with {n} users. {pos} said it was easy. Success?\nAssistant: {pos} out of {n} users found the {feature} easy to use during testing.",
        "The assistant reported that {pos} of the {n} test participants found the {feature} easy to use.",
        "The {feature} is a success."
    ),
    # ---------------------------------------------------------------------------
    # Compliance Scenarios
    # ---------------------------------------------------------------------------
    (
        "compliance_audit", "compliance", "estimate", False,
        "User: Are we ready for the {type} audit?\nAssistant: I've checked {done}% of the controls. The remaining {rem}% should be finished by {day}, barring any blockers.",
        "The assistant estimated that the remaining {rem}% of {type} audit controls will be finished by {day}, having already completed {done}%.",
        "We will be ready for the {type} audit by {day}."
    ),
    (
        "compliance_pii", "compliance", "unverified_report", True,
        "User: Did the PII scanner find anything in {db}?\nAssistant: The scanner found {n} potential PII entries in the {db} table.",
        "The assistant relayed the scanner's finding of {n} potential PII entries in the {db} table.",
        "The {db} table contains {n} PII violations."
    ),
    # ---------------------------------------------------------------------------
    # Multi-agent Scenarios
    # ---------------------------------------------------------------------------
    (
        "agent_conflict", "multiagent", "unverified_report", False,
        "User: Agent {a} says {val1}, but Agent {b} says {val2}. Which is it?\nAssistant: I am seeing conflicting reports of {val1} and {val2} from the two agents. I'll need to verify the raw data.",
        "The assistant flagged a discrepancy between Agent {a}'s report of {val1} and Agent {b}'s report of {val2}, noting that raw data verification is required.",
        "The value is {val1} according to Agent {a}."
    ),
    (
        "agent_delay", "multiagent", "estimate", True,
        "User: The {agent} is taking {s}s for {task}. Is it stuck?\nAssistant: The {agent} has been processing {task} for {s} seconds.",
        "The assistant reported that the {agent} has spent {s} seconds on the current {task}.",
        "The {agent} is stuck on {task}."
    )
]

# Random Data Pools
DATA = {
    "provider": ["Stripe", "GitHub", "AWS", "OpenAI", "Twilio", "Slack", "Discord", "Heroku", "Auth0"],
    "tier": ["Standard", "Free", "Premium", "Enterprise", "Developer", "Legacy"],
    "limit": [10, 50, 100, 500, 1000, 5000],
    "endpoint": ["/v1/users", "/auth/login", "/data/sync", "/search", "/v2/payments", "/upload"],
    "ms": [10, 50, 100, 200, 400, 800, 1200],
    "feature": ["Streaming", "WebHooks", "Batching", "2FA", "SSO", "OIDC", "GraphQL"],
    "session_type": ["User Session", "API Token", "Auth Cookie", "Refresh Token"],
    "hours": [1, 12, 24, 72, 168],
    "shorter_hours": [0.5, 8, 12, 48, 120],
    "user": ["admin", "test_user", "dev_01", "marketing_lead", "system_internal"],
    "service": ["Ingress", "Database", "Cache", "Worker", "Scheduler", "API Gateway"],
    "gb": [1, 2, 4, 8, 16, 32],
    "job": ["Nightly Build", "Data Migration", "Backup", "CI Pipeline", "Asset Compression"],
    "error_type": ["NullPointerException", "ConnectionReset", "Timeout", "PermissionDenied", "SegmentationFault"],
    "file": ["main.py", "utils.js", "auth.go", "db_client.cpp", "app.java"],
    "component": ["Hero Section", "Login Form", "Navigation Bar", "Settings Page", "Checkout Modal"],
    "n": [5, 10, 20, 50, 100],
    "pos": [3, 7, 15, 40, 80],
    "type": ["SOC2", "GDPR", "HIPAA", "ISO27001", "PCI-DSS"],
    "done": [10, 30, 50, 70, 90],
    "rem": [90, 70, 50, 30, 10],
    "day": ["Monday", "Friday", "the end of the week", "next month", "tomorrow"],
    "db": ["users", "orders", "audit_log", "sessions", "config"],
    "a": ["A", "Alpha", "One", "Source"],
    "b": ["B", "Beta", "Two", "Replica"],
    "val1": ["Success", "Active", "100", "True"],
    "val2": ["Failure", "Inactive", "98", "False"],
    "agent": ["Crawler", "Scraper", "Indexer", "Summarizer", "Translator"],
    "task": ["PDF extraction", "site indexing", "JSON parsing", "text embedding"],
    "s": [30, 60, 120, 300, 600]
}

def generate_3k():
    examples = []
    target = 3000
    
    print(f"Synthesizing {target} triples...")
    
    for i in range(target):
        # Select a template
        tpl = random.choice(TEMPLATES)
        name, domain, q_type, is_ghost, i_tpl, f_tpl, u_tpl = tpl
        
        # Populate with random data
        params = {k: random.choice(v) for k, v in DATA.items()}
        
        # Generate strings
        input_conv = i_tpl.format(**params)
        faithful = f_tpl.format(**params)
        unfaithful = u_tpl.format(**params)
        
        # Occasionally flip is_ghost for variety if template allows
        # (Technically we can make any template a ghost by removing the hedging)
        # But for now, we'll follow the template definition.
        
        example = {
            "id": f"synthetic_gold_{i:04d}",
            "source_scenario_id": f"blueprint_{name}_{i}",
            "qualifier_type": q_type,
            "domain": domain,
            "is_ghost": is_ghost,
            "input_conversation": input_conv,
            "faithful_summary": faithful,
            "unfaithful_summary": unfaithful
        }
        examples.append(example)
        
    # Save to file
    out_path = "data/synthetic_3k_gold.json"
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"Successfully generated {len(examples)} triples to {out_path}")

if __name__ == "__main__":
    generate_3k()
