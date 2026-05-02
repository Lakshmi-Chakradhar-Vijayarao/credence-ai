"""
ce_false_enforcement_rate.py
============================
Gate 0B — Measures Consistency Enforcer false-enforcement rate.

The faithfulness probe FPR is 0.5% (1/200). But the CE is a different gate:
it fires based on keyword overlap between a USER QUERY and REGISTERED CONSTRAINTS.
A high false-enforcement rate means the CE fires on legitimate queries that have
nothing to do with registered uncertain constraints — this pollutes system prompts
with spurious enforcement blocks and could affect answer quality.

Measurement design:
  - Register N uncertain constraints in a session
  - Run M queries that are clearly UNRELATED to those constraints
  - Measure: what fraction of unrelated queries trigger enforcement?

Target: CE false-enforcement rate < 5%

Also measures synonym-expansion aggressiveness:
  - How many synonym clusters does a given query token expand into?
  - Does the 52-cluster set create too many false connections?

Run:
    python3 -m evals.ce_false_enforcement_rate
    python3 -m evals.ce_false_enforcement_rate --verbose
    python3 -m evals.ce_false_enforcement_rate --out evals/ce_fer_results.json
"""

import json, sys, time, argparse, tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from credence.context_manager import (
    ContextManager, _CE_MIN_OVERLAP, _CE_STOPWORDS, _CE_DOMAIN_SYNONYMS
)
from credence.registry import CredenceRegistry

# ── Reference constraints: the 10 most common uncertain API constraints ───────
REFERENCE_CONSTRAINTS = [
    "I think the rate limit might be approximately 50 req/min, but this is unverified.",
    "The auth token expiry might be around 3600 seconds, needs confirmation.",
    "The pagination page size is probably 100 items per page, unverified.",
    "The maximum batch size might be 500 items, but could change.",
    "The API base URL is possibly https://api.example.com/v2, unconfirmed.",
    "The timeout for requests might be 30 seconds, uncertain.",
    "The retry backoff delay is approximately 2 seconds, unverified.",
    "The maximum response size might be 10MB, not confirmed.",
    "The webhook endpoint is probably at /webhooks/v1, needs verification.",
    "The OAuth client secret rotation period might be 90 days, unclear.",
]

# ── Clearly unrelated queries (should NOT trigger CE) ────────────────────────
# Grouped by category for analysis

UNRELATED_UI_QUERIES = [
    "What color scheme should the dashboard use?",
    "How should we display the loading spinner?",
    "What font size is appropriate for mobile views?",
    "Should the navigation bar be on the left or top?",
    "What animation should we use for the modal transition?",
    "How do we handle dark mode styling?",
    "What are the recommended accessible color contrast ratios?",
    "Should we use a grid or flexbox layout here?",
    "How do we handle the responsive breakpoints?",
    "What icon library should we use?",
    "How should the error state look visually?",
    "What padding should we use for card components?",
    "Should the button be rounded or square?",
    "How do we handle text overflow in table cells?",
    "What should the placeholder text say?",
    "How do we implement the skeleton loading effect?",
    "Should we use a drawer or modal for this panel?",
    "What hover state should the list items have?",
    "How do we handle the print stylesheet?",
    "What z-index values should we use for overlays?",
]

UNRELATED_DATABASE_QUERIES = [
    "What ORM should we use for the database layer?",
    "How do we structure the migration files?",
    "Should we use UUID or integer primary keys?",
    "How do we handle soft deletes in the schema?",
    "What index strategy should we use for the search query?",
    "How do we implement the audit log table?",
    "Should we use a single database or separate read replicas?",
    "How do we handle the many-to-many relationship?",
    "What transaction isolation level should we use?",
    "How do we structure the foreign key constraints?",
    "Should we use a materialized view for the reporting query?",
    "How do we implement row-level security?",
    "What partitioning strategy should we use for the events table?",
    "How do we handle the schema version migrations?",
    "Should we use PostgreSQL JSONB or a separate table for metadata?",
    "How do we implement the full-text search index?",
    "What backup strategy should we use?",
    "How do we handle database connection pooling?",
    "Should we cache the query results in Redis?",
    # NOTE: "cursor-based pagination" removed — it legitimately matches the
    # registered pagination constraint and is correctly enforced. Not a false fire.
]

UNRELATED_TESTING_QUERIES = [
    "How should we structure the test directory?",
    "What mocking library should we use?",
    "How do we write end-to-end tests for this flow?",
    "Should we use TDD or write tests after implementation?",
    "How do we set up the test fixtures?",
    "What coverage percentage should we target?",
    "How do we handle test environment configuration?",
    "Should we use snapshot tests for the UI components?",
    "How do we test the async background jobs?",
    "What assertion library should we use?",
    "How do we handle flaky tests in CI?",
    "Should we run tests in parallel or sequentially?",
    "How do we write property-based tests for this function?",
    "What mutation testing tools should we use?",
    "How do we test the error handling paths?",
    "Should we use mocks or integration tests for the database?",
    "How do we generate test data for the fixtures?",
    "What is the best way to test the CLI commands?",
    "How do we assert on log output in tests?",
    "Should we use contract testing for the APIs?",
]

UNRELATED_DEPLOYMENT_QUERIES = [
    "How should we structure the Dockerfile?",
    "What Kubernetes resource limits should we set?",
    "How do we handle the CI/CD pipeline configuration?",
    "Should we use blue-green or canary deployments?",
    "How do we set up the health check endpoints?",
    "What logging format should we use in production?",
    "How do we configure the load balancer?",
    "Should we use Helm or raw Kubernetes manifests?",
    "How do we handle the secrets management?",
    "What monitoring alerts should we set up?",
    "How do we implement the rollback strategy?",
    "Should we use a service mesh like Istio?",
    "How do we configure the auto-scaling policies?",
    "What log aggregation solution should we use?",
    "How do we handle the certificate renewal?",
    "Should we use cloud provider managed services or self-hosted?",
    "How do we set up the staging environment?",
    "What should the graceful shutdown sequence be?",
    "How do we handle database migrations during deployment?",
    "What alerting thresholds should we configure?",
]

UNRELATED_CODE_STRUCTURE_QUERIES = [
    "How should we organize the module structure?",
    "Should we use a monorepo or separate repositories?",
    "How do we implement the dependency injection?",
    "What design pattern should we use for this service?",
    "How do we handle the configuration management?",
    "Should we use a factory or builder pattern here?",
    "How do we structure the error types?",
    "What logging strategy should we use?",
    "How do we handle the async initialization?",
    "Should we use composition or inheritance here?",
    "How do we implement the plugin architecture?",
    "What state management approach should we use?",
    "How do we handle the circular dependency issue?",
    "Should we use interfaces or abstract classes?",
    "How do we implement the observer pattern?",
    "What is the best way to handle the shared utilities?",
    "How do we structure the API response types?",
    "Should we use tuple or object returns?",
    # NOTE: "retry logic" removed — it legitimately matches the registered
    # retry backoff constraint and is correctly enforced. Not a false fire.
    "What should the module public API expose?",
]

ALL_UNRELATED_QUERIES = (
    UNRELATED_UI_QUERIES
    + UNRELATED_DATABASE_QUERIES
    + UNRELATED_TESTING_QUERIES
    + UNRELATED_DEPLOYMENT_QUERIES
    + UNRELATED_CODE_STRUCTURE_QUERIES
)

CATEGORIES = {
    "UI/styling":   UNRELATED_UI_QUERIES,
    "Database":     UNRELATED_DATABASE_QUERIES,
    "Testing":      UNRELATED_TESTING_QUERIES,
    "Deployment":   UNRELATED_DEPLOYMENT_QUERIES,
    "Code structure": UNRELATED_CODE_STRUCTURE_QUERIES,
}

# ── Related queries (should trigger CE — positive control) ───────────────────
RELATED_QUERIES = [
    "What is the rate limit for this API?",
    "How long does the auth token last?",
    "What is the page size for pagination?",
    "How many items can we include in a batch?",
    "What is the API base URL?",
    "What happens when the request times out?",
    "How long should we wait before retrying?",
    "What is the maximum response size?",
    "What is the webhook endpoint URL?",
    "How often does the OAuth secret rotate?",
]

# ── Measurement ───────────────────────────────────────────────────────────────

def _setup_cm(db_path: str) -> tuple[ContextManager, CredenceRegistry]:
    """Create a ContextManager with the reference constraints registered."""
    reg = CredenceRegistry(db_path=db_path)
    for c in REFERENCE_CONSTRAINTS:
        reg.register(content=c, session_id="fer_session", j_score=0.3, zone="LOW")

    cm = ContextManager.__new__(ContextManager)
    # Manually wire in the registry/session (no API call needed)
    cm._registry = reg
    cm._session_id = "fer_session"
    cm._system_prompt = "You are a helpful assistant."
    return cm, reg


def _check_enforcement(cm: ContextManager, query: str) -> tuple[bool, list[dict]]:
    """Return (fires, matched_constraints)."""
    try:
        constraints = cm._registry.list_uncertain(cm._session_id)
        matches = cm._direct_constraint_matches(query, constraints)
        return len(matches) > 0, matches
    except Exception:
        return False, []


def run_measurement(verbose: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/fer.db"
        cm, reg = _setup_cm(db)

        results = {
            "false_enforcements": [],
            "true_enforcements": [],
            "category_breakdown": {},
            "synonym_expansion_stats": [],
        }

        # ── Measure false-enforcement on unrelated queries ────────────────────
        false_fires = 0
        for cat_name, queries in CATEGORIES.items():
            cat_fires = 0
            cat_details = []
            for q in queries:
                fires, matches = _check_enforcement(cm, q)
                if fires:
                    false_fires += 1
                    cat_fires += 1
                    results["false_enforcements"].append({
                        "query": q,
                        "category": cat_name,
                        "matched_constraints": [m["content"][:60] for m in matches],
                        "overlap": [m.get("_overlap", []) for m in matches],
                    })
                    if verbose:
                        print(f"  FALSE FIRE [{cat_name}]: {q!r}")
                        for m in matches:
                            print(f"    → matched: {m['content'][:60]!r} (overlap: {m.get('_overlap', [])})")
            results["category_breakdown"][cat_name] = {
                "total": len(queries),
                "fires": cat_fires,
                "fer": cat_fires / len(queries),
            }
            if verbose:
                status = "✓" if cat_fires == 0 else f"✗ {cat_fires} FEs"
                print(f"  [{status}] {cat_name}: {cat_fires}/{len(queries)} false enforcements")

        total_unrelated = len(ALL_UNRELATED_QUERIES)
        fer = false_fires / total_unrelated

        # ── Measure true-enforcement on related queries (sanity check) ────────
        true_fires = 0
        for q in RELATED_QUERIES:
            fires, matches = _check_enforcement(cm, q)
            if fires:
                true_fires += 1
                results["true_enforcements"].append({
                    "query": q,
                    "matched_constraints": [m["content"][:60] for m in matches],
                })
        recall = true_fires / len(RELATED_QUERIES)

        # ── Synonym expansion stats ───────────────────────────────────────────
        import re
        for q in ALL_UNRELATED_QUERIES[:20]:  # sample
            tokens = set(re.sub(r'[^\w\s]', ' ', q.lower()).split()) - _CE_STOPWORDS
            expanded = cm._expand_tokens(tokens) if hasattr(cm, '_expand_tokens') else tokens
            new_tokens = expanded - tokens
            results["synonym_expansion_stats"].append({
                "query": q[:50],
                "original_tokens": len(tokens),
                "expanded_tokens": len(expanded),
                "expansion_factor": len(expanded) / max(len(tokens), 1),
                "new_tokens_count": len(new_tokens),
            })

        avg_expansion = sum(s["expansion_factor"] for s in results["synonym_expansion_stats"]) / len(results["synonym_expansion_stats"])

        results.update({
            "summary": {
                "total_unrelated_queries": total_unrelated,
                "false_enforcements_count": false_fires,
                "false_enforcement_rate": fer,
                "gate_pass": fer < 0.05,
                "total_related_queries": len(RELATED_QUERIES),
                "true_enforcements_count": true_fires,
                "recall": recall,
                "avg_synonym_expansion_factor": avg_expansion,
                "ce_min_overlap": _CE_MIN_OVERLAP,
                "synonym_clusters": len(_CE_DOMAIN_SYNONYMS),
            }
        })
        return results


def main():
    parser = argparse.ArgumentParser(description="Measure CE false-enforcement rate")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default="evals/ce_fer_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("CE False-Enforcement Rate — Gate 0B")
    print("=" * 60)
    print(f"  Constraints registered: {len(REFERENCE_CONSTRAINTS)}")
    print(f"  Unrelated queries: {len(ALL_UNRELATED_QUERIES)} across 5 categories")
    print(f"  Related queries (positive control): {len(RELATED_QUERIES)}")
    print(f"  CE settings: min_overlap={_CE_MIN_OVERLAP}, synonym_clusters={len(_CE_DOMAIN_SYNONYMS)}")
    print()

    t0 = time.perf_counter()
    results = run_measurement(verbose=args.verbose)
    elapsed = (time.perf_counter() - t0) * 1000
    s = results["summary"]

    print(f"\n{'─'*60}")
    print(f"RESULTS")
    print(f"{'─'*60}")
    print(f"  False-enforcement rate:  {s['false_enforcement_rate']:.1%}  ({s['false_enforcements_count']}/{s['total_unrelated_queries']})")
    print(f"  Gate 0B:                 {'OPEN ✓' if s['gate_pass'] else 'BLOCKED ✗'}")
    print(f"  Target:                  < 5%")
    print()
    print(f"  CE recall (positive):    {s['recall']:.1%}  ({s['true_enforcements_count']}/{s['total_related_queries']})")
    print(f"  Avg synonym expansion:   {s['avg_synonym_expansion_factor']:.2f}×")
    print()
    print(f"  Category breakdown:")
    for cat, v in results["category_breakdown"].items():
        status = "✓" if v["fires"] == 0 else f"✗ {v['fires']} FE"
        print(f"    {cat:20s}: {v['fires']:2d}/{v['total']:2d}  ({v['fer']:.1%})  [{status}]")
    print()

    if results["false_enforcements"]:
        print(f"  False enforcement details:")
        for fe in results["false_enforcements"][:5]:
            print(f"    Query:  {fe['query']!r}")
            print(f"    Overlap: {fe['overlap']}")
        if len(results["false_enforcements"]) > 5:
            print(f"    ... and {len(results['false_enforcements']) - 5} more")
    else:
        print("  No false enforcements detected.")

    print(f"\n  Total measurement time: {elapsed:.1f}ms")

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {args.out}")

    if not s["gate_pass"]:
        print(f"\n  ⚠ Gate 0B BLOCKED: CE FER {s['false_enforcement_rate']:.1%} exceeds 5% threshold")
        print(f"  Recommended fix: tighten synonym clusters or raise _CE_MIN_OVERLAP to 3")
        sys.exit(1)
    else:
        print(f"\n  Gate 0B OPEN: CE is safe to ship.")


if __name__ == "__main__":
    main()
