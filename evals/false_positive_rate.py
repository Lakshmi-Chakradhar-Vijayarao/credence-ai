"""
false_positive_rate.py
======================
Gate 0 — The most critical unmeasured number in Credence.

Measures: what fraction of clearly non-uncertain sentences trigger the faithfulness probe?
Target:   FPR < 5%
Method:   200 sentences across 5 categories, zero epistemic uncertainty in any of them.

A probe with FPR > 10% blocks legitimate compressions constantly — it is a liability.
A probe with FPR < 5% is defensible.

Run:
    python3 -m evals.false_positive_rate
    python3 -m evals.false_positive_rate --verbose
    python3 -m evals.false_positive_rate --out evals/fpr_results.json
"""

import json, sys, time, argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from credence.context_manager import _UNCERTAINTY_MARKERS, ContextManager

_cm = ContextManager.__new__(ContextManager)

def _probe(text: str) -> bool:
    return _cm._has_uncertainty(text)

# ── 200 sentences, zero epistemic uncertainty ──────────────────────────────────

SENTENCES = {
    "technical_facts": [
        "Python uses indentation to define code blocks.",
        "SHA-256 produces a 256-bit hash output.",
        "HTTP status 404 means the resource was not found.",
        "Redis is an in-memory data structure store.",
        "TCP guarantees packet delivery and ordering.",
        "PostgreSQL supports ACID transactions.",
        "Git uses a directed acyclic graph to represent history.",
        "Docker containers share the host operating system kernel.",
        "JSON does not support comments.",
        "SQL PRIMARY KEY constraints enforce row uniqueness.",
        "RSA encryption uses two mathematically related keys.",
        "IPv4 addresses are 32 bits long.",
        "The HTTP GET method is idempotent.",
        "Base64 encoding increases data size by approximately 33 percent.",
        "TLS 1.3 removed support for RSA key exchange.",
        "WebSockets use the same port as HTTP by default.",
        "JWT tokens consist of three Base64-encoded sections.",
        "The Linux kernel is written primarily in C.",
        "Kubernetes uses etcd as its backing store.",
        "A mutex allows only one thread to hold it at a time.",
        "Unicode UTF-8 is backwards compatible with ASCII.",
        "A B-tree index supports range queries efficiently.",
        "gRPC uses Protocol Buffers for serialization by default.",
        "Kafka topics are divided into partitions.",
        "A foreign key references the primary key of another table.",
        "HTTPS uses port 443 by default.",
        "A REST API uses HTTP verbs to indicate the operation.",
        "The Python GIL prevents true multi-threaded CPU parallelism.",
        "Nginx can act as a reverse proxy and load balancer.",
        "A bloom filter can produce false positives but not false negatives.",
        "S3 objects are stored in buckets.",
        "GraphQL allows clients to specify exactly which fields they need.",
        "Prometheus stores metrics as time series data.",
        "A deadlock occurs when two threads each wait for the other to release a lock.",
        "OAuth 2.0 separates authentication from authorization.",
        "A CDN caches content at edge locations close to users.",
        "The OSI model has seven layers.",
        "A queue is a first-in first-out data structure.",
        "Elasticsearch uses inverted indexes for full-text search.",
        "CORS headers control which origins can make cross-origin requests.",
    ],
    "api_definitions": [
        "The API endpoint accepts POST requests to /api/v2/users.",
        "Authentication requires a Bearer token in the Authorization header.",
        "The response includes a pagination cursor in the next_cursor field.",
        "Error responses return a JSON object with an error and message field.",
        "The endpoint returns HTTP 200 on success and 400 on invalid input.",
        "Rate limiting is applied per API key, not per IP address.",
        "The API version is specified in the URL path, not the header.",
        "Requests must include Content-Type: application/json.",
        "The batch endpoint accepts up to 100 items per request.",
        "Webhook payloads are signed using HMAC-SHA256.",
        "The SDK supports Python 3.8 and above.",
        "Retry logic should implement exponential backoff.",
        "The session token is returned in a Set-Cookie header.",
        "API keys are generated in the developer dashboard.",
        "The endpoint requires the X-Request-ID header for tracing.",
        "File uploads use multipart/form-data encoding.",
        "The API uses snake_case for all JSON field names.",
        "Pagination uses cursor-based navigation, not page numbers.",
        "The timestamp fields use ISO 8601 format.",
        "Deleted resources return HTTP 204 with no body.",
        "The maximum request body size is 10 MB.",
        "Responses include an ETag header for conditional requests.",
        "The API gateway enforces a 30-second request timeout.",
        "OAuth scopes are space-separated in the authorization request.",
        "The token endpoint returns an access_token and refresh_token.",
        "API keys must be rotated every 90 days per the security policy.",
        "The v1 API is deprecated and will be removed on 2027-01-01.",
        "Idempotent requests use the Idempotency-Key header.",
        "The GraphQL endpoint is located at /graphql.",
        "Long-running operations return a job_id for polling.",
        "The API returns 429 when the rate limit is exceeded.",
        "Responses are compressed with gzip when the client sends Accept-Encoding: gzip.",
        "The SDK automatically refreshes expired tokens.",
        "Service accounts use a JSON key file for authentication.",
        "The API supports PATCH for partial updates.",
        "Logical deletes set the deleted_at timestamp rather than removing the row.",
        "The health endpoint is at /health and returns HTTP 200.",
        "CORS is enabled for all origins on the public API.",
        "The API uses semantic versioning.",
        "The request ID is echoed back in the X-Request-ID response header.",
    ],
    "code_and_errors": [
        "def calculate_total(items: list[float]) -> float:",
        "class UserRepository(BaseRepository):",
        "import asyncio",
        "from typing import Optional, List",
        "raise ValueError('Input must be a positive integer')",
        "ConnectionRefusedError: [Errno 111] Connection refused",
        "FileNotFoundError: No such file or directory: '/etc/config.yaml'",
        "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        "IndexError: list index out of range",
        "KeyError: 'user_id'",
        "return json.dumps(result, indent=2)",
        "assert len(items) > 0, 'Items list cannot be empty'",
        "logger.info('Request processed successfully in %d ms', elapsed)",
        "cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
        "response.raise_for_status()",
        "with open(config_path, 'r') as f:",
        "data = response.json()",
        "session.commit()",
        "os.makedirs(output_dir, exist_ok=True)",
        "subprocess.run(['git', 'push', 'origin', 'main'], check=True)",
        "timeout = int(os.environ.get('REQUEST_TIMEOUT', '30'))",
        "RATE_LIMIT = 100",
        "MAX_RETRIES = 3",
        "DEFAULT_PAGE_SIZE = 50",
        "TOKEN_EXPIRY_SECONDS = 3600",
        "PermissionError: [Errno 13] Permission denied: '/var/log/app.log'",
        "RuntimeError: CUDA out of memory",
        "OSError: [Errno 28] No space left on device",
        "StopIteration",
        "RecursionError: maximum recursion depth exceeded",
        "AttributeError: 'NoneType' object has no attribute 'id'",
        "OverflowError: integer overflow in range()",
        "UnicodeDecodeError: 'utf-8' codec can't decode byte",
        "ZeroDivisionError: division by zero",
        "ModuleNotFoundError: No module named 'numpy'",
        "SystemExit: 1",
        "WARNING: Retrying (Attempt 2 of 3)",
        "ERROR: Failed to connect to database after 3 retries",
        "INFO: Server started on port 8080",
        "DEBUG: Cache miss for key user:42",
    ],
    "documentation": [
        "Returns a list of User objects sorted by creation date.",
        "Raises ValueError if the input is not a valid email address.",
        "See the Authentication section for details on token handling.",
        "The configuration file must be placed in the project root.",
        "All timestamps are in UTC.",
        "The method is thread-safe.",
        "Calling this method twice has no additional effect.",
        "The connection pool size defaults to 10.",
        "Parameters are validated before the operation is executed.",
        "The response schema is documented in the API reference.",
        "This function runs in O(n log n) time.",
        "The cache is invalidated automatically on write operations.",
        "Child classes must implement the process() method.",
        "The batch size controls how many records are processed at once.",
        "Logs are written to stdout in JSON format.",
        "The worker processes tasks from the queue sequentially.",
        "Environment variables override configuration file settings.",
        "The health check endpoint returns 200 when the service is ready.",
        "Migration scripts are run in alphabetical order.",
        "Database connections are returned to the pool after use.",
        "The compression algorithm is selected based on content type.",
        "Sessions expire after 24 hours of inactivity.",
        "The event bus delivers messages at-least-once.",
        "Service discovery uses DNS-based routing.",
        "Backpressure is applied when the queue depth exceeds 1000.",
        "The circuit breaker opens after 5 consecutive failures.",
        "Audit logs are retained for 90 days.",
        "Secrets are stored in environment variables, not config files.",
        "The job scheduler runs tasks at the specified cron expression.",
        "Metrics are exported in OpenMetrics format.",
        "The middleware chain is executed in registration order.",
        "Graceful shutdown waits up to 30 seconds for requests to complete.",
        "The replica set requires a minimum of 3 nodes.",
        "Write-ahead logging ensures durability across crashes.",
        "The index is rebuilt automatically after bulk imports.",
        "Row-level security policies restrict data access by tenant.",
        "The deployment pipeline runs tests before promoting to production.",
        "Feature flags control rollout to specific user segments.",
        "The event store is append-only.",
        "Compression reduces storage by 60 to 80 percent for text data.",
    ],
    "assertions_and_definitions": [
        "A microservice should do one thing and do it well.",
        "Idempotent operations produce the same result when applied multiple times.",
        "A primary key uniquely identifies a row in a table.",
        "Normalization reduces data redundancy.",
        "A transaction is atomic: it either completes fully or not at all.",
        "Encryption at rest protects data stored on disk.",
        "Encryption in transit protects data moving over the network.",
        "A load balancer distributes traffic across multiple servers.",
        "Horizontal scaling adds more instances; vertical scaling adds more resources.",
        "A stateless service does not retain session data between requests.",
        "CAP theorem states that a distributed system can guarantee at most two of: consistency, availability, partition tolerance.",
        "Event sourcing stores state changes as a sequence of events.",
        "CQRS separates read and write models.",
        "The strangler fig pattern incrementally replaces a legacy system.",
        "Blue-green deployment keeps two production environments and switches traffic.",
        "A saga manages distributed transactions across services.",
        "Circuit breaking prevents cascading failures in distributed systems.",
        "Immutable infrastructure replaces rather than updates servers.",
        "Infrastructure as code manages servers through version-controlled files.",
        "Observability is built on three pillars: logs, metrics, and traces.",
        "A service mesh handles inter-service communication concerns.",
        "Rate limiting protects a service from being overwhelmed.",
        "The twelve-factor app methodology defines best practices for cloud-native applications.",
        "Continuous integration merges code frequently to detect conflicts early.",
        "A feature branch isolates development work until it is ready to merge.",
        "Semantic versioning uses MAJOR.MINOR.PATCH format.",
        "A pull request is a request to merge one branch into another.",
        "Code review improves quality and spreads knowledge.",
        "Unit tests verify individual functions in isolation.",
        "Integration tests verify that components work together correctly.",
        "End-to-end tests verify the full system from user input to output.",
        "Test-driven development writes tests before the implementation.",
        "Refactoring improves code structure without changing behaviour.",
        "Technical debt is the cost of shortcuts taken during development.",
        "A monorepo stores multiple projects in a single repository.",
        "GitOps uses Git as the source of truth for infrastructure state.",
        "Dependency injection decouples a class from its dependencies.",
        "The open-closed principle states that classes should be open for extension but closed for modification.",
        "A facade simplifies a complex subsystem behind a clean interface.",
        "The repository pattern abstracts data access behind an interface.",
    ],
}

assert sum(len(v) for v in SENTENCES.values()) == 200, \
    f"Expected 200 sentences, got {sum(len(v) for v in SENTENCES.values())}"


def run(verbose: bool = False) -> dict:
    results = {}
    total = 0
    total_fires = 0

    for category, sentences in SENTENCES.items():
        fires = []
        for s in sentences:
            fired = _probe(s)
            fires.append({"sentence": s, "fired": fired})
            if fired and verbose:
                print(f"  [FP] [{category}] {s[:80]}")
        n_fires = sum(1 for f in fires if f["fired"])
        fpr = n_fires / len(sentences)
        results[category] = {
            "n": len(sentences),
            "fires": n_fires,
            "fpr": round(fpr, 4),
            "false_positives": [f["sentence"] for f in fires if f["fired"]],
        }
        total += len(sentences)
        total_fires += n_fires

    overall_fpr = total_fires / total
    passed = overall_fpr < 0.05

    return {
        "total_sentences": total,
        "total_fires": total_fires,
        "overall_fpr": round(overall_fpr, 4),
        "passed": passed,
        "threshold": 0.05,
        "verdict": "PASS" if passed else ("WARN" if overall_fpr < 0.10 else "FAIL"),
        "categories": results,
        "marker_count": len(_UNCERTAINTY_MARKERS),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default="evals/fpr_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("CREDENCE — GATE 0: False Positive Rate Measurement")
    print("=" * 60)
    print(f"Probe markers: {len(_UNCERTAINTY_MARKERS)}")
    print(f"Test sentences: 200 (40 per category × 5 categories)")
    print(f"Target FPR: < 5%\n")

    t0 = time.perf_counter()
    results = run(verbose=args.verbose)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"{'Category':<25} {'N':>4} {'Fires':>6} {'FPR':>8}  {'Status'}")
    print("-" * 55)
    for cat, data in results["categories"].items():
        status = "✓" if data["fpr"] < 0.05 else ("⚠" if data["fpr"] < 0.10 else "✗")
        print(f"{cat:<25} {data['n']:>4} {data['fires']:>6} {data['fpr']:>8.1%}  {status}")

    print("-" * 55)
    print(f"{'OVERALL':<25} {results['total_sentences']:>4} "
          f"{results['total_fires']:>6} {results['overall_fpr']:>8.1%}  "
          f"{'✓ PASS' if results['passed'] else '✗ FAIL'}")

    print(f"\nProbe latency: {elapsed_ms/200:.3f}ms per sentence")
    print(f"\nVerdict: {results['verdict']}")

    if results["verdict"] == "PASS":
        print("→ Gate 0 OPEN. Probe FPR is within acceptable range.")
        print("  All downstream claims are defensible.")
    elif results["verdict"] == "WARN":
        print("→ Gate 0 MARGINAL. FPR between 5–10%.")
        print("  Document which categories cause false positives before claiming < 5%.")
        for cat, data in results["categories"].items():
            if data["fires"]:
                print(f"  {cat}: {data['fires']} FP(s):")
                for fp in data["false_positives"]:
                    print(f"    - {fp[:80]}")
    else:
        print("→ Gate 0 BLOCKED. FPR > 10%.")
        print("  Do not publish any claims until the marker list is pruned.")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
