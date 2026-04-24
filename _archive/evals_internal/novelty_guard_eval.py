"""
evals/novelty_guard_eval.py
============================
Measures novelty guard false-positive and false-negative rates.

The novelty guard (_check_novelty) is supposed to:
  - FIRE (return True)  on topic pivots — new domain entered
  - NOT fire (False)    on stable-domain sessions — same topic throughout

Without measurement, we don't know if it fires 5% or 30% of the time on
stable sessions. A high false-positive rate means CAMS is refusing valid
compressions constantly, explaining the high token count vs naive window.

This eval is pure Python (no API) — it feeds synthetic responses through
_check_novelty() and measures accuracy.

Stable domain sessions: same technical domain throughout
  → expected: _check_novelty() returns False on all turns
  → false positive = fires on a stable session

Topic pivot sessions: abrupt domain change mid-session
  → expected: _check_novelty() returns True at pivot turn
  → false negative = fails to fire at pivot

Run:
    python -m evals.novelty_guard_eval

Results saved to evals/novelty_guard_results.json
No API key required.
"""

import os, sys, json
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credence.context_manager import ContextManager


# ---------------------------------------------------------------------------
# Synthetic response sequences
# ---------------------------------------------------------------------------

# Stable domain: all responses about the same technical area
STABLE_SEQUENCES = [
    {
        "name": "Database engineering",
        "responses": [
            "PostgreSQL uses MVCC (Multi-Version Concurrency Control) to handle concurrent transactions without locking.",
            "An index scan in PostgreSQL traverses the B-tree index structure to find matching rows, then fetches them from the heap.",
            "The EXPLAIN ANALYZE command shows the actual execution plan with real row counts and timing statistics.",
            "Connection pooling reduces overhead by reusing database connections across multiple application requests.",
            "Vacuum in PostgreSQL reclaims space from dead tuples created by MVCC's update and delete operations.",
            "Partitioning a large table by date range can dramatically improve query performance on time-series data.",
            "The query planner uses table statistics collected by ANALYZE to estimate selectivity and choose execution plans.",
            "Foreign key constraints enforce referential integrity but can slow down bulk inserts if not deferred.",
        ],
    },
    {
        "name": "Machine learning training",
        "responses": [
            "Gradient descent updates model weights by computing the gradient of the loss function with respect to each parameter.",
            "Batch normalization normalizes layer inputs during training, which stabilizes gradients and allows higher learning rates.",
            "Dropout randomly deactivates neurons during training, acting as an ensemble of different network architectures.",
            "The learning rate schedule controls how step size changes during training — warmup then cosine decay is common.",
            "Validation loss diverging from training loss indicates overfitting — the model memorizes training data.",
            "Weight initialization matters significantly; Xavier and He initialization help prevent vanishing gradients.",
            "Mixed precision training uses float16 for most computations to reduce memory usage and speed up training.",
            "Early stopping monitors validation metrics and halts training when improvement plateaus.",
        ],
    },
    {
        "name": "HTTP and web APIs",
        "responses": [
            "RESTful APIs use HTTP verbs — GET retrieves resources, POST creates, PUT replaces, PATCH updates, DELETE removes.",
            "JWT tokens consist of three base64-encoded sections: header, payload, and signature separated by dots.",
            "Rate limiting protects APIs from abuse by restricting how many requests a client can make per time window.",
            "CORS headers tell browsers which origins are allowed to make cross-site requests to an API endpoint.",
            "HTTP/2 multiplexes multiple requests over a single TCP connection, reducing latency from connection overhead.",
            "Idempotent operations produce the same result regardless of how many times they are executed.",
            "OAuth 2.0 uses authorization codes and refresh tokens to delegate access without sharing credentials.",
            "The ETag header enables conditional requests — clients send the value back to check if content changed.",
        ],
    },
]

# Topic pivot sequences: abrupt domain change at turn N
PIVOT_SEQUENCES = [
    {
        "name": "Database → Cryptography pivot at turn 5",
        "pivot_turn": 4,  # 0-indexed
        "responses": [
            "PostgreSQL indexes improve query performance by allowing the planner to find rows without scanning the full table.",
            "Write-ahead logging ensures durability by recording changes before they are applied to data files.",
            "Connection pools like PgBouncer sit between the application and PostgreSQL to reuse idle connections.",
            "Vacuum reclaims storage from dead tuples and updates statistics used by the query planner.",
            # Pivot turn — cryptography domain
            "RSA encryption relies on the computational difficulty of factoring the product of two large prime numbers.",
            "SHA-256 produces a 256-bit hash digest that is deterministic but practically irreversible.",
            "Elliptic curve cryptography achieves equivalent security to RSA with much shorter key lengths.",
            "HMAC combines a secret key with a hash function to provide both integrity and authenticity verification.",
        ],
    },
    {
        "name": "ML training → Systems programming pivot at turn 4",
        "pivot_turn": 3,
        "responses": [
            "Gradient descent iteratively adjusts model weights to minimize the training loss function.",
            "Convolutional layers detect local patterns in images by applying learned filters across the input.",
            "The attention mechanism allows transformers to weigh the relevance of each position to every other position.",
            # Pivot turn — systems programming
            "Memory safety in Rust is enforced at compile time through the ownership and borrowing system.",
            "Stack allocation is faster than heap allocation because it requires only adjusting the stack pointer.",
            "Garbage collectors reclaim memory automatically but introduce pause times that affect latency.",
            "Zero-cost abstractions in C++ and Rust mean high-level constructs compile to the same code as manual implementations.",
        ],
    },
    {
        "name": "Web APIs → Compiler design pivot at turn 6",
        "pivot_turn": 5,
        "responses": [
            "HTTP status codes communicate the result of a request — 2xx success, 4xx client error, 5xx server error.",
            "Pagination limits the size of API responses and uses cursors or page numbers to navigate large result sets.",
            "Webhooks push event notifications to a registered URL instead of requiring the client to poll.",
            "API versioning strategies include URL path versioning, header versioning, and query parameter versioning.",
            "Content negotiation allows clients to request specific response formats via the Accept header.",
            # Pivot turn — compiler design
            "Lexical analysis tokenizes source code into a stream of tokens that the parser can process.",
            "Abstract syntax trees represent the hierarchical structure of parsed source code in memory.",
            "Register allocation assigns variables to CPU registers to minimize expensive memory accesses.",
        ],
    },
]


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

@dataclass
class NoveltyEvalResult:
    sequence_name:     str
    sequence_type:     str   # "stable" | "pivot"
    pivot_turn:        int   # -1 for stable
    turn_firings:      list[bool]   # True = novelty guard fired
    false_positives:   int   # stable: any firing; pivot: firings before pivot
    false_negatives:   int   # pivot: guard did not fire at or after pivot turn
    true_positives:    int   # pivot: guard fired at or after pivot turn
    accuracy_note:     str


def _run_sequence(responses: list[str], name: str, seq_type: str,
                  pivot_turn: int = -1) -> NoveltyEvalResult:
    mgr = ContextManager.__new__(ContextManager)
    mgr._content_vocab        = set()
    mgr._recent_vocab_window  = []

    firings = []
    for i, response in enumerate(responses):
        fired = mgr._check_novelty(response)
        mgr._update_content_vocab(response)
        firings.append(fired)
        status = "FIRE" if fired else "    "
        marker = " ← PIVOT" if i == pivot_turn else ""
        print(f"    [{status}] turn {i+1}: {response[:60]}…{marker}")

    if seq_type == "stable":
        fp = sum(firings)
        fn = 0
        tp = 0
        note = f"{fp} false positives (should be 0)"
    else:
        fp = sum(firings[:pivot_turn])
        tp = sum(firings[pivot_turn:])
        fn = 1 if tp == 0 else 0
        note = f"tp={tp} fp={fp} fn={fn}"

    return NoveltyEvalResult(
        sequence_name=name,
        sequence_type=seq_type,
        pivot_turn=pivot_turn,
        turn_firings=firings,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        accuracy_note=note,
    )


def main():
    print("Novelty Guard Evaluation")
    print("=" * 60)
    results = []

    print("\nSTABLE SEQUENCES (guard should NOT fire):")
    print("-" * 60)
    total_stable_fp = 0
    for seq in STABLE_SEQUENCES:
        print(f"\n  {seq['name']}")
        r = _run_sequence(seq["responses"], seq["name"], "stable")
        results.append(r)
        total_stable_fp += r.false_positives
        status = "PASS" if r.false_positives == 0 else f"FAIL ({r.false_positives} FP)"
        print(f"  Result: {status}")

    print("\n\nPIVOT SEQUENCES (guard SHOULD fire at pivot):")
    print("-" * 60)
    total_pivot_tp = 0
    total_pivot_fn = 0
    for seq in PIVOT_SEQUENCES:
        print(f"\n  {seq['name']}")
        r = _run_sequence(seq["responses"], seq["name"], "pivot", seq["pivot_turn"])
        results.append(r)
        total_pivot_tp += r.true_positives
        total_pivot_fn += r.false_negatives
        status = "PASS" if r.true_positives > 0 and r.false_positives == 0 else \
                 f"PARTIAL" if r.true_positives > 0 else "FAIL (missed pivot)"
        print(f"  Result: {status}  ({r.accuracy_note})")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    stable_fp_rate = total_stable_fp / sum(len(s["responses"]) for s in STABLE_SEQUENCES)
    pivot_recall   = total_pivot_tp / len(PIVOT_SEQUENCES)
    print(f"  Stable sessions:  false positive rate = {stable_fp_rate*100:.1f}%  "
          f"({total_stable_fp} fires / {sum(len(s['responses']) for s in STABLE_SEQUENCES)} turns)")
    print(f"  Pivot sessions:   detection rate      = {pivot_recall*100:.1f}%  "
          f"({total_pivot_tp}/{len(PIVOT_SEQUENCES)} pivots detected)")
    print()
    if stable_fp_rate <= 0.10:
        print("  ✓ False positive rate acceptable (≤10%)")
    else:
        print(f"  ✗ False positive rate HIGH ({stable_fp_rate*100:.1f}%) — "
              f"guard is over-firing on stable sessions, blocking valid compressions")
    if pivot_recall >= 0.67:
        print("  ✓ Pivot detection rate acceptable (≥67%)")
    else:
        print(f"  ✗ Pivot detection rate LOW ({pivot_recall*100:.1f}%) — "
              f"guard is missing topic changes")
    print("=" * 60)

    output = {
        "stable_fp_rate":   round(stable_fp_rate, 4),
        "pivot_recall":     round(pivot_recall, 4),
        "total_stable_fp":  total_stable_fp,
        "total_pivot_tp":   total_pivot_tp,
        "total_pivot_fn":   total_pivot_fn,
        "sequences":        [asdict(r) for r in results],
    }
    out_path = "evals/novelty_guard_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
