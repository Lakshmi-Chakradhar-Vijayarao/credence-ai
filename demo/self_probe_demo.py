"""
demo/self_probe_demo.py
========================
The undeniable demo: AI-generated values caught before they ship.

Run:
    python demo/self_probe_demo.py

No API key needed. Uses the SQLite registry only.

Flow:
  1. User mentions Stripe integration — some values are uncertain
  2. AI generates StripeClient code (as any coding agent would)
  3. credence_self_probe extracts all domain-relevant values
  4. credence_scan annotates the code with CREDENCE tiers
  5. credence_gate blocks the file write
  6. User confirms the rate limit → credence_autoverify clears it
  7. credence_gate re-checked → now proceeds for verified values
  8. Final annotated output shows only remaining unverified values
"""

import os
import sys
import tempfile
import textwrap

# ── Setup: use a throwaway DB so demo is always fresh ──────────────────────
_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_db.close()
os.environ["CREDENCE_DB_PATH"] = _db.name

from credence.registry import CredenceRegistry
from credence.temporal_patterns import scan_temporal, scan_domain_assignments
from credence.mcp_server import _scan_output, _get_registry, _expand_tokens

SESSION = "stripe_demo_session"

# ── Formatting helpers ──────────────────────────────────────────────────────

def _hr(char="─", width=70):
    print(char * width)

def _section(title: str):
    print()
    _hr("═")
    print(f"  {title}")
    _hr("═")

def _step(n: int, label: str):
    print()
    _hr()
    print(f"  Step {n}: {label}")
    _hr()

def _user(msg: str):
    print(f"\n  👤  USER: {msg}")

def _agent(msg: str):
    print(f"\n  🤖  AGENT: {msg}")

def _tool(name: str, result: dict):
    print(f"\n  ⚙  {name}:")
    for k, v in result.items():
        if k in ("annotated_output", "message", "annotation_hint"):
            continue
        if isinstance(v, list) and len(v) > 3:
            print(f"       {k}: [{len(v)} items]")
        else:
            print(f"       {k}: {v}")

# ── The generated code ──────────────────────────────────────────────────────

GENERATED_CODE = '''\
```python
import requests

class StripeClient:
    """Stripe API client."""

    BASE_URL    = "https://api.stripe.com/v1"
    API_VERSION = "2023-10-16"
    RATE_LIMIT  = 100
    TOKEN_EXPIRY = 3600
    MAX_RETRIES  = 3
    TIMEOUT_MS   = 5000

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Stripe-Version": self.API_VERSION,
        })

    def charge(self, amount: int, currency: str = "usd") -> dict:
        resp = self.session.post(
            f"{self.BASE_URL}/charges",
            json={"amount": amount, "currency": currency},
            timeout=self.TIMEOUT_MS / 1000,
        )
        resp.raise_for_status()
        return resp.json()
```'''


def run_demo():
    reg = _get_registry()

    _section("CREDENCE SELF-PROBE DEMO")
    print("""
  Scenario: You ask a coding agent to write a Stripe integration.
  The agent generates a StripeClient class.
  Before that code reaches you, Credence intercepts it.
    """)

    # ── Step 1: User registers an uncertain value ───────────────────────────
    _step(1, "User states an uncertain value")
    _user("I think the Stripe rate limit is around 100 requests per minute.")

    cid_rate = reg.register(
        content    = "I think the Stripe rate limit is around 100 requests per minute",
        session_id = SESSION,
        j_score    = 0.30,
        zone       = "LOW",
        source     = "user_stated",
        constraint_type = "vendor_claim",
    )
    _tool("credence_register", {
        "constraint_id": cid_rate,
        "content":       "I think the Stripe rate limit is around 100 req/min",
        "j_score":       0.30,
        "source_type":   "vendor_claim",
        "status":        "registered as UNVERIFIED",
    })
    _agent("Registered. I'll enforce this before any code using this value ships.")

    # ── Step 2: Agent generates code ────────────────────────────────────────
    _step(2, "Agent generates StripeClient code")
    _agent("Here is the generated code (before Credence intercepts it):")
    print()
    for line in GENERATED_CODE.splitlines():
        print(f"    {line}")

    # ── Step 3: credence_self_probe ─────────────────────────────────────────
    _step(3, "credence_self_probe — zero API, zero model judgment")
    _agent("Calling credence_self_probe before showing you this code...")

    t_hits = scan_temporal(GENERATED_CODE)
    d_hits = scan_domain_assignments(GENERATED_CODE)

    stale_ids = []
    for h in t_hits:
        cid = reg.register(
            content=h.constraint_content,
            session_id=SESSION,
            j_score=h.j_score,
            zone="LOW",
            source="temporal_scan",
            constraint_type="vendor_claim",
        )
        stale_ids.append((cid, h.value, h.category, h.verify_hint))

    domain_ids = []
    for h in d_hits:
        cid = reg.register(
            content=h.constraint_content,
            session_id=SESSION,
            j_score=h.j_score,
            zone="LOW",
            source="self_probe",
            constraint_type="config",
        )
        domain_ids.append((cid, h.var_name, h.domain))

    print(f"\n  ⚙  credence_self_probe:")
    print(f"       stale_count:    {len(stale_ids)}  (temporal patterns)")
    print(f"       domain_count:   {len(domain_ids)}  (domain assignments)")
    print(f"       total_registered: {len(stale_ids)+len(domain_ids)}")
    print(f"       principle:      Unknown = unverified. No confidence scoring.")
    print()
    print("       STALE (auto-registered, verify before ship):")
    for cid, val, cat, hint in stale_ids:
        print(f"         [{cat}]  {val!r:25s}  → {hint}")
    print()
    print("       DOMAIN-UNCERTAIN (auto-registered, unverified by default):")
    for cid, var, domain in domain_ids:
        print(f"         [{domain}]  {var}")

    # ── Step 4: credence_scan — annotated output ────────────────────────────
    _step(4, "credence_scan — annotated output (what the user actually sees)")

    annotated, hits = _scan_output(GENERATED_CODE, reg, SESSION, turn=0)

    _agent("Annotated code with CREDENCE tiers:")
    print()
    for line in annotated.splitlines():
        print(f"    {line}")

    print(f"\n  Total annotations: {len(hits)}")
    tiers = {}
    for h in hits:
        line = h.get("line", "")
        if "stale" in line:
            tiers["stale"] = tiers.get("stale", 0) + 1
        elif "HIGH RISK" in line:
            tiers["HIGH RISK"] = tiers.get("HIGH RISK", 0) + 1
        elif "AI-generated" in line:
            tiers["AI-generated"] = tiers.get("AI-generated", 0) + 1
        elif "unverified" in line:
            tiers["unverified"] = tiers.get("unverified", 0) + 1
    for tier, count in tiers.items():
        print(f"    {tier}: {count}")

    # ── Step 5: credence_gate blocks the file write ─────────────────────────
    _step(5, "credence_gate — blocks file write")
    _agent("Checking gate before writing stripe_client.py...")

    uncertain = reg.list_uncertain(SESSION)
    # action text covers the write — uses real synonym expansion (rate→rate_limit etc.)
    action_text = "write stripe_client rate limit token expiry timeout version retries"
    import re as _re
    from credence.mcp_server import _CE_STOPWORDS
    raw_tokens    = set(_re.sub(r"[^\w\s]", " ", action_text.lower()).split()) - _CE_STOPWORDS
    action_tokens = _expand_tokens(raw_tokens)
    blocking = []
    for c in uncertain:
        c_raw    = set(_re.sub(r"[^\w\s]", " ", c["content"].lower()).split()) - _CE_STOPWORDS
        c_tokens = _expand_tokens(c_raw)
        if len(action_tokens & c_tokens) >= 2:
            blocking.append(c)

    if blocking:
        print(f"\n  ⚙  credence_gate:")
        print(f"       proceed:          False")
        print(f"       blocked_by:       {len(blocking)} constraint(s)")
        for b in blocking[:4]:
            print(f"         → {b['content'][:70]}")
        print(f"       recommendation:   BLOCK — verify before writing file")
        _agent("BLOCKED. The following values are unverified and overlap with this write:")
        for b in blocking[:4]:
            print(f"    ⚠  {b['content'][:80]}")
        print("\n  Cannot write stripe_client.py until values are verified.")
    else:
        print(f"\n  ⚙  credence_gate: proceed=True (no overlapping unverified constraints)")

    # ── Step 6: User confirms rate limit ────────────────────────────────────
    _step(6, "User confirms the rate limit with evidence")
    _user("Actually confirmed — it's 100 req/min per Stripe docs at stripe.com/docs/rate-limits")

    confirm_text = "confirmed it is 100 req/min per Stripe docs at stripe.com/docs/rate-limits"
    confirm_lower = confirm_text.lower()
    _CONFIRM = frozenset({"confirmed","actually","i checked","per the docs","per docs",
                          "verified","it is","as per","according to"})
    has_signal = any(p in confirm_lower for p in _CONFIRM)
    verified_ids = []
    if has_signal:
        for c in reg.list_uncertain(SESSION):
            c_tokens = set(_re.sub(r"[^\w\s]", " ", c["content"].lower()).split()) - _CE_STOPWORDS
            confirm_tokens = set(_re.sub(r"[^\w\s]", " ", confirm_lower).split()) - _CE_STOPWORDS
            if len(confirm_tokens & c_tokens) >= 2:
                reg.verify(c["constraint_id"], f"auto-verified: {confirm_text[:80]}")
                verified_ids.append(c["constraint_id"])

    _tool("credence_autoverify", {
        "confirmed_signal":  "yes",
        "verified_count":    len(verified_ids),
        "verified_ids":      verified_ids,
        "message":           f"Auto-verified {len(verified_ids)} constraint(s) matching confirmation",
    })

    # ── Step 7: Gate re-checked ─────────────────────────────────────────────
    _step(7, "credence_gate re-checked after verification")
    _agent("Re-checking gate for stripe_client.py write...")

    uncertain_after = reg.list_uncertain(SESSION)
    blocking_after = []
    for c in uncertain_after:
        c_raw_a    = set(_re.sub(r"[^\w\s]", " ", c["content"].lower()).split()) - _CE_STOPWORDS
        c_tokens_a = _expand_tokens(c_raw_a)
        if len(action_tokens & c_tokens_a) >= 2:
            blocking_after.append(c)

    print(f"\n  ⚙  credence_gate (after verification):")
    print(f"       unverified_remaining: {len(uncertain_after)}")
    print(f"       blocking_count:       {len(blocking_after)}")
    if len(blocking_after) == 0:
        cleared = len(blocking) - len(blocking_after)
        if cleared > 0:
            print(f"       cleared:             {cleared} constraint(s) verified by user")
        print(f"       recommendation:      PROCEED — all blocking constraints cleared")
        _agent("Rate limit verified. File write proceeds. Remaining unverified values are annotated in code.")
    else:
        print(f"       recommendation:      BLOCK — {len(blocking_after)} constraint(s) still unverified")
        for b in blocking_after[:3]:
            print(f"         → {b['content'][:70]}")

    # ── Step 8: Final summary ───────────────────────────────────────────────
    _section("SUMMARY")

    all_constraints = reg._conn.execute(
        "SELECT * FROM constraints WHERE session_id=? ORDER BY verified, j_score",
        (SESSION,)
    ).fetchall()

    print(f"""
  Values in this session: {len(all_constraints)}

  VERIFIED (safe to ship):
""")
    for row in all_constraints:
        if row["verified"]:
            print(f"    ✓  {row['content'][:70]}")

    print(f"""
  STILL UNVERIFIED (annotated in code, blocked from silent ship):
""")
    for row in all_constraints:
        if not row["verified"]:
            src  = row["source"] or "user_stated"
            tier = "[stale]" if src == "temporal_scan" else "[unverified]"
            print(f"    ⚠  {tier} {row['content'][:65]}")

    _hr("═")
    print("""
  This is the product.

  No confidence scoring.
  No model self-assessment.
  No external API.

  Every value that flows from conversation into code
  is tracked and must be verified before it ships.

  Unknown = unverified. That's it.
""")
    _hr("═")


if __name__ == "__main__":
    try:
        run_demo()
    finally:
        try:
            os.unlink(_db.name)
        except Exception:
            pass
