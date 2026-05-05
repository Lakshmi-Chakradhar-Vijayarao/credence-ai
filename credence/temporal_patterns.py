"""
credence/temporal_patterns.py
==============================
Temporal uncertainty pattern library — zero API, zero model calls.

Design principle: Unknown = unverified. No confidence scoring.
Three states only: unverified | stale | verified.

Two extraction passes:

  Pass 1 — Temporal (structurally stale):
    Values that were likely correct at training time but drift regardless
    of who wrote them. API date versions, semver strings, auth magic numbers
    when paired with auth-related variable names, rate limit literals.

  Pass 2 — Domain assignments (session-introduced, high-signal only):
    Variable names in the HIGH-SIGNAL set: rate limits, token lifetimes,
    API versions, pricing. NOT generic conventions (timeout, port, retries).
    Only flag what is clearly externally sourced, not what is a dev convention.

Session-origin contract (enforced by CLAUDE.md wiring, not code):
  These functions are called on GENERATED code only — code the agent just
  wrote in this session. They are not called on existing files being read
  or refactored. The caller (credence_self_probe) is responsible for this.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Temporal pattern definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalPattern:
    name:          str
    pattern:       re.Pattern
    category:      str
    stale_reason:  str
    verify_hint:   str
    # Variable name pattern: if set, only fire when variable name matches.
    # If None, fires on the value alone (e.g., API date strings are always stale).
    var_name_re:   re.Pattern | None = None


_PATTERNS: list[TemporalPattern] = [

    # API date versions — always stale regardless of variable name
    # e.g. "2023-10-16", "2024-01"
    TemporalPattern(
        name         = "api_date_version",
        pattern      = re.compile(r'["\'](\d{4}-\d{2}(?:-\d{2})?)["\']'),
        category     = "api_version",
        stale_reason = "API date versions release regularly; this string may be outdated",
        verify_hint  = "Check vendor changelog for current version",
        var_name_re  = None,  # always flag — date strings in code are structural stale
    ),

    # Semver strings — always stale
    # e.g. "3.11.2", "v2.1.0"
    TemporalPattern(
        name         = "semver",
        pattern      = re.compile(r'["\']v?(\d+\.\d+\.\d+(?:\.\d+)?)["\']'),
        category     = "semver",
        stale_reason = "Library version strings go stale as new releases land",
        verify_hint  = "Check PyPI/npm/crates.io for latest",
    ),

    # API path version segments — always stale
    # e.g. "https://api.stripe.com/v1/"
    TemporalPattern(
        name         = "api_path_version",
        pattern      = re.compile(r'["\'][^"\']*/(v\d+(?:-\d{4}-\d{2}-\d{2})?)/[^"\']*["\']'),
        category     = "api_version",
        stale_reason = "API path versions deprecate; old versions may be sunset",
        verify_hint  = "Verify current supported version in vendor API docs",
    ),

    # Auth lifetime magic numbers — ONLY when variable name suggests auth/session/token
    # 3600=1h, 7200=2h, 86400=1d, 604800=7d
    TemporalPattern(
        name         = "auth_lifetime_magic",
        pattern      = re.compile(
            r'\b(3600|7200|14400|21600|43200|86400|172800|604800|2592000|31536000)\b'
        ),
        category     = "auth_lifetime",
        stale_reason = "Token/session lifetime values are configurable per provider",
        verify_hint  = "Confirm against your auth provider's current token settings",
        var_name_re  = re.compile(
            r'token|expir|session|auth|ttl|lifetime|refresh|access',
            re.IGNORECASE
        ),
    ),

    # Rate limit inline literals
    # e.g. "100 req/min", "1000/hour"
    TemporalPattern(
        name         = "rate_limit_inline",
        pattern      = re.compile(
            r'\b(\d+)\s*(?:req|request|call|hit|rps|rpm)s?\s*/\s*(?:min(?:ute)?|hour|sec(?:ond)?|day)',
            re.IGNORECASE
        ),
        category     = "rate_limit",
        stale_reason = "Rate limits change with pricing tiers and vendor policy",
        verify_hint  = "Check your account dashboard or vendor docs for current limits",
    ),

    # Cost-per-unit pricing
    # e.g. $0.002 per 1k tokens
    TemporalPattern(
        name         = "pricing",
        pattern      = re.compile(
            r'\$(\d+(?:\.\d+)?)\s*(?:per|/)\s*\d+[kKmMbB]?',
            re.IGNORECASE
        ),
        category     = "pricing",
        stale_reason = "Pricing changes frequently; hardcoded costs become inaccurate",
        verify_hint  = "Verify at vendor pricing page; do not hardcode — read from config",
    ),
]


# J-scores for temporal pattern categories — lower = more stale risk.
# Kept here so mcp_server.py and the demo share a single source of truth.
TEMPORAL_J_SCORES: dict[str, float] = {
    "api_date_version":    0.18,
    "semver":              0.22,
    "api_path_version":    0.20,
    "auth_lifetime_magic": 0.25,
    "rate_limit_inline":   0.20,
    "pricing":             0.15,
}


# ---------------------------------------------------------------------------
# Domain assignment patterns — HIGH-SIGNAL ONLY
#
# Only variables that are clearly externally sourced (rates, lifetimes,
# API versions, pricing). NOT generic conventions (timeout, port, retries).
#
# Design rule: if a developer would reasonably write this value from memory
# as a convention (e.g., MAX_RETRIES=3, PORT=8080), do NOT flag it.
# Only flag values that require verification against an external source.
# ---------------------------------------------------------------------------

# (var_name_regex, domain, stale_reason)
_HIGH_SIGNAL_DOMAINS: list[tuple[re.Pattern, str, str]] = [

    (re.compile(r'rate.?limit|max.?req|req.?per|rpm|rps|tpm|throttle|quota', re.IGNORECASE),
     "rate_limit",
     "Rate limits are set by the vendor and vary by plan — must verify against your account"),

    (re.compile(
        r'token.?expir|token.?ttl|token.?life|token.?timeout|'
        r'access.?token.?ttl|refresh.?token.?ttl|session.?expir|'
        r'auth.?expir|expires.?in',
        re.IGNORECASE
    ),
     "auth_lifetime",
     "Token lifetimes are configurable per provider — confirm against current policy"),

    (re.compile(r'api.?version|stripe.?version|openai.?version|schema.?version|spec.?version',
                re.IGNORECASE),
     "api_version",
     "API versions deprecate; always verify you are using a currently supported version"),

    (re.compile(r'\bprice\b|cost.?per|rate.?per|unit.?price|billing.?rate|per.?token',
                re.IGNORECASE),
     "pricing",
     "Pricing changes frequently — hardcoded costs produce incorrect billing calculations"),
]


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class TemporalHit:
    value:              str
    pattern_name:       str
    category:           str
    stale_reason:       str
    verify_hint:        str
    line:               str
    source:             str = "temporal_scan"
    constraint_content: str = ""


@dataclass
class DomainHit:
    var_name:           str
    value:              str
    domain:             str
    stale_reason:       str
    line:               str
    source:             str = "self_probe"
    constraint_content: str = ""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def scan_temporal(text: str) -> list[TemporalHit]:
    """
    Scan code text for structurally stale values.

    Only inspects fenced code blocks if present; falls back to full text.
    For patterns with var_name_re set, only fires when the enclosing
    assignment line's variable name matches the pattern.
    """
    code_block_re = re.compile(r'```[^\n]*\n(.*?)```', re.DOTALL)
    blocks = code_block_re.findall(text)
    scan_target = "\n".join(blocks) if blocks else text

    hits:  list[TemporalHit] = []
    seen:  set[tuple[str, str]] = set()

    lines = scan_target.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue

        # Extract variable name from this line for context-aware patterns
        var_m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*[=:]', line)
        var_name = var_m.group(1) if var_m else ""

        for pat in _PATTERNS:
            # Context gate: if pattern requires a matching var name, check it
            if pat.var_name_re is not None and not pat.var_name_re.search(var_name):
                continue

            for m in pat.pattern.finditer(line):
                val = m.group(1) if m.lastindex else m.group(0)
                key = (pat.name, val)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(TemporalHit(
                    value              = val,
                    pattern_name       = pat.name,
                    category           = pat.category,
                    stale_reason       = pat.stale_reason,
                    verify_hint        = pat.verify_hint,
                    line               = line.rstrip(),
                    constraint_content = f"[stale:{pat.category}] {val} — {pat.stale_reason}",
                ))

    return hits


def scan_domain_assignments(text: str) -> list[DomainHit]:
    """
    Scan code for HIGH-SIGNAL domain assignments introduced in this session.

    HIGH-SIGNAL = externally sourced values the developer cannot know without
    checking: rate limits, token lifetimes, API versions, pricing.

    NOT flagged: generic conventions a developer would set from memory
    (timeout, port, retries, worker counts, page sizes).

    Session-origin contract: call this only on generated code, not on code
    being read from existing files. The CLAUDE.md wiring enforces this.
    """
    code_block_re = re.compile(r'```[^\n]*\n(.*?)```', re.DOTALL)
    blocks = code_block_re.findall(text)
    scan_target = "\n".join(blocks) if blocks else text

    assign_re = re.compile(
        r'^\s*([A-Z_][A-Z0-9_]*|[a-z_][a-z0-9_]*)\s*[=:]\s*([^\n#]+)',
        re.MULTILINE,
    )

    hits: list[DomainHit] = []
    seen: set[tuple[str, str]] = set()

    for m in assign_re.finditer(scan_target):
        var_name = m.group(1)
        raw_val  = m.group(2).strip().rstrip(",;")
        line     = m.group(0).rstrip()

        num_m = re.search(r'\b(\d+(?:\.\d+)?)\b', raw_val)
        str_m = re.search(r'["\']([^"\']+)["\']', raw_val)
        value = num_m.group(1) if num_m else (str_m.group(1) if str_m else raw_val[:40])

        for var_pat, domain, reason in _HIGH_SIGNAL_DOMAINS:
            if var_pat.search(var_name):
                key = (domain, var_name)
                if key in seen:
                    break
                seen.add(key)
                hits.append(DomainHit(
                    var_name           = var_name,
                    value              = value,
                    domain             = domain,
                    stale_reason       = reason,
                    line               = line,
                    constraint_content = f"[AI-generated:{domain}] {var_name} = {value} — {reason}",
                ))
                break

    return hits
