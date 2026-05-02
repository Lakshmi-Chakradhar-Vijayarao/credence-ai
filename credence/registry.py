"""
credence/registry.py
================
CredenceRegistry: SQLite-backed persistent store for uncertain constraints.

Tracks which claims were stated with uncertainty, in which session, with what
confidence level, and whether they were later verified by the user.

Enables:
  - Cross-session constraint tracking (survives credence_save/credence_load cycles)
  - Verification write-back (MeMo-inspired: explicit confirmed value replaces uncertain one)
  - Contradiction detection (new claim vs. already-verified constraint)
  - Audit trail of unverified assumptions before shipping code

Zero new pip dependencies — uses sqlite3 and hashlib from stdlib.
Thread-safe for single-writer use (check_same_thread=False + SQLite row locking).
"""

import os
import re
import sqlite3
import hashlib
import threading
from datetime import datetime, timezone
from typing import Optional

# Short common words excluded from Jaccard similarity computation.
# Kept minimal — we want content words, not function words.
_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "have", "from",
    "are", "was", "were", "has", "had", "been", "can", "will",
    "not", "but", "all", "any", "its", "into", "over", "also",
    "than", "only", "such", "very", "more", "just", "you", "its",
    "may", "might", "should", "would", "could", "about", "what",
})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS constraints (
    constraint_id      TEXT PRIMARY KEY,
    content            TEXT NOT NULL,
    session_id         TEXT NOT NULL,
    j_score            REAL NOT NULL,
    zone               TEXT NOT NULL,
    verified           INTEGER NOT NULL DEFAULT 0,
    verified_value     TEXT,
    registered_at_turn INTEGER NOT NULL DEFAULT 0,
    source             TEXT NOT NULL DEFAULT 'user_stated',
    expires_at         TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session  ON constraints(session_id);
CREATE INDEX IF NOT EXISTS idx_verified ON constraints(verified);
CREATE INDEX IF NOT EXISTS idx_expires  ON constraints(expires_at);

CREATE TABLE IF NOT EXISTS constraint_events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    constraint_id TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    j_score       REAL,
    zone          TEXT,
    event_type    TEXT NOT NULL,
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_constraint ON constraint_events(constraint_id);
CREATE INDEX IF NOT EXISTS idx_events_type       ON constraint_events(event_type);
"""


class CredenceRegistry:
    """
    Persistent cross-session store for uncertain epistemic constraints.

    Usage:
        registry = CredenceRegistry()
        cid = registry.register("I think the rate limit is 100 req/min", "session-1")
        registry.verify(cid, "Confirmed: 100 req/min per vendor docs")
        pending = registry.list_uncertain("session-1")   # []

    db_path: path to SQLite file. Use ":memory:" for tests.
    """

    def __init__(self, db_path: str = "epistemic_registry.db"):
        self._db_path = db_path
        self._write_lock = threading.RLock()  # reentrant — register() holds it across nested calls
        self._conn    = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after initial schema without breaking existing DBs."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(constraints)")}
        if "registered_at_turn" not in cols:
            self._conn.execute(
                "ALTER TABLE constraints ADD COLUMN registered_at_turn INTEGER NOT NULL DEFAULT 0"
            )
        if "source" not in cols:
            self._conn.execute(
                "ALTER TABLE constraints ADD COLUMN source TEXT NOT NULL DEFAULT 'user_stated'"
            )
        if "expires_at" not in cols:
            self._conn.execute(
                "ALTER TABLE constraints ADD COLUMN expires_at TEXT"
            )
        if "validation_status" not in cols:
            self._conn.execute(
                "ALTER TABLE constraints ADD COLUMN validation_status TEXT NOT NULL DEFAULT 'unverified'"
            )
        if "contradicted_by" not in cols:
            self._conn.execute(
                "ALTER TABLE constraints ADD COLUMN contradicted_by TEXT"
            )
        if "project_id" not in cols:
            self._conn.execute(
                "ALTER TABLE constraints ADD COLUMN project_id TEXT"
            )
        if "is_memory" not in cols:
            self._conn.execute(
                "ALTER TABLE constraints ADD COLUMN is_memory INTEGER NOT NULL DEFAULT 0"
            )
        if "constraint_type" not in cols:
            self._conn.execute(
                "ALTER TABLE constraints ADD COLUMN constraint_type TEXT NOT NULL DEFAULT 'observation'"
            )
        if "verified_by" not in cols:
            self._conn.execute(
                "ALTER TABLE constraints ADD COLUMN verified_by TEXT"
            )
        if "verified_evidence" not in cols:
            self._conn.execute(
                "ALTER TABLE constraints ADD COLUMN verified_evidence TEXT"
            )
        # Create project index if it doesn't exist
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_project ON constraints(project_id)"
        )
        # marker_events table — passive flywheel data collection (Phase 1)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS marker_events (
                event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL,
                session_type  TEXT NOT NULL DEFAULT 'unknown',
                marker        TEXT NOT NULL,
                fired_at      TEXT NOT NULL,
                qual_survival REAL,
                fcr_outcome   INTEGER
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_marker ON marker_events(marker)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_marker_session ON marker_events(session_id)"
        )

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    # Source type constants — document where a constraint came from
    SOURCE_USER_STATED    = "user_stated"    # user explicitly said this, possibly with hedging
    SOURCE_AUTO_EXTRACTED = "auto_extracted" # Haiku claim extraction found it implicitly
    SOURCE_SCOUT          = "scout"          # Scout classifier detected it
    SOURCE_MODEL_INFERRED = "model_inferred" # model derived this, not directly user-stated

    # Constraint type constants — epistemic category of the claim.
    # Higher-risk types (vendor claims, compliance specs) decay slower;
    # low-stakes working hypotheses decay faster.
    CTYPE_OBSERVATION  = "observation"    # user's own measurement or direct observation
    CTYPE_VENDOR_CLAIM = "vendor_claim"   # value stated by a vendor / external party
    CTYPE_ESTIMATE     = "estimate"       # approximation or rough guess
    CTYPE_ASSUMPTION   = "assumption"     # working hypothesis or assumed default
    CTYPE_COMPLIANCE   = "compliance"     # regulatory / legal / policy constraint
    CTYPE_PERFORMANCE  = "performance"    # latency, throughput, resource measurements
    CTYPE_CONFIG       = "config"         # configuration value (could change with env)

    # Per-type decay rates (factor per turn, compared to base _DECAY_RATE=0.95).
    # Vendor claims decay more slowly — they don't expire, they just become stale.
    # Working assumptions decay faster — they should be confirmed quickly.
    _TYPE_DECAY_RATES: dict[str, float] = {
        "observation":  0.97,   # slow decay — user's own data
        "vendor_claim": 0.98,   # very slow — third-party stated values, high stakes
        "estimate":     0.93,   # faster — rough approximations
        "assumption":   0.90,   # fastest — working hypotheses should be resolved quickly
        "compliance":   0.99,   # almost no decay — regulatory constraints stay relevant
        "performance":  0.95,   # standard rate — measurements may drift
        "config":       0.94,   # slightly faster — config changes frequently
    }

    def register(
        self,
        content:          str,
        session_id:       str,
        j_score:          float = 0.30,
        zone:             str   = "LOW",
        turn_idx:         int   = 0,
        source:           str   = SOURCE_USER_STATED,
        ttl_turns:        Optional[int] = None,
        constraint_type:  str   = "observation",
    ) -> str:
        """
        Register an uncertain constraint. Returns constraint_id (12-char hash).

        Idempotent: registering the same content twice returns the existing ID
        without creating a duplicate row (INSERT OR IGNORE).

        turn_idx:         conversation turn at registration time — used for confidence decay.
        source:           provenance tag — user_stated / auto_extracted / scout / model_inferred.
        ttl_turns:        if set, constraint expires after this many turns from registration.
                          Verified constraints never expire regardless of TTL.
        constraint_type:  epistemic category — controls per-type decay rate.
                          One of: observation, vendor_claim, estimate, assumption,
                          compliance, performance, config.
        """
        j_score = max(0.0, min(1.0, float(j_score)))
        zone = zone if zone in ("HIGH", "MEDIUM", "LOW") else "MEDIUM"
        # Per-session cap: prevents unbounded DB growth from spammy agents.
        # INSERT OR IGNORE means idempotent re-registers don't count toward the cap.
        _MAX_PER_SESSION = int(os.environ.get("CREDENCE_MAX_CONSTRAINTS", "500"))
        existing_count = self._conn.execute(
            "SELECT COUNT(*) FROM constraints WHERE session_id=?", (session_id,)
        ).fetchone()[0]
        if existing_count >= _MAX_PER_SESSION:
            # Return a stable sentinel ID so callers don't blow up; just don't insert.
            return self._content_id(content)
        cid = self._content_id(content)
        now = self._now()
        expires_at = None
        if ttl_turns is not None:
            expires_at = str(turn_idx + ttl_turns)  # stored as turn number for simplicity
        ctype = constraint_type if constraint_type in self._TYPE_DECAY_RATES else "observation"
        with self._write_lock:  # RLock — holds across nested log_event/_dispute calls
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO constraints
                  (constraint_id, content, session_id, j_score, zone,
                   verified, verified_value, registered_at_turn, source, expires_at,
                   created_at, updated_at, constraint_type)
                VALUES (?, ?, ?, ?, ?, 0, NULL, ?, ?, ?, ?, ?, ?)
                """,
                (cid, content, session_id, round(j_score, 4), zone,
                 turn_idx, source, expires_at, now, now, ctype),
            )
            self._conn.commit()
            if cursor.rowcount > 0:
                # Only log on first insertion, not on idempotent re-register
                self.log_event(cid, "register", j_score=round(j_score, 4), zone=zone,
                               notes=f"session={session_id} turn={turn_idx}")
                # Verification drift: if new content conflicts with a verified constraint,
                # automatically mark that verified constraint as DISPUTED.
                conflict = self._check_conflict_with_verified(content, session_id)
                if conflict:
                    old_cid  = conflict["constraint_id"]
                    nums_new = sorted({
                        n for n in re.findall(r'\b(\d+(?:\.\d+)?)\b', content)
                        if len(n.replace(".", "")) >= 2
                    })
                    nums_old = sorted({
                        n for n in re.findall(r'\b(\d+(?:\.\d+)?)\b', conflict["content"])
                        if len(n.replace(".", "")) >= 2
                    })
                    new_vals_str = ", ".join(nums_new[:2]) or "new info"
                    reason = (
                        f"new registration (turn={turn_idx}) has value(s) {nums_new[:2]} "
                        f"vs verified value(s) {nums_old[:2]}"
                    )
                    self._dispute_constraint(old_cid, reason, new_vals_str)
        return cid

    # ------------------------------------------------------------------
    # Certainty Trajectory (event log)
    # ------------------------------------------------------------------

    def log_event(
        self,
        constraint_id: str,
        event_type:    str,
        j_score:       Optional[float] = None,
        zone:          Optional[str]   = None,
        notes:         Optional[str]   = None,
    ) -> None:
        """
        Append a timestamped event to the constraint's certainty trajectory.

        event_type conventions:
          register      — constraint first observed
          chat_update   — credence_chat referenced this constraint (j_score updated)
          scout         — Scout classifier auto-registered this constraint
          verify        — constraint confirmed by user
          contradict    — new claim contradicts this constraint
        """
        with self._write_lock:
            self._conn.execute(
                """
                INSERT INTO constraint_events
                  (constraint_id, timestamp, j_score, zone, event_type, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (constraint_id, self._now(), j_score, zone, event_type, notes),
            )
            self._conn.commit()

    def get_trajectory(self, constraint_id: str) -> list[dict]:
        """
        Return the full certainty trajectory for a constraint, oldest first.

        Each event: {event_id, constraint_id, timestamp, j_score, zone, event_type, notes}
        """
        rows = self._conn.execute(
            """
            SELECT * FROM constraint_events
             WHERE constraint_id=?
             ORDER BY event_id ASC
            """,
            (constraint_id,),
        ).fetchall()
        return [
            {
                "event_id":      row["event_id"],
                "constraint_id": row["constraint_id"],
                "timestamp":     row["timestamp"],
                "j_score":       row["j_score"],
                "zone":          row["zone"],
                "event_type":    row["event_type"],
                "notes":         row["notes"],
            }
            for row in rows
        ]

    def get_trajectories_for_session(self, session_id: str) -> dict[str, list[dict]]:
        """Return {constraint_id: [events]} for every constraint in a session."""
        constraints = self.get_all(session_id)
        return {c["constraint_id"]: self.get_trajectory(c["constraint_id"]) for c in constraints}

    def verify(
        self,
        constraint_id:  str,
        verified_value: str,
        evidence:       str = "",
        source:         str = "user",
    ) -> dict:
        """
        Mark a constraint as verified with its confirmed factual value.

        Args:
            constraint_id:  Constraint to verify.
            verified_value: The confirmed value (e.g. "100 req/min per vendor docs section 4.2").
            evidence:       What was checked to confirm this ("checked Stripe dashboard",
                            "confirmed in production logs", "vendor email 2026-05-02").
                            Empty string is accepted but strongly discouraged — an empty
                            evidence field means the audit trail shows no basis for the claim.
            source:         Who or what did the verification ("user", "api_response",
                            "agent:<name>", "test", "external_doc").

        Returns the updated row dict, or {"error": ...} if not found.
        """
        now = self._now()
        with self._write_lock:
            cursor = self._conn.execute(
                """
                UPDATE constraints
                   SET verified=1, verified_value=?, validation_status='verified',
                       verified_by=?, verified_evidence=?, updated_at=?
                 WHERE constraint_id=?
                """,
                (verified_value, source, evidence[:500] if evidence else "", now, constraint_id),
            )
            self._conn.commit()
        if cursor.rowcount == 0:
            return {"error": f"constraint_id '{constraint_id}' not found"}
        row = self._conn.execute(
            "SELECT * FROM constraints WHERE constraint_id=?", (constraint_id,)
        ).fetchone()
        audit_note = f"source={source} | evidence={evidence[:200] if evidence else '(none provided)'} | confirmed_value={verified_value[:100]}"
        self.log_event(constraint_id, "verify", notes=audit_note)
        return self._row_to_dict(row)

    def mark_contradiction(self, constraint_id: str, reason: str) -> None:
        """
        Public API: mark a constraint as DISPUTED because a newer message contradicts it.

        Called by the Contradiction Detector in context_manager._detect_contradiction()
        when Opus 4.7 identifies a conflicting value in a new user message. Demotes
        the constraint to disputed=True so it re-enters the enforcement pipeline.
        """
        self._dispute_constraint(constraint_id, reason, reason)

    def _dispute_constraint(
        self, constraint_id: str, reason: str, new_values_str: str
    ) -> None:
        """
        Mark a previously-verified constraint as DISPUTED.

        Fires when a new registration contradicts a verified constraint —
        different numeric values, same topic. The verified constraint is
        demoted to disputed (verified=0) and re-enters the enforcement pipeline:
        Truth Buffer, Consistency Enforcer, and GTS will treat it as unresolved.
        """
        now = self._now()
        with self._write_lock:
            self._conn.execute(
                """
                UPDATE constraints
                   SET validation_status='disputed', contradicted_by=?, verified=0, updated_at=?
                 WHERE constraint_id=?
                """,
                (new_values_str, now, constraint_id),
            )
            self._conn.commit()
        self.log_event(constraint_id, "contradict", notes=reason)

    def _check_conflict_with_verified(
        self, content: str, session_id: str
    ) -> Optional[dict]:
        """
        Check whether content conflicts with any verified constraint in this session.

        Conflict condition (both must hold):
          1. Topic overlap: Jaccard similarity of content words >= 0.15
          2. Numeric mismatch: new content has a ≥2-digit number not in the
             verified constraint's numeric set

        Returns the first conflicting verified constraint dict, or None.
        No API call. Pure text comparison.
        """
        nums_new = {
            n for n in re.findall(r'\b(\d+(?:\.\d+)?)\b', content)
            if len(n.replace(".", "")) >= 2
        }
        if not nums_new:
            return None  # no numeric value — can't detect conflict deterministically

        new_words = self._content_words(content)
        rows = self._conn.execute(
            "SELECT * FROM constraints WHERE session_id=? AND verified=1",
            (session_id,),
        ).fetchall()
        for row in rows:
            c = self._row_to_dict(row)
            if c.get("validation_status") == "disputed":
                continue  # already disputed — skip
            sim = self._jaccard(new_words, self._content_words(c["content"]))
            if sim < 0.15:
                continue
            nums_c = {
                n for n in re.findall(r'\b(\d+(?:\.\d+)?)\b', c["content"])
                if len(n.replace(".", "")) >= 2
            }
            if nums_c and not (nums_new & nums_c):
                return c  # topic overlap + different values → conflict
        return None

    def list_uncertain(self, session_id: str, current_turn: int = 0) -> list[dict]:
        """
        Return all unverified and DISPUTED constraints for a session.

        DISPUTED constraints appear first — they represent verified facts that
        were later contradicted and have the highest epistemic risk.
        """
        rows = self._conn.execute(
            """
            SELECT * FROM constraints
             WHERE session_id=? AND (verified=0 OR validation_status='disputed')
             ORDER BY
               CASE WHEN validation_status='disputed' THEN 0 ELSE 1 END,
               created_at DESC
            """,
            (session_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = self._row_to_dict(r)
            # Filter TTL-expired unverified constraints (verified ones never expire)
            exp = d.get("expires_at")
            if exp is not None:
                try:
                    if current_turn > int(exp):
                        continue
                except (ValueError, TypeError):
                    pass
            result.append(d)
        return result

    def get_all(self, session_id: str) -> list[dict]:
        """Return all constraints for a session (verified and unverified)."""
        rows = self._conn.execute(
            """
            SELECT * FROM constraints
             WHERE session_id=?
             ORDER BY created_at DESC
            """,
            (session_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_verified(self, session_id: str) -> list[dict]:
        """Return all verified constraints for a session, newest first."""
        rows = self._conn.execute(
            """
            SELECT * FROM constraints
             WHERE session_id=? AND verified=1
             ORDER BY updated_at DESC
            """,
            (session_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Similarity and contradiction
    # ------------------------------------------------------------------

    def query_similar(
        self,
        content:   str,
        threshold: float = 0.25,
    ) -> list[dict]:
        """
        Find all constraints whose content is similar to the query (Jaccard >= threshold).
        Returns list of dicts with added 'similarity' key, sorted descending.
        """
        query_words = self._content_words(content)
        if not query_words:
            return []
        rows    = self._conn.execute("SELECT * FROM constraints").fetchall()
        results = []
        for row in rows:
            sim = self._jaccard(query_words, self._content_words(row["content"]))
            if sim >= threshold:
                d = self._row_to_dict(row)
                d["similarity"] = round(sim, 4)
                results.append(d)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def check_contradiction(self, claim: str, session_id: str) -> list[dict]:
        """
        Find verified constraints in this session that are topically similar to claim.

        A verified constraint that is similar to a new claim is a potential
        contradiction — the new claim may conflict with what was already confirmed.
        Uses a lower threshold (0.20) than query_similar for broader recall.
        """
        claim_words = self._content_words(claim)
        if not claim_words:
            return []
        rows = self._conn.execute(
            "SELECT * FROM constraints WHERE session_id=? AND verified=1",
            (session_id,),
        ).fetchall()
        results = []
        for row in rows:
            sim = self._jaccard(claim_words, self._content_words(row["content"]))
            if sim >= 0.20:
                d = self._row_to_dict(row)
                d["similarity"] = round(sim, 4)
                results.append(d)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Auto-extraction (EG-2)
    # ------------------------------------------------------------------

    def extract_and_register_claims(
        self,
        turn_text:  str,
        session_id: str,
        turn_idx:   int,
        client,
        model:      str = "claude-haiku-4-5-20251001",
    ) -> list[str]:
        """
        Use Haiku to extract uncertain factual claims from a turn and auto-register them.

        Addresses the probe vocabulary gap: catches implicit uncertainty expressions
        ("vendor gave us ~50 req/min", "the estimate was $40K", "docs suggested 24h")
        that the keyword probe misses because they don't use canonical hedging phrases.

        Returns list of registered constraint_ids (may be empty for high-confidence turns).
        """
        import json as _json
        import re as _re

        prompt = (
            "Extract factual claims from this text as a JSON array. "
            "Include only claims with medium or low confidence — "
            "estimates, vendor-supplied values, assumptions, approximate figures, "
            "and anything said as 'probably', 'roughly', 'I think', 'we were told', "
            "'the docs say', 'reportedly', etc. "
            "Skip confirmed facts and general knowledge. "
            "Reply with ONLY valid JSON (no markdown), empty array [] if none found.\n\n"
            'Format: [{"claim": "...", "confidence": "low|medium", '
            '"type": "estimate|assumption|vendor_claim|approximation"}]\n\n'
            f"Text:\n{turn_text[:700]}"
        )
        try:
            resp = client.messages.create(
                model    = model,
                messages = [{"role": "user", "content": prompt}],
                max_tokens = 350,
            )
            raw = resp.content[0].text.strip() if resp.content else "[]"
            # Strip markdown fences if present
            raw = _re.sub(r"^```[a-z]*\s*", "", raw, flags=_re.MULTILINE).strip()
            raw = _re.sub(r"\s*```$", "", raw, flags=_re.MULTILINE).strip()
            items = _json.loads(raw)
            if not isinstance(items, list):
                return []
        except Exception:
            return []

        # Cache verified constraints for this session to avoid re-registering
        # confirmed values as new unverified ones (the CE4 failure mode).
        verified = self.list_verified(session_id)
        verified_words = [self._content_words(v["content"]) for v in verified]

        registered: list[str] = []
        for item in items:
            claim      = (item.get("claim") or "").strip()
            confidence = (item.get("confidence") or "high").lower()
            if not claim or confidence not in ("low", "medium"):
                continue

            # Skip if this claim overlaps significantly with an already-verified
            # constraint — extracting it again would undo verification.
            claim_words = self._content_words(claim)
            skip = False
            for vw in verified_words:
                if self._jaccard(claim_words, vw) >= 0.45:
                    skip = True
                    break
            if skip:
                continue

            j_score = 0.28 if confidence == "low" else 0.48
            zone    = "LOW"    if confidence == "low" else "MEDIUM"
            cid     = self.register(
                claim[:500], session_id,
                j_score=j_score, zone=zone, turn_idx=turn_idx,
                source=self.SOURCE_AUTO_EXTRACTED,
            )
            self.log_event(
                cid, "auto_extract",
                j_score=j_score, zone=zone,
                notes=f"turn={turn_idx} type={item.get('type','?')[:30]}",
            )
            registered.append(cid)
        return registered

    def get_relevant_claims(
        self,
        query:      str,
        session_id: str,
        max_claims: int = 5,
    ) -> list[dict]:
        """
        Return unverified claims relevant to the current user query.

        Uses content-word overlap as a relevance proxy. More targeted than injecting
        ALL unverified claims into the Truth Buffer — keeps the system prompt lean while
        ensuring the model sees the epistemic context that actually matters for this turn.
        """
        all_uncertain = self.list_uncertain(session_id)
        if not all_uncertain:
            return []

        query_words = self._content_words(query)
        if not query_words:
            return all_uncertain[:max_claims]

        scored: list[tuple[float, dict]] = []
        for c in all_uncertain:
            overlap = self._jaccard(query_words, self._content_words(c["content"]))
            scored.append((overlap, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Return topically related claims; fall back to all if none overlap
        relevant = [c for score, c in scored if score > 0.0]
        return (relevant if relevant else all_uncertain)[:max_claims]

    # ------------------------------------------------------------------
    # Confidence Decay
    # ------------------------------------------------------------------

    _DECAY_RATE = 0.95  # default per-turn decay factor (used when constraint_type unknown)

    def get_effective_confidence(self, constraint_id: str, current_turn: int) -> float:
        """
        Decayed confidence: j_score × decay_rate^(turns since registration).

        Decay rate is per-type (see _TYPE_DECAY_RATES). Vendor claims decay slowly
        (rate=0.98); working assumptions decay faster (rate=0.90). Verified constraints
        return their stored j_score unchanged (verification stops decay).

        Returns 0.0 if constraint_id not found.
        """
        row = self._conn.execute(
            "SELECT j_score, verified, registered_at_turn, constraint_type FROM constraints WHERE constraint_id=?",
            (constraint_id,),
        ).fetchone()
        if row is None:
            return 0.0
        if row["verified"]:
            return float(row["j_score"])
        ctype = row["constraint_type"] if row["constraint_type"] else "observation"
        decay = self._TYPE_DECAY_RATES.get(ctype, self._DECAY_RATE)
        turns_elapsed = max(0, current_turn - (row["registered_at_turn"] or 0))
        return round(float(row["j_score"]) * (decay ** turns_elapsed), 4)

    def update_confidence(
        self,
        constraint_id: str,
        new_j:         float,
        zone:          str,
        notes:         Optional[str] = None,
    ) -> None:
        """
        Update a constraint's j_score and zone in-place, and log a chat_update event.

        Called when new evidence in the conversation raises or lowers confidence
        on an existing unverified constraint (e.g. user says "actually I looked
        it up — it's definitely 100 req/min").
        """
        now = self._now()
        with self._write_lock:
            self._conn.execute(
                """
                UPDATE constraints
                   SET j_score=?, zone=?, updated_at=?
                 WHERE constraint_id=? AND verified=0
                """,
                (round(new_j, 4), zone, now, constraint_id),
            )
            self._conn.commit()
        self.log_event(
            constraint_id, "chat_update",
            j_score=round(new_j, 4), zone=zone,
            notes=notes,
        )

    def get_effective_uncertain(
        self,
        session_id:   str,
        current_turn: int,
        max_claims:   int = 10,
    ) -> list[dict]:
        """
        Return unverified constraints sorted by decayed confidence (lowest first).

        Claims that have been unverified for many turns bubble to the top —
        they are the most epistemically stale and most in need of verification.
        Adds 'effective_confidence' key to each dict.
        """
        all_uncertain = self.list_uncertain(session_id)
        for c in all_uncertain:
            c["effective_confidence"] = self.get_effective_confidence(
                c["constraint_id"], current_turn
            )
        # DISPUTED first, then by effective_confidence ascending (stalest first)
        all_uncertain.sort(key=lambda x: (
            0 if x.get("validation_status") == "disputed" else 1,
            x["effective_confidence"],
        ))
        return all_uncertain[:max_claims]

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> int:
        """Delete all constraints for a session. Returns number of rows deleted."""
        with self._write_lock:
            cursor = self._conn.execute(
                "DELETE FROM constraints WHERE session_id=?", (session_id,)
            )
            self._conn.commit()
        return cursor.rowcount

    def record_marker_events(
        self,
        session_id:    str,
        markers_fired: list[str],
        qual_survival: float,
        session_type:  str = "unknown",
    ) -> None:
        """
        Record which uncertainty markers fired and the resulting qual_survival.

        Called passively from credence_post_compress on every compression event.
        Data feeds the marker weight learning flywheel (Phase 3, activates at
        n_sessions >= 200). fcr_outcome is set when qual_survival < 0.50.
        """
        now = self._now()
        fcr = 1 if qual_survival < 0.50 else 0
        rows = [
            (session_id, session_type, m, now, qual_survival, fcr)
            for m in markers_fired
        ]
        with self._write_lock:
            self._conn.executemany(
                "INSERT INTO marker_events "
                "(session_id, session_type, marker, fired_at, qual_survival, fcr_outcome) "
                "VALUES (?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()

    def get_marker_stats(self) -> list[dict]:
        """
        Aggregate precision/recall per marker across all recorded sessions.
        Returns empty list when n_sessions < 10 (insufficient data).
        """
        n_sessions = self._conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM marker_events"
        ).fetchone()[0]

        if n_sessions < 10:
            return []

        rows = self._conn.execute("""
            SELECT marker,
                   COUNT(*)                        AS n_fires,
                   SUM(CASE WHEN fcr_outcome=1 THEN 1 ELSE 0 END) AS n_fcr,
                   AVG(qual_survival)              AS avg_qual_survival
            FROM marker_events
            GROUP BY marker
            ORDER BY n_fires DESC
        """).fetchall()

        total_fcr = self._conn.execute(
            "SELECT SUM(fcr_outcome) FROM marker_events"
        ).fetchone()[0] or 1

        result = []
        for row in rows:
            n_fires = row["n_fires"]
            n_fcr   = row["n_fcr"]
            prec    = n_fcr / n_fires if n_fires > 0 else 0.0
            rec     = n_fcr / total_fcr if total_fcr > 0 else 0.0
            f1      = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
            result.append({
                "marker":           row["marker"],
                "n_fires":          n_fires,
                "n_fcr":            n_fcr,
                "precision":        round(prec, 4),
                "recall":           round(rec, 4),
                "f1":               round(f1, 4),
                "avg_qual_survival": round(row["avg_qual_survival"] or 1.0, 4),
            })
        return result

    # ------------------------------------------------------------------
    # Phase 3 — Ghost constraint heuristics (active from day one, deterministic)
    # ------------------------------------------------------------------

    # Hedging markers whose absence in vendor_claim source_type indicates a
    # potential ghost constraint — an implicitly uncertain fact with no hedging.
    _GHOST_HEDGING_MARKERS: frozenset = frozenset({
        "i think", "i believe", "not sure", "not certain", "unclear", "unsure",
        "roughly", "approximately", "around", "about", "maybe", "perhaps",
        "possibly", "probably", "might be", "could be", "seems", "appears",
        "likely", "estimate", "estimated", "reportedly", "allegedly",
        "i was told", "we were told", "vendor said", "docs say", "docs suggest",
        "supposedly", "tentative", "provisional", "preliminary",
    })

    def flag_ghost_constraints(self, session_id: str) -> list[dict]:
        """
        Scan vendor_claim constraints in session for ghost constraint risk.

        A ghost constraint is a vendor-supplied fact registered without ANY hedging
        language — it looks certain but is actually unverified. These score HIGH-J
        because their language is assertive, so the faithfulness probe misses them.

        A constraint is flagged as a potential ghost when:
          1. constraint_type == 'vendor_claim'
          2. No hedging marker is present in the content text
          3. The constraint is unverified

        Returns list of flagged constraint dicts with 'ghost_risk' key added.
        Active from day one — no data threshold.
        """
        rows = self._conn.execute(
            """
            SELECT * FROM constraints
             WHERE session_id=? AND verified=0
             AND constraint_type='vendor_claim'
            """,
            (session_id,),
        ).fetchall()

        flagged = []
        for row in rows:
            c = self._row_to_dict(row)
            content_lower = c["content"].lower()
            has_hedge = any(m in content_lower for m in self._GHOST_HEDGING_MARKERS)
            if not has_hedge:
                c["ghost_risk"] = True
                c["ghost_reason"] = (
                    "vendor_claim source with no hedging language — "
                    "may be a ghost constraint (unverified fact presented as certain)"
                )
                flagged.append(c)
        return flagged

    # ------------------------------------------------------------------
    # Phase 3 — Marker weight learning (dormant until n_sessions >= 200)
    # ------------------------------------------------------------------

    _MARKER_LEARN_THRESHOLD = 200  # sessions required before weights update

    def update_marker_weights(self) -> dict:
        """
        Recompute marker weights from accumulated marker_events data.

        Markers with high FCR rate (they fire but compression still strips
        qualifiers) are down-weighted. Markers with low FCR rate (they fire
        and qualifiers survive) are up-weighted.

        Dormant when n_sessions < 200. Returns status dict.
        """
        n_sessions = self._conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM marker_events"
        ).fetchone()[0]

        if n_sessions < self._MARKER_LEARN_THRESHOLD:
            return {
                "status": "dormant",
                "n_sessions": n_sessions,
                "threshold": self._MARKER_LEARN_THRESHOLD,
                "message": (
                    f"Marker weight learning requires {self._MARKER_LEARN_THRESHOLD} sessions. "
                    f"Current: {n_sessions}. Collecting data passively."
                ),
            }

        stats = self.get_marker_stats()
        updated = []
        for s in stats:
            # Precision = fraction of fires that led to FCR (marker caught a real risk)
            # High precision → increase weight (marker is reliable)
            # Low precision  → decrease weight (marker is noisy)
            new_weight = round(0.5 + 0.5 * s["precision"], 4)
            updated.append({
                "marker":       s["marker"],
                "old_precision": s["precision"],
                "new_weight":   new_weight,
                "f1":           s["f1"],
            })

        return {
            "status":    "updated",
            "n_sessions": n_sessions,
            "markers_updated": len(updated),
            "top_reliable":   [m for m in sorted(updated, key=lambda x: x["new_weight"], reverse=True)][:5],
            "top_noisy":      [m for m in sorted(updated, key=lambda x: x["new_weight"])][:5],
        }

    # ------------------------------------------------------------------
    # Phase 3 — Thompson sampling bandit (dormant until n_sessions >= 100)
    # ------------------------------------------------------------------

    _BANDIT_THRESHOLD = 100  # sessions required before bandit activates

    # Bandit state stored in-memory (lost on restart — by design;
    # re-warms from marker_events on next sufficient data collection).
    _bandit_state: dict = {}

    def get_bandit_state(self) -> dict:
        """
        Return current Thompson sampling bandit state per session type.

        The bandit manages adaptive compression thresholds:
          - theta_high: when to compress (default 0.70)
          - theta_low:  when to preserve (default 0.45)

        Each (session_type, threshold) arm is a Beta(alpha, beta) distribution.
        - Success (qual_survival=1): alpha += 1
        - Failure (qual_survival=0): beta += 1

        Dormant when n_sessions < 100. Returns status dict with
        either the learned thresholds or a 'learning' message.
        """
        import math

        n_sessions = self._conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM marker_events"
        ).fetchone()[0]

        if n_sessions < self._BANDIT_THRESHOLD:
            return {
                "status": "learning",
                "n_sessions": n_sessions,
                "threshold": self._BANDIT_THRESHOLD,
                "message": (
                    f"Thompson sampling bandit requires {self._BANDIT_THRESHOLD} sessions. "
                    f"Current: {n_sessions}. Using static thresholds (high=0.70, low=0.45)."
                ),
                "current_thresholds": {
                    "theta_high": 0.70,
                    "theta_low":  0.45,
                },
            }

        # Aggregate qual_survival per session_type from marker_events
        rows = self._conn.execute("""
            SELECT session_type,
                   COUNT(*)                                   AS n_events,
                   SUM(CASE WHEN qual_survival >= 0.80 THEN 1 ELSE 0 END) AS n_success,
                   SUM(CASE WHEN qual_survival <  0.80 THEN 1 ELSE 0 END) AS n_fail,
                   AVG(qual_survival)                         AS avg_qual
            FROM marker_events
            WHERE qual_survival IS NOT NULL
            GROUP BY session_type
        """).fetchall()

        learned_thresholds = {}
        for row in rows:
            stype   = row["session_type"] or "general"
            n_succ  = row["n_success"]
            n_fail  = row["n_fail"]
            avg_q   = row["avg_qual"] or 0.5

            # Beta distribution mean = alpha / (alpha + beta)
            # Informed prior: alpha=2, beta=2 (mild compression-positive prior)
            alpha = 2 + n_succ
            beta  = 2 + n_fail
            mean  = alpha / (alpha + beta)
            # 95% CI width: approx 2 * sqrt(alpha*beta / (alpha+beta)^2 / (alpha+beta+1))
            var   = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            ci_half = 2 * math.sqrt(var)

            # Map learned mean → threshold adjustment
            # Higher qual_survival mean → lower theta_high (compress more aggressively)
            # Lower qual_survival mean → higher theta_high (compress more conservatively)
            theta_high = round(max(0.60, min(0.85, 0.70 + (0.5 - mean) * 0.30)), 4)
            theta_low  = round(max(0.30, min(0.55, theta_high - 0.25)), 4)

            learned_thresholds[stype] = {
                "theta_high":    theta_high,
                "theta_low":     theta_low,
                "beta_mean":     round(mean, 4),
                "ci_half":       round(ci_half, 4),
                "n_events":      row["n_events"],
                "avg_qual_survival": round(avg_q, 4),
            }

        return {
            "status":              "active",
            "n_sessions":          n_sessions,
            "learned_thresholds":  learned_thresholds,
            "default_thresholds":  {"theta_high": 0.70, "theta_low": 0.45},
            "message":             (
                "Bandit active. Thresholds adapted per session type from "
                f"{n_sessions} sessions of data."
            ),
        }

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        keys = row.keys()
        return {
            "constraint_id":      row["constraint_id"],
            "content":            row["content"],
            "session_id":         row["session_id"],
            "j_score":            row["j_score"],
            "zone":               row["zone"],
            "verified":           bool(row["verified"]),
            "verified_value":     row["verified_value"],
            "registered_at_turn": row["registered_at_turn"] if "registered_at_turn" in keys else 0,
            "source":             row["source"]             if "source"             in keys else "user_stated",
            "expires_at":         row["expires_at"]         if "expires_at"         in keys else None,
            "validation_status":  row["validation_status"]  if "validation_status"  in keys else "unverified",
            "contradicted_by":    row["contradicted_by"]    if "contradicted_by"    in keys else None,
            "created_at":         row["created_at"],
            "updated_at":         row["updated_at"],
            "constraint_type":    row["constraint_type"] if "constraint_type" in keys else "observation",
        }

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        """Jaccard similarity between two word sets."""
        if not a or not b:
            return 0.0
        union = len(a | b)
        return len(a & b) / union if union > 0 else 0.0

    @classmethod
    def _content_words(cls, text: str) -> set:
        """
        Normalize text to a set of content words for similarity comparison.
        Keeps words >= 3 chars, non-stopword, non-digit.
        """
        words = re.sub(r"[^\w\s]", " ", text.lower()).split()
        return {
            w for w in words
            if len(w) >= 3
            and not w.isdigit()
            and w not in _STOPWORDS
        }

    # ------------------------------------------------------------------
    # Cross-session memory
    # ------------------------------------------------------------------

    def snapshot_to_project(self, session_id: str, project_id: str) -> list[dict]:
        """
        Tag all unverified constraints from session_id as cross-session memories
        for project_id. Returns list of constraints snapshotted.

        Idempotent: calling twice won't duplicate — it just updates project_id + is_memory.
        """
        rows = self._conn.execute(
            """
            SELECT * FROM constraints
            WHERE session_id=? AND (verified=0 OR validation_status='disputed')
            """,
            (session_id,),
        ).fetchall()

        saved = []
        for row in rows:
            self._conn.execute(
                "UPDATE constraints SET project_id=?, is_memory=1, updated_at=? WHERE constraint_id=?",
                (project_id, self._now(), row["constraint_id"]),
            )
            saved.append(dict(row))
        self._conn.commit()
        return saved

    def recall_project_memories(self, project_id: str) -> list[dict]:
        """
        Return all unverified memory-tagged constraints for a project,
        sorted by j_score ascending (least certain first — most important to re-inject).

        Excludes cross_session_memory copies — those are injected into new sessions
        for Truth Buffer use, but should not appear in the canonical memory list.
        """
        rows = self._conn.execute(
            """
            SELECT * FROM constraints
            WHERE project_id=? AND is_memory=1
            AND source != 'cross_session_memory'
            AND (verified=0 OR validation_status='disputed')
            ORDER BY j_score ASC
            """,
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def inject_memories_into_session(
        self,
        project_id: str,
        new_session_id: str,
    ) -> list[str]:
        """
        Copy project memories into new_session_id so the new session's Truth Buffer
        picks them up immediately. Returns list of constraint_ids injected.

        Uses INSERT OR IGNORE — safe to call multiple times.
        """
        memories = self.recall_project_memories(project_id)
        injected = []
        for m in memories:
            new_cid = self._content_id(m["content"] + "::memory::" + new_session_id)
            now = self._now()
            self._conn.execute(
                """
                INSERT OR IGNORE INTO constraints
                  (constraint_id, content, session_id, j_score, zone,
                   verified, verified_value, registered_at_turn, source, expires_at,
                   created_at, updated_at, project_id, is_memory)
                VALUES (?, ?, ?, ?, ?, 0, NULL, 0, 'cross_session_memory', NULL, ?, ?, ?, 1)
                """,
                (new_cid, m["content"], new_session_id, m["j_score"], m["zone"],
                 now, now, project_id),
            )
            if self._conn.execute(
                "SELECT changes()"
            ).fetchone()[0] > 0:
                self.log_event(new_cid, "register", j_score=m["j_score"], zone=m["zone"],
                               notes=f"cross_session_memory from project={project_id}")
                injected.append(new_cid)
        self._conn.commit()
        return injected

    def propagate_verification(
        self,
        constraint_id: str,
        verified_value: str,
    ) -> int:
        """
        Verify a constraint AND back-propagate to the source cross-session original.

        When a cross_session_memory copy in a new session is verified, the original
        constraint in the source project should also be marked verified — otherwise
        epistemic debt reports show it as perpetually unresolved even after confirmation.

        Steps:
          1. Verify the given constraint_id (may be the copy OR the original).
          2. Find the original: a constraint with matching content and source != 'cross_session_memory'.
          3. Verify the original too.

        Returns number of rows updated (1 = only local, 2 = local + original).
        """
        updated = 0

        # Verify the local copy
        result = self.verify(constraint_id, verified_value)
        if "error" not in result:
            updated += 1
        else:
            return 0   # local not found — nothing to propagate

        # Look up the content to find the source original
        content = result.get("content", "")
        if not content:
            return updated

        # Find matching non-copy constraint(s) — same content, not a memory copy
        rows = self._conn.execute(
            """
            SELECT constraint_id FROM constraints
             WHERE content=?
               AND source != 'cross_session_memory'
               AND verified=0
            """,
            (content,),
        ).fetchall()

        for row in rows:
            orig_id = row["constraint_id"]
            if orig_id == constraint_id:
                continue   # already verified above
            src_result = self.verify(orig_id, verified_value)
            if "error" not in src_result:
                updated += 1

        return updated

    def get_all_project_constraints(self, project_id: str) -> list[dict]:
        """All constraints (verified + unverified) for a project, ordered by recency."""
        rows = self._conn.execute(
            "SELECT * FROM constraints WHERE project_id=? ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _content_id(content: str) -> str:
        """First 12 hex chars of SHA-256 hash of content — deterministic constraint ID."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _now() -> str:
        """ISO 8601 UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()
