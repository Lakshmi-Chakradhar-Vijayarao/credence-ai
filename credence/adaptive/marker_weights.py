"""
marker_weights.py — Bayesian marker weight learning.

Tracks which of the 198 uncertainty markers actually predict FCR events.
Updates weights using online Bayesian learning. Over 100-200 sessions,
down-weights low-precision markers (false positives) and up-weights
high-precision markers (genuine uncertainty signals).

Metrics per marker:
  precision = P(FCR occurred | marker fired)
  recall    = P(marker fired | FCR occurred)
  f1        = 2 * precision * recall / (precision + recall)

Markers with f1 < DROP_THRESHOLD are candidates for pruning.
Markers with precision > BOOST_THRESHOLD get increased sensitivity.
"""

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from credence.context_manager import _UNCERTAINTY_MARKERS

DROP_THRESHOLD = 0.10    # f1 below this → low-signal marker, candidate for pruning
BOOST_THRESHOLD = 0.70   # precision above this → high-value marker
MIN_OBSERVATIONS = 5     # need at least N observations before updating weight


@dataclass
class MarkerStats:
    marker: str
    n_fires: int = 0
    n_fires_with_fcr: int = 0     # fires that co-occurred with actual FCR
    n_fcr_total: int = 0          # total FCR events in sessions where marker was active
    weight: float = 1.0           # current weight (1.0 = default, < 1.0 = down-weighted)

    @property
    def precision(self) -> float:
        if self.n_fires < MIN_OBSERVATIONS:
            return 0.5  # uniform prior
        return self.n_fires_with_fcr / max(self.n_fires, 1)

    @property
    def recall(self) -> float:
        if self.n_fcr_total < MIN_OBSERVATIONS:
            return 0.5
        return self.n_fires_with_fcr / max(self.n_fcr_total, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def is_low_signal(self) -> bool:
        return self.n_fires >= MIN_OBSERVATIONS and self.f1 < DROP_THRESHOLD

    @property
    def is_high_signal(self) -> bool:
        return self.n_fires >= MIN_OBSERVATIONS and self.precision >= BOOST_THRESHOLD


class MarkerWeightLearner:
    """
    Tracks marker firing rates and FCR co-occurrence to learn which markers
    are genuinely predictive vs. noise. No API calls, no GPU.

    Usage:
        learner = MarkerWeightLearner(db_path="marker_weights.db")

        # After each compression decision:
        fired = learner.get_fired_markers(text)
        learner.record_session(fired_markers=fired, fcr_occurred=True)

        # Get effective weights for probe decision:
        weight = learner.get_marker_weight("approximately")
    """

    def __init__(self, db_path: str = "marker_weights.db"):
        self._db_path = db_path
        self._init_db()
        self._markers = list(_UNCERTAINTY_MARKERS)

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS marker_stats (
                    marker          TEXT PRIMARY KEY,
                    n_fires         INTEGER NOT NULL DEFAULT 0,
                    n_fires_with_fcr INTEGER NOT NULL DEFAULT 0,
                    n_fcr_total     INTEGER NOT NULL DEFAULT 0,
                    weight          REAL NOT NULL DEFAULT 1.0,
                    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    fired_markers   TEXT NOT NULL,
                    fcr_occurred    INTEGER NOT NULL,
                    timestamp       TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            # Seed all markers with default weight
            for marker in _UNCERTAINTY_MARKERS:
                conn.execute(
                    "INSERT OR IGNORE INTO marker_stats (marker) VALUES (?)",
                    (marker,)
                )
            conn.commit()

    def get_fired_markers(self, text: str) -> list[str]:
        """Return list of markers from the 198 that fired on this text."""
        text_lower = text.lower()
        return [m for m in self._markers if m in text_lower]

    def record_session(self, fired_markers: list[str], fcr_occurred: bool):
        """
        Update marker stats after a compression event.
        fired_markers: which markers were present in the compressed text
        fcr_occurred: did the downstream answer show false certainty?
        """
        with sqlite3.connect(self._db_path) as conn:
            fcr_int = 1 if fcr_occurred else 0
            for marker in fired_markers:
                if fcr_occurred:
                    conn.execute("""
                        UPDATE marker_stats
                        SET n_fires = n_fires + 1,
                            n_fires_with_fcr = n_fires_with_fcr + 1,
                            n_fcr_total = n_fcr_total + 1,
                            updated_at = datetime('now')
                        WHERE marker = ?
                    """, (marker,))
                else:
                    conn.execute("""
                        UPDATE marker_stats
                        SET n_fires = n_fires + 1,
                            updated_at = datetime('now')
                        WHERE marker = ?
                    """, (marker,))
            # Increment n_fcr_total for all active markers when FCR occurred
            if fcr_occurred and fired_markers:
                placeholders = ",".join("?" * len(fired_markers))
                conn.execute(f"""
                    UPDATE marker_stats
                    SET n_fcr_total = n_fcr_total + 1
                    WHERE marker NOT IN ({placeholders})
                """, fired_markers)

            conn.execute(
                "INSERT INTO session_events (fired_markers, fcr_occurred) VALUES (?,?)",
                (json.dumps(fired_markers), fcr_int)
            )
            conn.commit()
        self._recompute_weights()

    def _recompute_weights(self):
        """Recompute weights for all markers that have sufficient observations."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT marker, n_fires, n_fires_with_fcr, n_fcr_total, weight "
                "FROM marker_stats"
            ).fetchall()
            updates = []
            for marker, n_fires, n_fcr_fires, n_fcr_total, current_weight in rows:
                stats = MarkerStats(
                    marker=marker,
                    n_fires=n_fires,
                    n_fires_with_fcr=n_fcr_fires,
                    n_fcr_total=n_fcr_total,
                    weight=current_weight,
                )
                new_weight = current_weight
                if stats.n_fires >= MIN_OBSERVATIONS:
                    if stats.is_high_signal:
                        # Boost: move weight toward 1.5 (capped)
                        new_weight = min(1.5, current_weight + 0.05 * (stats.precision - 0.5))
                    elif stats.is_low_signal:
                        # Down-weight: move toward 0.3 floor
                        new_weight = max(0.3, current_weight - 0.05 * (DROP_THRESHOLD - stats.f1))
                    else:
                        # Stable: slight regression toward 1.0
                        new_weight = current_weight + 0.01 * (1.0 - current_weight)
                if abs(new_weight - current_weight) > 0.001:
                    updates.append((round(new_weight, 4), marker))
            if updates:
                conn.executemany(
                    "UPDATE marker_stats SET weight=?, updated_at=datetime('now') WHERE marker=?",
                    updates
                )
                conn.commit()

    def get_marker_weight(self, marker: str) -> float:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT weight FROM marker_stats WHERE marker=?", (marker,)
            ).fetchone()
        return row[0] if row else 1.0

    def low_signal_markers(self) -> list[dict]:
        """Return markers with f1 < DROP_THRESHOLD and sufficient observations."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT marker, n_fires, n_fires_with_fcr, n_fcr_total, weight "
                "FROM marker_stats WHERE n_fires >= ? ORDER BY weight ASC",
                (MIN_OBSERVATIONS,)
            ).fetchall()
        result = []
        for marker, n_fires, n_fcr_fires, n_fcr_total, weight in rows:
            stats = MarkerStats(marker, n_fires, n_fcr_fires, n_fcr_total, weight)
            if stats.is_low_signal:
                result.append({
                    "marker": marker, "weight": weight,
                    "f1": round(stats.f1, 3), "precision": round(stats.precision, 3),
                    "recall": round(stats.recall, 3), "n_fires": n_fires,
                })
        return result

    def summary(self) -> dict:
        with sqlite3.connect(self._db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM marker_stats").fetchone()[0]
            observed = conn.execute(
                "SELECT COUNT(*) FROM marker_stats WHERE n_fires >= ?",
                (MIN_OBSERVATIONS,)
            ).fetchone()[0]
            low_signal = conn.execute(
                "SELECT COUNT(*) FROM marker_stats WHERE n_fires >= ? AND "
                "CAST(n_fires_with_fcr AS REAL)/MAX(n_fires,1) < ?",
                (MIN_OBSERVATIONS, DROP_THRESHOLD)
            ).fetchone()[0]
        return {
            "total_markers": total,
            "observed_markers": observed,
            "low_signal_candidates": low_signal,
            "effective_fpr_estimate": round(low_signal / max(total, 1), 3),
        }
