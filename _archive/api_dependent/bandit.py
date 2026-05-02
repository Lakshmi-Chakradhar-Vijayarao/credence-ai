"""
bandit.py — Thompson Sampling bandit for compression threshold adaptation.

Learns optimal theta_high per session type without any API calls or GPU.
Updates after every compression event using a Beta distribution.

State: Beta(alpha, beta) per (session_type, theta_setting)
Reward: qualifier_survival — bool, measured after compression
Policy: Thompson sample → pick theta_setting with highest draw

Session types:
  debug        — uncertain hypotheses frequent, lower theta preferred
  design       — architecture stays uncertain longer, higher theta preferred
  research     — most conservative, very high theta
  code_review  — mixed, moderate theta
  general      — default
"""

import json
import math
import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


THETA_ARMS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
SESSION_TYPES = ["debug", "design", "research", "code_review", "general"]

# Informed priors from our compression_faithfulness study:
# design/research sessions should start conservative (higher theta preferred)
# debug sessions should start moderate
_PRIORS: dict[tuple, tuple] = {
    ("debug",       0.65): (3, 1),   # mild prior: 0.65 works well for debug
    ("debug",       0.70): (2, 1),
    ("design",      0.75): (3, 1),   # mild prior: 0.75 for design
    ("design",      0.80): (2, 1),
    ("research",    0.80): (3, 1),   # conservative for research
    ("research",    0.85): (2, 1),
    ("code_review", 0.70): (2, 1),
    ("general",     0.70): (2, 1),
}


@dataclass
class BanditState:
    session_type: str
    theta: float
    alpha: float
    beta: float
    n_trials: int = 0
    n_successes: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)


class EpistemicBandit:
    """
    Thompson Sampling bandit that learns optimal compression thresholds per session type.
    Persists state to SQLite. Thread-safe per session.

    Usage:
        bandit = EpistemicBandit(db_path="epistemic_bandit.db")
        theta = bandit.select_threshold("debug")
        # ... compress with theta ...
        bandit.update("debug", theta, qual_survived=True)
    """

    def __init__(self, db_path: str = "epistemic_bandit.db"):
        self._db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bandit_state (
                    session_type  TEXT NOT NULL,
                    theta         REAL NOT NULL,
                    alpha         REAL NOT NULL DEFAULT 1.0,
                    beta          REAL NOT NULL DEFAULT 1.0,
                    n_trials      INTEGER NOT NULL DEFAULT 0,
                    n_successes   INTEGER NOT NULL DEFAULT 0,
                    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (session_type, theta)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bandit_events (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_type  TEXT NOT NULL,
                    theta         REAL NOT NULL,
                    reward        INTEGER NOT NULL,
                    timestamp     TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            # Seed with informed priors
            for (stype, theta), (alpha, beta) in _PRIORS.items():
                conn.execute("""
                    INSERT OR IGNORE INTO bandit_state
                        (session_type, theta, alpha, beta)
                    VALUES (?, ?, ?, ?)
                """, (stype, theta, alpha, beta))
            conn.commit()

    def _get_state(self, session_type: str, theta: float) -> BanditState:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT alpha, beta, n_trials, n_successes FROM bandit_state "
                "WHERE session_type=? AND theta=?",
                (session_type, theta)
            ).fetchone()
        if row:
            return BanditState(session_type, theta, *row)
        return BanditState(session_type, theta, alpha=1.0, beta=1.0)

    def select_threshold(self, session_type: str) -> float:
        """
        Thompson sample: draw from Beta for each arm, return theta with highest draw.
        Falls back to 0.70 (calibrated default) if no data.
        """
        stype = session_type if session_type in SESSION_TYPES else "general"
        scores = {}
        for theta in THETA_ARMS:
            state = self._get_state(stype, theta)
            scores[theta] = state.sample
        return max(scores, key=scores.get)

    def update(self, session_type: str, theta: float, qual_survived: bool):
        """
        Update Beta distribution after observing a reward.
        qual_survived=True → qualifier was preserved → success (alpha += 1)
        qual_survived=False → qualifier was stripped → failure (beta += 1)
        """
        stype = session_type if session_type in SESSION_TYPES else "general"
        reward = 1 if qual_survived else 0
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT INTO bandit_state (session_type, theta, alpha, beta, n_trials, n_successes)
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(session_type, theta) DO UPDATE SET
                    alpha      = alpha + ?,
                    beta       = beta  + ?,
                    n_trials   = n_trials + 1,
                    n_successes = n_successes + ?,
                    updated_at = datetime('now')
            """, (stype, theta,
                  1.0 + reward, 1.0 + (1 - reward),  # INSERT values
                  reward,                              # n_successes for INSERT
                  reward, 1 - reward, reward))         # UPDATE deltas
            conn.execute(
                "INSERT INTO bandit_events (session_type, theta, reward) VALUES (?,?,?)",
                (stype, theta, reward)
            )
            conn.commit()

    def state_summary(self) -> list[dict]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT session_type, theta, alpha, beta, n_trials, n_successes "
                "FROM bandit_state ORDER BY session_type, theta"
            ).fetchall()
        return [
            {
                "session_type": r[0], "theta": r[1],
                "alpha": r[2], "beta": r[3],
                "n_trials": r[4], "n_successes": r[5],
                "mean_reward": round(r[2] / (r[2] + r[3]), 3),
            }
            for r in rows
        ]

    def optimal_theta(self, session_type: str) -> float:
        """Return the theta with the highest mean reward (exploitation mode)."""
        stype = session_type if session_type in SESSION_TYPES else "general"
        best_theta, best_mean = 0.70, 0.0
        for theta in THETA_ARMS:
            state = self._get_state(stype, theta)
            if state.mean > best_mean:
                best_mean = state.mean
                best_theta = theta
        return best_theta

    def convergence_report(self) -> dict:
        """
        How many trials needed for convergence per session type?
        Convergence: std(Beta) < 0.05 (tight distribution).
        """
        report = {}
        with sqlite3.connect(self._db_path) as conn:
            for stype in SESSION_TYPES:
                rows = conn.execute(
                    "SELECT theta, alpha, beta, n_trials FROM bandit_state "
                    "WHERE session_type=? ORDER BY n_trials DESC",
                    (stype,)
                ).fetchall()
                total_trials = sum(r[3] for r in rows)
                # Beta std = sqrt(alpha*beta / ((alpha+beta)^2 * (alpha+beta+1)))
                stds = []
                for _, a, b, _ in rows:
                    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
                    stds.append(math.sqrt(var))
                min_std = min(stds) if stds else 1.0
                report[stype] = {
                    "total_trials": total_trials,
                    "min_arm_std": round(min_std, 4),
                    "converged": min_std < 0.05,
                    "optimal_theta": self.optimal_theta(stype),
                }
        return report
