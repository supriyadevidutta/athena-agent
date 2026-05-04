"""
SQLite-backed store for the agent.

Five tables:
    backtest_runs       -- one row per backtest, queryable by the agent
    signal_evaluations  -- one row per signal/symbol/timestamp triplet
    memories            -- episodic + semantic memory the agent writes
    skill_meta          -- usage counters, last-modified, pinning
    sessions            -- conversation logs (Hermes pattern, lightweight here)

The agent queries these directly via tools. "Have I tested anything like this
before" → SQL on backtest_runs. "Did this signal fire last week" → SQL on
signal_evaluations. Without this, the agent forgets everything between turns.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from ..tools.research.runs import RunManifest


SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id          TEXT PRIMARY KEY,
    strategy        TEXT NOT NULL,
    engine          TEXT NOT NULL,
    universe        TEXT NOT NULL,    -- JSON array of symbols
    interval        TEXT NOT NULL,
    start_date      TEXT,
    end_date        TEXT,
    params_json     TEXT NOT NULL,
    cost_model_json TEXT NOT NULL,
    code_hash       TEXT,
    parent_run_id   TEXT,
    tags_json       TEXT,
    sharpe          REAL,
    sortino         REAL,
    calmar          REAL,
    cagr            REAL,
    max_dd          REAL,
    n_obs           INTEGER,
    n_trials        INTEGER,
    psr             REAL,
    dsr             REAL,
    sr_threshold    REAL,
    notes           TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_strategy  ON backtest_runs(strategy);
CREATE INDEX IF NOT EXISTS idx_runs_created   ON backtest_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_sharpe    ON backtest_runs(sharpe);

CREATE TABLE IF NOT EXISTS signal_evaluations (
    eval_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_name     TEXT NOT NULL,
    ts_utc          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    fired           INTEGER NOT NULL,    -- 0/1
    value           REAL,
    context_json    TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sig_name       ON signal_evaluations(signal_name);
CREATE INDEX IF NOT EXISTS idx_sig_symbol_ts  ON signal_evaluations(symbol, ts_utc);
CREATE INDEX IF NOT EXISTS idx_sig_fired      ON signal_evaluations(fired);

CREATE TABLE IF NOT EXISTS memories (
    memory_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL,       -- "episodic" | "semantic"
    text            TEXT NOT NULL,
    tags_json       TEXT,
    related_run_id  TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_kind   ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_mem_run    ON memories(related_run_id);

CREATE TABLE IF NOT EXISTS skill_meta (
    skill_name      TEXT PRIMARY KEY,
    use_count       INTEGER NOT NULL DEFAULT 0,
    last_used_at    TEXT,
    last_modified_at TEXT NOT NULL,
    pinned          INTEGER NOT NULL DEFAULT 0,
    archived        INTEGER NOT NULL DEFAULT 0,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    summary         TEXT
);
"""


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    # ---- Backtest runs --------------------------------------------------

    def record_run(self, manifest: RunManifest, stats: dict) -> None:
        d = stats.get("deflated", {}) or {}
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO backtest_runs (
                    run_id, strategy, engine, universe, interval,
                    start_date, end_date, params_json, cost_model_json,
                    code_hash, parent_run_id, tags_json,
                    sharpe, sortino, calmar, cagr, max_dd, n_obs,
                    n_trials, psr, dsr, sr_threshold, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                manifest.run_id, manifest.strategy, manifest.engine,
                json.dumps(manifest.universe), manifest.interval,
                manifest.start, manifest.end,
                json.dumps(manifest.params, default=str),
                json.dumps(manifest.cost_model, default=str),
                manifest.code_hash, manifest.parent_run_id,
                json.dumps(manifest.tags),
                stats.get("sharpe"), stats.get("sortino"),
                stats.get("calmar"), stats.get("cagr"),
                stats.get("max_dd"), stats.get("n_obs"),
                d.get("n_trials"), d.get("psr"), d.get("dsr"),
                d.get("sr_threshold"),
                manifest.notes, manifest.created_at,
            ))

    def find_runs(
        self,
        strategy: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        min_dsr: Optional[float] = None,
        tag: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        sql = "SELECT * FROM backtest_runs WHERE 1=1"
        args: list[Any] = []
        if strategy:
            sql += " AND strategy = ?"
            args.append(strategy)
        if min_sharpe is not None:
            sql += " AND sharpe >= ?"
            args.append(min_sharpe)
        if min_dsr is not None:
            sql += " AND dsr >= ?"
            args.append(min_dsr)
        if tag:
            sql += " AND tags_json LIKE ?"
            args.append(f"%{json.dumps(tag)[1:-1]}%")
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    # ---- Signal evaluations --------------------------------------------

    def record_eval(
        self,
        signal_name: str,
        ts_utc: datetime,
        symbol: str,
        fired: bool,
        value: Optional[float] = None,
        context: Optional[dict] = None,
    ) -> int:
        with self._conn() as c:
            cur = c.execute("""
                INSERT INTO signal_evaluations
                  (signal_name, ts_utc, symbol, fired, value, context_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_name, ts_utc.isoformat(), symbol,
                1 if fired else 0, value,
                json.dumps(context) if context else None,
                datetime.now(timezone.utc).isoformat(),
            ))
            return int(cur.lastrowid)

    def recent_evals(
        self,
        signal_name: Optional[str] = None,
        symbol: Optional[str] = None,
        fired_only: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        sql = "SELECT * FROM signal_evaluations WHERE 1=1"
        args: list[Any] = []
        if signal_name:
            sql += " AND signal_name = ?"
            args.append(signal_name)
        if symbol:
            sql += " AND symbol = ?"
            args.append(symbol)
        if fired_only:
            sql += " AND fired = 1"
        sql += " ORDER BY ts_utc DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def last_eval(self, signal_name: str, symbol: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("""
                SELECT * FROM signal_evaluations
                WHERE signal_name = ? AND symbol = ?
                ORDER BY ts_utc DESC LIMIT 1
            """, (signal_name, symbol)).fetchone()
            return dict(row) if row else None

    # ---- Memories -------------------------------------------------------

    def add_memory(
        self,
        text: str,
        kind: str = "episodic",
        tags: Optional[list[str]] = None,
        related_run_id: Optional[str] = None,
    ) -> int:
        if kind not in ("episodic", "semantic"):
            raise ValueError("kind must be 'episodic' or 'semantic'")
        with self._conn() as c:
            cur = c.execute("""
                INSERT INTO memories (kind, text, tags_json, related_run_id, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                kind, text, json.dumps(tags or []), related_run_id,
                datetime.now(timezone.utc).isoformat(),
            ))
            return int(cur.lastrowid)

    def search_memories(
        self,
        query: str,
        kind: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        # Plain LIKE for now; vector search arrives in Week 4 with knowledge tools.
        sql = "SELECT * FROM memories WHERE text LIKE ?"
        args: list[Any] = [f"%{query}%"]
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    # ---- Skill metadata -------------------------------------------------

    def touch_skill(self, name: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute("""
                INSERT INTO skill_meta (skill_name, use_count, last_used_at, last_modified_at)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(skill_name) DO UPDATE SET
                  use_count = use_count + 1,
                  last_used_at = excluded.last_used_at
            """, (name, now, now))

    def upsert_skill_meta(
        self,
        name: str,
        pinned: Optional[bool] = None,
        archived: Optional[bool] = None,
        notes: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute("""
                INSERT INTO skill_meta (skill_name, last_modified_at, pinned, archived, notes)
                VALUES (?, ?, COALESCE(?, 0), COALESCE(?, 0), ?)
                ON CONFLICT(skill_name) DO UPDATE SET
                  last_modified_at = excluded.last_modified_at,
                  pinned = COALESCE(?, skill_meta.pinned),
                  archived = COALESCE(?, skill_meta.archived),
                  notes = COALESCE(?, skill_meta.notes)
            """, (name, now,
                  int(pinned) if pinned is not None else None,
                  int(archived) if archived is not None else None,
                  notes,
                  int(pinned) if pinned is not None else None,
                  int(archived) if archived is not None else None,
                  notes))

    def list_skills(self, include_archived: bool = False) -> list[dict]:
        sql = "SELECT * FROM skill_meta"
        if not include_archived:
            sql += " WHERE archived = 0"
        sql += " ORDER BY use_count DESC"
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql).fetchall()]
