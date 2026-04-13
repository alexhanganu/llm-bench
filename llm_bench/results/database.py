"""
SQLite-backed results store for llm-bench.

Stores speed and quality benchmark results with deduplication.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("~/.cache/llm-bench/results.db").expanduser()

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS speed_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id        TEXT    NOT NULL,
    quantization    TEXT    NOT NULL,
    backend         TEXT    NOT NULL,
    tokens_per_sec  REAL,
    ttft_ms         REAL,
    memory_gb       REAL,
    total_time_sec  REAL,
    prompt_tokens   INTEGER,
    generated_tokens INTEGER,
    hardware_tag    TEXT,
    ts              DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_id, quantization, hardware_tag)
);

CREATE TABLE IF NOT EXISTS quality_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id     TEXT NOT NULL,
    quantization TEXT NOT NULL,
    benchmark    TEXT NOT NULL,
    score        REAL,
    correct      INTEGER,
    total        INTEGER,
    elapsed_sec  REAL,
    hardware_tag TEXT,
    ts           DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_id, quantization, benchmark, hardware_tag)
);

CREATE TABLE IF NOT EXISTS hardware_profiles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tag         TEXT UNIQUE NOT NULL,
    profile_json TEXT NOT NULL,
    ts          DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Database class
# ─────────────────────────────────────────────────────────────────────────────

class ResultsDB:
    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.path = Path(db_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Context manager ────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.executescript(_DDL)

    # ── Write ──────────────────────────────────────────────────────────────

    def upsert_speed(
        self,
        model_id: str,
        quantization: str,
        backend: str,
        tokens_per_sec: float,
        ttft_ms: float,
        memory_gb: float,
        total_time_sec: float,
        prompt_tokens: int = 0,
        generated_tokens: int = 0,
        hardware_tag: str = "default",
    ) -> None:
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO speed_results
                  (model_id, quantization, backend, tokens_per_sec, ttft_ms,
                   memory_gb, total_time_sec, prompt_tokens, generated_tokens, hardware_tag)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(model_id, quantization, hardware_tag)
                DO UPDATE SET
                  tokens_per_sec=excluded.tokens_per_sec,
                  ttft_ms=excluded.ttft_ms,
                  memory_gb=excluded.memory_gb,
                  total_time_sec=excluded.total_time_sec,
                  ts=CURRENT_TIMESTAMP
                """,
                (model_id, quantization, backend, tokens_per_sec, ttft_ms,
                 memory_gb, total_time_sec, prompt_tokens, generated_tokens, hardware_tag),
            )

    def upsert_quality(
        self,
        model_id: str,
        quantization: str,
        benchmark: str,
        score: float,
        correct: int,
        total: int,
        elapsed_sec: float,
        hardware_tag: str = "default",
    ) -> None:
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO quality_results
                  (model_id, quantization, benchmark, score, correct, total, elapsed_sec, hardware_tag)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(model_id, quantization, benchmark, hardware_tag)
                DO UPDATE SET
                  score=excluded.score,
                  correct=excluded.correct,
                  total=excluded.total,
                  elapsed_sec=excluded.elapsed_sec,
                  ts=CURRENT_TIMESTAMP
                """,
                (model_id, quantization, benchmark, score, correct, total, elapsed_sec, hardware_tag),
            )

    def save_hardware_profile(self, tag: str, profile_dict: Dict[str, Any]) -> None:
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO hardware_profiles (tag, profile_json)
                VALUES (?, ?)
                ON CONFLICT(tag) DO UPDATE SET profile_json=excluded.profile_json, ts=CURRENT_TIMESTAMP
                """,
                (tag, json.dumps(profile_dict)),
            )

    # ── Read ───────────────────────────────────────────────────────────────

    def get_speed_results(
        self,
        model_ids: Optional[List[str]] = None,
        hardware_tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM speed_results WHERE 1=1"
        params: list = []
        if model_ids:
            placeholders = ",".join("?" * len(model_ids))
            query += f" AND model_id IN ({placeholders})"
            params.extend(model_ids)
        if hardware_tag:
            query += " AND hardware_tag=?"
            params.append(hardware_tag)
        query += " ORDER BY tokens_per_sec DESC"

        with self._conn() as con:
            rows = con.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_quality_results(
        self,
        model_ids: Optional[List[str]] = None,
        benchmark: Optional[str] = None,
        hardware_tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM quality_results WHERE 1=1"
        params: list = []
        if model_ids:
            placeholders = ",".join("?" * len(model_ids))
            query += f" AND model_id IN ({placeholders})"
            params.extend(model_ids)
        if benchmark:
            query += " AND benchmark=?"
            params.append(benchmark)
        if hardware_tag:
            query += " AND hardware_tag=?"
            params.append(hardware_tag)
        query += " ORDER BY score DESC"

        with self._conn() as con:
            rows = con.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_all_results_merged(
        self,
        hardware_tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return one row per (model_id, quantization) with both speed and quality data.
        """
        speed = {
            (r["model_id"], r["quantization"]): r
            for r in self.get_speed_results(hardware_tag=hardware_tag)
        }
        quality = self.get_quality_results(hardware_tag=hardware_tag)

        # Pivot quality by model+quant
        quality_map: Dict[tuple, Dict[str, float]] = {}
        for r in quality:
            key = (r["model_id"], r["quantization"])
            quality_map.setdefault(key, {})[r["benchmark"]] = r["score"]

        all_keys = set(speed.keys()) | set(quality_map.keys())
        rows = []
        for key in all_keys:
            row: Dict[str, Any] = {}
            if key in speed:
                row.update(speed[key])
            else:
                row["model_id"], row["quantization"] = key
            row.update(quality_map.get(key, {}))
            rows.append(row)

        return rows

    def list_benchmarked_models(self) -> List[str]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT DISTINCT model_id FROM speed_results UNION "
                "SELECT DISTINCT model_id FROM quality_results ORDER BY model_id"
            ).fetchall()
        return [r["model_id"] for r in rows]

    def clear_model(self, model_id: str) -> None:
        with self._conn() as con:
            con.execute("DELETE FROM speed_results WHERE model_id=?", (model_id,))
            con.execute("DELETE FROM quality_results WHERE model_id=?", (model_id,))
        logger.info("Cleared results for %s", model_id)
