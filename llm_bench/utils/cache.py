"""
Lightweight result cache for llm-bench.

Wraps the SQLite ResultsDB with a simple get/set interface
so benchmark runners can skip re-running models that already
have fresh results stored.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

_DEFAULT_DB = Path("~/.cache/llm-bench/results.db").expanduser()
_STALE_DAYS = 30  # results older than this are considered stale


class BenchmarkCache:
    """
    Thin cache layer on top of ResultsDB.

    Usage
    -----
    cache = BenchmarkCache()
    if cache.has_speed("llama-3.1-8b", "4bit", "my-gpu"):
        row = cache.get_speed("llama-3.1-8b", "4bit", "my-gpu")
    else:
        # run benchmark …
        cache.put_speed(result, hardware_tag="my-gpu")
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        from llm_bench.results.database import ResultsDB
        self._db = ResultsDB(db_path=db_path)

    # ── Speed ──────────────────────────────────────────────────────────────

    def has_speed(self, model_id: str, quantization: str, hardware_tag: str = "default") -> bool:
        rows = self._db.get_speed_results(
            model_ids=[model_id], hardware_tag=hardware_tag
        )
        return any(r["quantization"] == quantization for r in rows)

    def get_speed(self, model_id: str, quantization: str,
                  hardware_tag: str = "default") -> Optional[Dict[str, Any]]:
        rows = self._db.get_speed_results(
            model_ids=[model_id], hardware_tag=hardware_tag
        )
        for r in rows:
            if r["quantization"] == quantization:
                return r
        return None

    def put_speed(self, result: Any, hardware_tag: str = "default") -> None:
        """Accept a SpeedResult dataclass and persist it."""
        self._db.upsert_speed(
            model_id=result.model_id,
            quantization=result.quantization,
            backend=result.backend,
            tokens_per_sec=result.tokens_per_second,
            ttft_ms=result.time_to_first_token_ms,
            memory_gb=result.memory_delta_gb,
            total_time_sec=result.total_time_sec,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            hardware_tag=hardware_tag,
        )

    # ── Quality ────────────────────────────────────────────────────────────

    def has_quality(self, model_id: str, quantization: str, benchmark: str,
                    hardware_tag: str = "default") -> bool:
        rows = self._db.get_quality_results(
            model_ids=[model_id], benchmark=benchmark, hardware_tag=hardware_tag
        )
        return any(r["quantization"] == quantization for r in rows)

    def put_quality(self, result: Any, hardware_tag: str = "default") -> None:
        """Accept a QualityResult dataclass and persist it."""
        self._db.upsert_quality(
            model_id=result.model_id,
            quantization=result.quantization,
            benchmark=result.benchmark,
            score=result.score,
            correct=result.correct,
            total=result.total,
            elapsed_sec=result.elapsed_sec,
            hardware_tag=hardware_tag,
        )

    # ── Convenience ────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, int]:
        """Return count of cached results by type."""
        models = self._db.list_benchmarked_models()
        speed  = self._db.get_speed_results()
        qual   = self._db.get_quality_results()
        return {
            "models": len(models),
            "speed_rows": len(speed),
            "quality_rows": len(qual),
        }
