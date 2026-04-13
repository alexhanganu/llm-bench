"""
Tests for llm-bench core modules.

Run:  pytest tests/ -v
      pytest tests/ -v --cov=llm_bench
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry:

    def test_all_models_have_required_fields(self):
        from llm_bench.models.registry import MODELS

        required = {"name", "family", "developer", "params", "context",
                    "hf_repo", "min_vram_gb", "license"}
        for mid, meta in MODELS.items():
            missing = required - meta.keys()
            assert not missing, f"Model '{mid}' missing fields: {missing}"

    def test_min_vram_has_4bit(self):
        from llm_bench.models.registry import MODELS
        for mid, meta in MODELS.items():
            assert "4bit" in meta["min_vram_gb"], \
                f"Model '{mid}' missing 4bit VRAM entry"

    def test_get_model_known(self):
        from llm_bench.models.registry import get_model
        meta = get_model("llama-3.1-8b")
        assert meta["family"] == "llama"
        assert meta["params"] == "8B"

    def test_get_model_unknown_raises(self):
        from llm_bench.models.registry import get_model
        with pytest.raises(KeyError):
            get_model("nonexistent-model-xyz")

    def test_list_models_by_vram_filters_correctly(self):
        from llm_bench.models.registry import list_models_by_vram
        small = list_models_by_vram(max_vram_gb=4, quantization="4bit")
        for mid, meta in small.items():
            assert meta["min_vram_gb"]["4bit"] <= 4, \
                f"Model {mid} VRAM {meta['min_vram_gb']['4bit']} exceeds 4 GB filter"

    def test_list_models_by_vram_large(self):
        from llm_bench.models.registry import list_models_by_vram, MODELS
        all_models = list_models_by_vram(max_vram_gb=200, quantization="4bit")
        assert len(all_models) == len(MODELS)

    def test_recommend_for_hardware(self):
        from llm_bench.models.registry import recommend_for_hardware
        recs = recommend_for_hardware(vram_gb=24, ram_gb=64)
        assert len(recs) > 0
        for mid, meta in recs.items():
            assert "recommended_quant" in meta

    def test_get_families(self):
        from llm_bench.models.registry import get_families
        families = get_families()
        assert "llama" in families
        assert "qwen" in families
        assert "mistral" in families

    def test_at_least_20_models(self):
        from llm_bench.models.registry import MODELS
        assert len(MODELS) >= 20, f"Expected 20+ models, got {len(MODELS)}"

    def test_context_lengths_positive(self):
        from llm_bench.models.registry import MODELS
        for mid, meta in MODELS.items():
            assert meta["context"] > 0, f"Model {mid} has non-positive context length"

    def test_gguf_entries_consistent(self):
        """If gguf_repo is present, gguf_file must also be present."""
        from llm_bench.models.registry import MODELS
        for mid, meta in MODELS.items():
            has_repo = "gguf_repo" in meta
            has_file = "gguf_file" in meta
            assert has_repo == has_file, \
                f"Model {mid}: gguf_repo/gguf_file must both be present or absent"


# ─────────────────────────────────────────────────────────────────────────────
# Speed Benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestSpeedBenchmark:

    def _make_mock_hf_model(self, tokens_per_call: int = 10):
        """Build a minimal mock HF model."""
        import torch

        model = MagicMock()
        tokenizer = MagicMock()

        # Simulate input ids
        fake_input = {"input_ids": torch.ones(1, 5, dtype=torch.long)}
        tokenizer.return_value = fake_input

        # Simulate generated output
        fake_output = torch.ones(1, 5 + tokens_per_call, dtype=torch.long)
        model.generate.return_value = fake_output

        # Make device accessible
        param = MagicMock()
        param.device = torch.device("cpu")
        model.parameters.return_value = iter([param])

        tokenizer.decode.return_value = "some generated text here"
        tokenizer.eos_token_id = 2

        return model, tokenizer

    def test_speed_result_fields(self):
        from llm_bench.benchmarks.speed import SpeedResult
        sr = SpeedResult(
            model_id="test", quantization="4bit",
            prompt_tokens=10, generated_tokens=50,
            tokens_per_second=42.5, time_to_first_token_ms=35.2,
            memory_delta_gb=4.1, total_time_sec=1.2,
        )
        d = sr.to_dict()
        assert d["tokens_per_second"] == 42.5
        assert d["model_id"] == "test"
        assert sr.is_ok

    def test_speed_result_error(self):
        from llm_bench.benchmarks.speed import SpeedResult
        sr = SpeedResult(
            model_id="test", quantization="4bit",
            prompt_tokens=0, generated_tokens=0,
            tokens_per_second=0, time_to_first_token_ms=0,
            memory_delta_gb=0, total_time_sec=0,
            error="load failed",
        )
        assert not sr.is_ok

    def test_detect_backend_hf(self):
        from llm_bench.benchmarks.speed import _detect_backend
        model = MagicMock()
        model.__class__.__name__ = "LlamaForCausalLM"
        assert _detect_backend(model) == "transformers"

    def test_detect_backend_llamacpp(self):
        from llm_bench.benchmarks.speed import _detect_backend
        model = MagicMock()
        model.__class__.__name__ = "Llama"
        assert _detect_backend(model) == "llamacpp"


# ─────────────────────────────────────────────────────────────────────────────
# Memory Benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryBenchmark:

    def test_take_snapshot_returns_snapshot(self):
        pytest.importorskip("psutil")
        from llm_bench.benchmarks.memory import take_snapshot, MemorySnapshot
        snap = take_snapshot()
        assert isinstance(snap, MemorySnapshot)
        assert snap.ram_used_gb > 0
        assert snap.ram_available_gb > 0

    def test_memory_tracker_context_manager(self):
        pytest.importorskip("psutil")
        from llm_bench.benchmarks.memory import memory_tracker
        with memory_tracker("test_op") as mem:
            _ = list(range(100_000))
        assert "delta_vram_gb" in mem
        assert "delta_ram_gb" in mem
        assert mem["delta_ram_gb"] >= 0

    def test_memory_profile_properties(self):
        from llm_bench.benchmarks.memory import MemoryProfile, MemorySnapshot
        snap = lambda vram: MemorySnapshot(
            timestamp=0, vram_allocated_gb=vram, vram_reserved_gb=0,
            vram_peak_gb=vram, ram_used_gb=8, ram_available_gb=48,
        )
        profile = MemoryProfile(
            model_id="test", quantization="4bit",
            baseline=snap(0), after_load=snap(5.5), after_inference=snap(6.2),
        )
        assert profile.load_vram_gb == pytest.approx(5.5)
        assert profile.total_vram_gb == pytest.approx(6.2)
        d = profile.to_dict()
        assert "load_vram_gb" in d


# ─────────────────────────────────────────────────────────────────────────────
# Hardware Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestHardwareDetect:

    def test_detect_hardware_runs(self):
        pytest.importorskip("psutil")
        from llm_bench.utils.hardware_detect import detect_hardware, HardwareProfile
        hw = detect_hardware()
        assert isinstance(hw, HardwareProfile)
        assert hw.ram_total_gb > 0
        assert hw.cpu_cores >= 1

    def test_hardware_summary_non_empty(self):
        pytest.importorskip("psutil")
        from llm_bench.utils.hardware_detect import detect_hardware
        hw = detect_hardware()
        s = hw.summary()
        assert "CPU" in s
        assert "RAM" in s

    def test_hardware_to_dict(self):
        pytest.importorskip("psutil")
        from llm_bench.utils.hardware_detect import detect_hardware
        hw = detect_hardware()
        d = hw.to_dict()
        assert "cpu_name" in d
        assert "ram_total_gb" in d
        assert "has_gpu" in d


# ─────────────────────────────────────────────────────────────────────────────
# Results Database
# ─────────────────────────────────────────────────────────────────────────────

class TestResultsDB:

    @pytest.fixture
    def db(self, tmp_path):
        from llm_bench.results.database import ResultsDB
        return ResultsDB(db_path=tmp_path / "test.db")

    def test_upsert_and_read_speed(self, db):
        db.upsert_speed(
            model_id="llama-3.1-8b", quantization="4bit",
            backend="transformers",
            tokens_per_sec=87.4, ttft_ms=38.0, memory_gb=5.8,
            total_time_sec=2.9, hardware_tag="rtx3090",
        )
        rows = db.get_speed_results(hardware_tag="rtx3090")
        assert len(rows) == 1
        assert rows[0]["model_id"] == "llama-3.1-8b"
        assert rows[0]["tokens_per_sec"] == pytest.approx(87.4)

    def test_upsert_and_read_quality(self, db):
        db.upsert_quality(
            model_id="qwen2.5-7b", quantization="4bit",
            benchmark="mmlu", score=0.721,
            correct=72, total=100, elapsed_sec=45.2,
            hardware_tag="rtx3090",
        )
        rows = db.get_quality_results(benchmark="mmlu")
        assert len(rows) == 1
        assert rows[0]["score"] == pytest.approx(0.721)

    def test_upsert_deduplication(self, db):
        """Second upsert with same key should update, not duplicate."""
        for score in [0.50, 0.72]:
            db.upsert_quality(
                model_id="test-model", quantization="4bit",
                benchmark="mmlu", score=score,
                correct=int(score * 100), total=100, elapsed_sec=1.0,
                hardware_tag="default",
            )
        rows = db.get_quality_results(model_ids=["test-model"])
        assert len(rows) == 1
        assert rows[0]["score"] == pytest.approx(0.72)

    def test_list_benchmarked_models(self, db):
        db.upsert_speed(
            model_id="model-a", quantization="4bit", backend="transformers",
            tokens_per_sec=50, ttft_ms=30, memory_gb=5, total_time_sec=1,
        )
        db.upsert_speed(
            model_id="model-b", quantization="4bit", backend="transformers",
            tokens_per_sec=80, ttft_ms=25, memory_gb=4, total_time_sec=0.9,
        )
        models = db.list_benchmarked_models()
        assert "model-a" in models
        assert "model-b" in models

    def test_clear_model(self, db):
        db.upsert_speed(
            model_id="temp-model", quantization="4bit", backend="transformers",
            tokens_per_sec=50, ttft_ms=30, memory_gb=5, total_time_sec=1,
        )
        db.clear_model("temp-model")
        rows = db.get_speed_results(model_ids=["temp-model"])
        assert rows == []

    def test_merged_results(self, db):
        db.upsert_speed(
            model_id="merged-model", quantization="4bit", backend="transformers",
            tokens_per_sec=90, ttft_ms=35, memory_gb=5.5, total_time_sec=2.8,
            hardware_tag="test-hw",
        )
        db.upsert_quality(
            model_id="merged-model", quantization="4bit", benchmark="mmlu",
            score=0.68, correct=68, total=100, elapsed_sec=40,
            hardware_tag="test-hw",
        )
        merged = db.get_all_results_merged(hardware_tag="test-hw")
        assert len(merged) == 1
        row = merged[0]
        assert row["model_id"] == "merged-model"
        assert row["mmlu"] == pytest.approx(0.68)

    def test_hardware_profile_round_trip(self, db):
        profile = {"gpu": "RTX 3090", "vram_gb": 24, "ram_gb": 64}
        db.save_hardware_profile("rtx3090", profile)
        # No crash = pass (profile stored, no read API yet)


# ─────────────────────────────────────────────────────────────────────────────
# Long-context helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestLongContext:

    def test_build_prompt_depth_zero(self):
        from llm_bench.benchmarks.long_context import _build_prompt, _NEEDLE_TEMPLATE
        needle = _NEEDLE_TEMPLATE.format(passphrase="TESTKEY-0001")
        prompt = _build_prompt(tokenizer=None, target_tokens=512, needle=needle, depth=0.0)
        assert "TESTKEY-0001" in prompt

    def test_build_prompt_depth_one(self):
        from llm_bench.benchmarks.long_context import _build_prompt, _NEEDLE_TEMPLATE
        needle = _NEEDLE_TEMPLATE.format(passphrase="TESTKEY-9999")
        prompt = _build_prompt(tokenizer=None, target_tokens=512, needle=needle, depth=1.0)
        assert "TESTKEY-9999" in prompt

    def test_long_context_result_accuracy(self):
        from llm_bench.benchmarks.long_context import LongContextResult, NeedleResult
        lc = LongContextResult(
            model_id="test", quantization="4bit",
            context_lengths=[4096], depth_levels=[0.5],
        )
        lc.results = [
            NeedleResult(4096, 0.5, "KEY", "KEY found", True, 1.2),
            NeedleResult(4096, 0.5, "KEY", "wrong", False, 1.1),
        ]
        assert lc.overall_accuracy == pytest.approx(0.5)

    def test_results_to_heatmap(self):
        pytest.importorskip("pandas")
        from llm_bench.benchmarks.long_context import LongContextResult, NeedleResult, results_to_heatmap
        lc = LongContextResult(
            model_id="test", quantization="4bit",
            context_lengths=[4096, 8192], depth_levels=[0.25, 0.75],
        )
        lc.results = [
            NeedleResult(4096, 0.25, "K", "K", True, 1.0),
            NeedleResult(4096, 0.75, "K", "K", True, 1.0),
            NeedleResult(8192, 0.25, "K", "", False, 1.0),
            NeedleResult(8192, 0.75, "K", "K", True, 1.0),
        ]
        pivot = results_to_heatmap(lc)
        assert pivot is not None
        assert "4K" in pivot.columns or "8K" in pivot.columns


# ─────────────────────────────────────────────────────────────────────────────
# Downloader helpers (unit-level, no network)
# ─────────────────────────────────────────────────────────────────────────────

class TestDownloader:

    def test_get_download_size_known_model(self):
        from llm_bench.utils.download import get_download_size
        size = get_download_size("llama-3.1-8b", quantization="4bit")
        assert size is not None
        assert 4 < size < 10  # ~5.3 GB

    def test_get_download_size_unknown_quant(self):
        from llm_bench.utils.download import get_download_size
        size = get_download_size("llama-3.1-8b", quantization="nonexistent")
        assert size is None

    def test_check_model_cached_false_when_missing(self, tmp_path):
        """check_model_cached returns False when GGUF not on disk."""
        from llm_bench.utils import download as dl_module
        # Patch the default cache path
        original = dl_module._DEFAULT_CACHE
        dl_module._DEFAULT_CACHE = tmp_path
        try:
            cached = dl_module.check_model_cached("llama-3.1-8b", backend="gguf")
            assert cached is False
        finally:
            dl_module._DEFAULT_CACHE = original


# ─────────────────────────────────────────────────────────────────────────────
# Export module
# ─────────────────────────────────────────────────────────────────────────────

class TestExport:

    @pytest.fixture
    def sample_results(self):
        return [
            {"model_id": "model-a", "quantization": "4bit", "tokens_per_sec": 87.4, "mmlu": 0.68},
            {"model_id": "model-b", "quantization": "4bit", "tokens_per_sec": 91.3, "mmlu": 0.72},
        ]

    def test_export_json(self, tmp_path, sample_results):
        from llm_bench.results.export import export_json
        out = export_json(sample_results, tmp_path / "results.json")
        assert out.exists()
        import json
        loaded = json.loads(out.read_text())
        assert len(loaded) == 2
        assert loaded[0]["model_id"] == "model-a"

    def test_export_csv(self, tmp_path, sample_results):
        from llm_bench.results.export import export_csv
        out = export_csv(sample_results, tmp_path / "results.csv")
        assert out.exists()
        import csv
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[1]["model_id"] == "model-b"

    def test_export_csv_empty_raises(self, tmp_path):
        from llm_bench.results.export import export_csv
        with pytest.raises(ValueError):
            export_csv([], tmp_path / "empty.csv")

    def test_results_to_markdown_table(self, sample_results):
        from llm_bench.results.export import results_to_markdown_table
        md = results_to_markdown_table(sample_results, columns=["model_id", "tokens_per_sec"])
        assert "| model_id | tokens_per_sec |" in md
        assert "model-a" in md
        assert "---" in md

    def test_results_to_markdown_table_empty(self):
        from llm_bench.results.export import results_to_markdown_table
        result = results_to_markdown_table([])
        assert "_No results_" in result


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkCache
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkCache:

    @pytest.fixture
    def cache(self, tmp_path):
        from llm_bench.utils.cache import BenchmarkCache
        return BenchmarkCache(db_path=tmp_path / "cache.db")

    def _make_speed_result(self, model_id="test-model", quant="4bit"):
        """Return a minimal SpeedResult-like object."""
        from types import SimpleNamespace
        return SimpleNamespace(
            model_id=model_id,
            quantization=quant,
            backend="transformers",
            tokens_per_second=87.4,
            time_to_first_token_ms=38.0,
            memory_delta_gb=5.8,
            total_time_sec=2.9,
            prompt_tokens=12,
            generated_tokens=256,
        )

    def _make_quality_result(self, model_id="test-model", quant="4bit"):
        from types import SimpleNamespace
        return SimpleNamespace(
            model_id=model_id,
            quantization=quant,
            benchmark="mmlu",
            score=0.712,
            correct=71,
            total=100,
            elapsed_sec=45.2,
        )

    def test_has_speed_false_initially(self, cache):
        assert cache.has_speed("no-such-model", "4bit") is False

    def test_put_and_has_speed(self, cache):
        sr = self._make_speed_result()
        cache.put_speed(sr, hardware_tag="test-hw")
        assert cache.has_speed("test-model", "4bit", hardware_tag="test-hw")

    def test_get_speed_returns_row(self, cache):
        sr = self._make_speed_result()
        cache.put_speed(sr, hardware_tag="test-hw")
        row = cache.get_speed("test-model", "4bit", hardware_tag="test-hw")
        assert row is not None
        assert row["tokens_per_sec"] == pytest.approx(87.4)

    def test_has_quality_false_initially(self, cache):
        assert cache.has_quality("no-such-model", "4bit", "mmlu") is False

    def test_put_and_has_quality(self, cache):
        qr = self._make_quality_result()
        cache.put_quality(qr, hardware_tag="test-hw")
        assert cache.has_quality("test-model", "4bit", "mmlu", hardware_tag="test-hw")

    def test_summary(self, cache):
        sr = self._make_speed_result()
        cache.put_speed(sr)
        qr = self._make_quality_result()
        cache.put_quality(qr)
        s = cache.summary()
        assert s["models"] >= 1
        assert s["speed_rows"] >= 1
        assert s["quality_rows"] >= 1

    def test_different_hardware_tags_isolated(self, cache):
        sr = self._make_speed_result()
        cache.put_speed(sr, hardware_tag="gpu-a")
        assert cache.has_speed("test-model", "4bit", hardware_tag="gpu-a")
        assert not cache.has_speed("test-model", "4bit", hardware_tag="gpu-b")
