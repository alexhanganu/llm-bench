"""
Memory profiling for llm-bench.

Tracks VRAM and RAM consumption during model loading and inference.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Single point-in-time memory reading."""
    timestamp: float
    vram_allocated_gb: float  # Currently allocated GPU memory
    vram_reserved_gb: float   # Reserved (includes fragmentation)
    vram_peak_gb: float       # Peak since last reset
    ram_used_gb: float        # Process RSS
    ram_available_gb: float   # System available RAM


@dataclass
class MemoryProfile:
    model_id: str
    quantization: str
    baseline: MemorySnapshot
    after_load: MemorySnapshot
    after_inference: MemorySnapshot

    @property
    def load_vram_gb(self) -> float:
        return max(0.0, self.after_load.vram_allocated_gb - self.baseline.vram_allocated_gb)

    @property
    def inference_vram_gb(self) -> float:
        return max(0.0, self.after_inference.vram_peak_gb - self.after_load.vram_allocated_gb)

    @property
    def total_vram_gb(self) -> float:
        return max(self.after_load.vram_allocated_gb, self.after_inference.vram_peak_gb)

    @property
    def load_ram_gb(self) -> float:
        return max(0.0, self.after_load.ram_used_gb - self.baseline.ram_used_gb)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "quantization": self.quantization,
            "load_vram_gb": round(self.load_vram_gb, 3),
            "inference_vram_gb": round(self.inference_vram_gb, 3),
            "total_vram_gb": round(self.total_vram_gb, 3),
            "load_ram_gb": round(self.load_ram_gb, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot helpers
# ─────────────────────────────────────────────────────────────────────────────

def take_snapshot() -> MemorySnapshot:
    """Capture current memory state."""
    import psutil

    proc = psutil.Process(os.getpid())
    ram_used = proc.memory_info().rss / 1e9
    ram_avail = psutil.virtual_memory().available / 1e9

    vram_alloc = vram_reserved = vram_peak = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated() / 1e9
            vram_reserved = torch.cuda.memory_reserved() / 1e9
            vram_peak = torch.cuda.max_memory_allocated() / 1e9
    except ImportError:
        pass

    return MemorySnapshot(
        timestamp=time.time(),
        vram_allocated_gb=vram_alloc,
        vram_reserved_gb=vram_reserved,
        vram_peak_gb=vram_peak,
        ram_used_gb=ram_used,
        ram_available_gb=ram_avail,
    )


def reset_peak_vram() -> None:
    """Reset the CUDA peak memory counter."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


@contextmanager
def memory_tracker(label: str = "") -> Generator[Dict[str, float], None, None]:
    """
    Context manager that yields a dict populated with before/after memory stats.

    Usage
    -----
    with memory_tracker("load model") as mem:
        model = load_big_model()
    print(mem["delta_vram_gb"])
    """
    reset_peak_vram()
    before = take_snapshot()
    result: Dict[str, float] = {}
    yield result
    after = take_snapshot()

    result["before_vram_gb"] = before.vram_allocated_gb
    result["after_vram_gb"] = after.vram_allocated_gb
    result["peak_vram_gb"] = after.vram_peak_gb
    result["delta_vram_gb"] = max(0.0, after.vram_allocated_gb - before.vram_allocated_gb)
    result["before_ram_gb"] = before.ram_used_gb
    result["after_ram_gb"] = after.ram_used_gb
    result["delta_ram_gb"] = max(0.0, after.ram_used_gb - before.ram_used_gb)

    if label:
        logger.debug(
            "[%s] ΔVRAM=%.2f GB  ΔRAM=%.2f GB  peak=%.2f GB",
            label,
            result["delta_vram_gb"],
            result["delta_ram_gb"],
            result["peak_vram_gb"],
        )


def profile_model(
    model: Any,
    tokenizer: Any,
    model_id: str,
    quantization: str,
    prompt: str = "Describe the concept of machine learning.",
    max_tokens: int = 64,
) -> MemoryProfile:
    """
    Build a MemoryProfile by running a short inference pass on an already-loaded
    model.  Call take_snapshot() *before* loading the model and pass the result
    as baseline.
    """
    baseline = take_snapshot()

    # Snapshot immediately after load context (model is already loaded here)
    after_load = take_snapshot()
    reset_peak_vram()

    # Short inference to measure generation footprint
    try:
        _run_inference(model, tokenizer, prompt, max_tokens)
    except Exception as exc:
        logger.warning("Inference for memory profile failed: %s", exc)

    after_inference = take_snapshot()

    return MemoryProfile(
        model_id=model_id,
        quantization=quantization,
        baseline=baseline,
        after_load=after_load,
        after_inference=after_inference,
    )


def _run_inference(model: Any, tokenizer: Any, prompt: str, max_tokens: int) -> None:
    """Fire a generation pass – supports both HF and llama.cpp."""
    cls = type(model).__name__
    if "Llama" in cls and "Causal" not in cls:
        # llama.cpp
        model(prompt, max_tokens=max_tokens, echo=False)
    else:
        # HF transformers
        import torch
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
