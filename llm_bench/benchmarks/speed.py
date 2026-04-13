"""
Speed benchmarking for llm-bench.

Measures:
  - tokens_per_second      (generation throughput)
  - time_to_first_token_ms (latency / TTFT)
  - memory_delta_gb        (VRAM consumed during generation)
  - total_time_sec         (wall-clock for the full run)

Supports both Hugging Face transformers models and llama.cpp (Llama) objects.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompts used for speed tests
# ─────────────────────────────────────────────────────────────────────────────

_SPEED_PROMPTS: List[str] = [
    "Write a detailed essay about the history of artificial intelligence and its impact on society.",
    "Explain the differences between supervised, unsupervised, and reinforcement learning with examples.",
    "Describe the architecture of a transformer neural network step by step.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpeedResult:
    model_id: str
    quantization: str
    prompt_tokens: int
    generated_tokens: int
    tokens_per_second: float
    time_to_first_token_ms: float
    memory_delta_gb: float
    total_time_sec: float
    backend: str = "transformers"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(self).items()}

    @property
    def is_ok(self) -> bool:
        return self.error is None


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark function
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_speed(
    model: Any,
    tokenizer: Any,
    model_id: str = "unknown",
    quantization: str = "4bit",
    num_tokens: int = 256,
    num_warmup_tokens: int = 16,
    num_runs: int = 1,
    prompt: Optional[str] = None,
) -> SpeedResult:
    """
    Measure generation speed for a loaded model.

    Parameters
    ----------
    model       : HF AutoModelForCausalLM OR llama_cpp.Llama
    tokenizer   : HF tokenizer (or None for llama.cpp)
    model_id    : name used in the result
    quantization: label for the result
    num_tokens  : number of tokens to generate during the timed run
    num_warmup_tokens: tokens generated for GPU warmup (not timed)
    num_runs    : average over this many timed runs
    prompt      : override the default prompt
    """
    # Detect backend
    backend = _detect_backend(model)

    if backend == "llamacpp":
        return _bench_llamacpp(model, model_id, quantization, num_tokens, num_warmup_tokens, prompt)
    else:
        return _bench_transformers(
            model, tokenizer, model_id, quantization,
            num_tokens, num_warmup_tokens, num_runs, prompt
        )


# ─────────────────────────────────────────────────────────────────────────────
# Transformers backend
# ─────────────────────────────────────────────────────────────────────────────

def _bench_transformers(
    model: Any,
    tokenizer: Any,
    model_id: str,
    quantization: str,
    num_tokens: int,
    num_warmup_tokens: int,
    num_runs: int,
    prompt: Optional[str],
) -> SpeedResult:
    try:
        import torch
    except ImportError as exc:
        return _err_result(model_id, quantization, "transformers", str(exc))

    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"

    text = prompt or _SPEED_PROMPTS[0]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]

    # ── Warmup ─────────────────────────────────────────────────────────────
    try:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=num_warmup_tokens, do_sample=False)
        if use_cuda:
            torch.cuda.synchronize()
    except Exception as exc:
        logger.warning("Warmup failed: %s", exc)

    # ── Time to first token ────────────────────────────────────────────────
    mem_before = _gpu_allocated_gb() if use_cuda else 0.0

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)

    if use_cuda:
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    # ── Throughput ─────────────────────────────────────────────────────────
    total_tokens = 0
    total_elapsed = 0.0

    for _ in range(num_runs):
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t1

        generated = out.shape[1] - prompt_tokens
        total_tokens += generated
        total_elapsed += elapsed

    tokens_per_sec = total_tokens / total_elapsed if total_elapsed > 0 else 0.0

    mem_after = _gpu_allocated_gb() if use_cuda else 0.0
    mem_delta = max(0.0, mem_after - mem_before)

    if use_cuda:
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        mem_delta = max(mem_delta, peak_gb - mem_before)

    return SpeedResult(
        model_id=model_id,
        quantization=quantization,
        prompt_tokens=prompt_tokens,
        generated_tokens=total_tokens // num_runs,
        tokens_per_second=round(tokens_per_sec, 2),
        time_to_first_token_ms=round(ttft_ms, 2),
        memory_delta_gb=round(mem_delta, 3),
        total_time_sec=round(total_elapsed / num_runs, 3),
        backend="transformers",
    )


# ─────────────────────────────────────────────────────────────────────────────
# llama.cpp backend
# ─────────────────────────────────────────────────────────────────────────────

def _bench_llamacpp(
    model: Any,
    model_id: str,
    quantization: str,
    num_tokens: int,
    num_warmup_tokens: int,
    prompt: Optional[str],
) -> SpeedResult:
    text = prompt or _SPEED_PROMPTS[0]

    # Warmup
    try:
        _ = model(text, max_tokens=num_warmup_tokens, echo=False)
    except Exception as exc:
        logger.warning("llama.cpp warmup failed: %s", exc)

    # TTFT
    t0 = time.perf_counter()
    _ = model(text, max_tokens=1, echo=False)
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    # Throughput
    t1 = time.perf_counter()
    output = model(text, max_tokens=num_tokens, echo=False)
    elapsed = time.perf_counter() - t1

    generated = output["usage"]["completion_tokens"]
    prompt_tokens = output["usage"]["prompt_tokens"]
    tokens_per_sec = generated / elapsed if elapsed > 0 else 0.0

    return SpeedResult(
        model_id=model_id,
        quantization=quantization,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated,
        tokens_per_second=round(tokens_per_sec, 2),
        time_to_first_token_ms=round(ttft_ms, 2),
        memory_delta_gb=0.0,  # llama.cpp doesn't expose this easily
        total_time_sec=round(elapsed, 3),
        backend="llamacpp",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch / compare helpers
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_speed_batch(
    models: List[Dict[str, Any]],
    num_tokens: int = 256,
    verbose: bool = True,
) -> List[SpeedResult]:
    """
    Benchmark a list of loaded models in sequence.

    Each element of `models` must be:
        {"model_id": str, "quantization": str, "model": <model>, "tokenizer": <tokenizer>}
    """
    results = []
    for entry in models:
        if verbose:
            logger.info("Benchmarking speed: %s (%s)", entry["model_id"], entry["quantization"])
        r = benchmark_speed(
            model=entry["model"],
            tokenizer=entry.get("tokenizer"),
            model_id=entry["model_id"],
            quantization=entry.get("quantization", "unknown"),
            num_tokens=num_tokens,
        )
        results.append(r)
        if verbose:
            logger.info(
                "  → %.1f tok/s | TTFT %.0f ms | Δmem %.2f GB",
                r.tokens_per_second, r.time_to_first_token_ms, r.memory_delta_gb,
            )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_backend(model: Any) -> str:
    cls = type(model).__name__
    if "Llama" in cls and "Causal" not in cls:
        return "llamacpp"
    return "transformers"


def _gpu_allocated_gb() -> float:
    try:
        import torch
        return torch.cuda.memory_allocated() / 1e9
    except Exception:
        return 0.0


def _err_result(model_id: str, quantization: str, backend: str, msg: str) -> SpeedResult:
    return SpeedResult(
        model_id=model_id,
        quantization=quantization,
        prompt_tokens=0,
        generated_tokens=0,
        tokens_per_second=0.0,
        time_to_first_token_ms=0.0,
        memory_delta_gb=0.0,
        total_time_sec=0.0,
        backend=backend,
        error=msg,
    )
