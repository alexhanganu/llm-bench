"""
Quality benchmarks for llm-bench.

Implements:
  - MMLU  (Massive Multitask Language Understanding) – multiple-choice accuracy
  - HumanEval  – Python code generation pass@1
  - TruthfulQA – factual accuracy (MC1 variant)
"""

from __future__ import annotations

import logging
import re
import textwrap
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    model_id: str
    quantization: str
    benchmark: str          # "mmlu" | "humaneval" | "truthfulqa"
    score: float            # 0.0 – 1.0
    correct: int
    total: int
    elapsed_sec: float
    error: Optional[str] = None

    @property
    def pct(self) -> str:
        return f"{self.score * 100:.1f}%"

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(self).items()}


# ─────────────────────────────────────────────────────────────────────────────
# MMLU
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_mmlu(
    model: Any,
    tokenizer: Any,
    model_id: str = "unknown",
    quantization: str = "4bit",
    num_samples: int = 100,
    seed: int = 42,
) -> QualityResult:
    """
    Evaluate on MMLU (cais/mmlu, "all" split).
    Uses greedy decoding with a multiple-choice A/B/C/D prompt.
    """
    t0 = time.perf_counter()
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
    except Exception as exc:
        return _err_result(model_id, quantization, "mmlu", str(exc))

    correct = 0
    generate_fn = _build_generate_fn(model, tokenizer)

    for ex in ds:
        q = ex["question"]
        choices = ex["choices"]
        label = ex["answer"]   # int 0-3

        prompt = (
            f"Question: {q}\n"
            f"A) {choices[0]}\n"
            f"B) {choices[1]}\n"
            f"C) {choices[2]}\n"
            f"D) {choices[3]}\n"
            "Answer (A/B/C/D):"
        )

        pred = generate_fn(prompt, max_tokens=4).strip().upper()
        # Extract first A/B/C/D from response
        match = re.search(r"[ABCD]", pred)
        if match and ord(match.group()) - ord("A") == label:
            correct += 1

    total = len(ds)
    return QualityResult(
        model_id=model_id,
        quantization=quantization,
        benchmark="mmlu",
        score=correct / total if total else 0.0,
        correct=correct,
        total=total,
        elapsed_sec=round(time.perf_counter() - t0, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# HumanEval  (pass@1, greedy)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_humaneval(
    model: Any,
    tokenizer: Any,
    model_id: str = "unknown",
    quantization: str = "4bit",
    num_samples: int = 50,
    timeout_sec: float = 10.0,
) -> QualityResult:
    """
    Evaluate pass@1 on HumanEval (openai/openai_humaneval).
    Executes generated code in a sandboxed subprocess with a timeout.
    """
    t0 = time.perf_counter()
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        ds = ds.select(range(min(num_samples, len(ds))))
    except Exception as exc:
        return _err_result(model_id, quantization, "humaneval", str(exc))

    correct = 0
    generate_fn = _build_generate_fn(model, tokenizer)

    for ex in ds:
        prompt_code = ex["prompt"]
        test_code = ex["test"]
        entry_point = ex["entry_point"]

        # Ask the model to complete the function
        completion = generate_fn(
            prompt_code,
            max_tokens=256,
            stop=["def ", "\nclass ", "\n#"],
        )

        full_code = prompt_code + completion + "\n" + test_code + f"\ncheck({entry_point})\n"

        if _safe_exec(full_code, timeout=timeout_sec):
            correct += 1

    total = len(ds)
    return QualityResult(
        model_id=model_id,
        quantization=quantization,
        benchmark="humaneval",
        score=correct / total if total else 0.0,
        correct=correct,
        total=total,
        elapsed_sec=round(time.perf_counter() - t0, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TruthfulQA  (MC1 accuracy)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_truthfulqa(
    model: Any,
    tokenizer: Any,
    model_id: str = "unknown",
    quantization: str = "4bit",
    num_samples: int = 100,
    seed: int = 42,
) -> QualityResult:
    """
    Evaluate TruthfulQA MC1 – pick the single best answer from a list.
    """
    t0 = time.perf_counter()
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
    except Exception as exc:
        return _err_result(model_id, quantization, "truthfulqa", str(exc))

    correct = 0
    generate_fn = _build_generate_fn(model, tokenizer)

    for ex in ds:
        question = ex["question"]
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]  # 1 = correct, 0 = incorrect

        # Build numbered options
        option_lines = "\n".join(f"{i+1}) {c}" for i, c in enumerate(choices))
        prompt = (
            f"Question: {question}\n"
            f"Options:\n{option_lines}\n"
            "Answer with the number of the correct option:"
        )

        pred_text = generate_fn(prompt, max_tokens=4).strip()
        match = re.search(r"\d+", pred_text)
        if match:
            pred_idx = int(match.group()) - 1
            if 0 <= pred_idx < len(labels) and labels[pred_idx] == 1:
                correct += 1

    total = len(ds)
    return QualityResult(
        model_id=model_id,
        quantization=quantization,
        benchmark="truthfulqa",
        score=correct / total if total else 0.0,
        correct=correct,
        total=total,
        elapsed_sec=round(time.perf_counter() - t0, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unified runner
# ─────────────────────────────────────────────────────────────────────────────

def run_quality_suite(
    model: Any,
    tokenizer: Any,
    model_id: str,
    quantization: str = "4bit",
    benchmarks: List[str] = ("mmlu", "humaneval", "truthfulqa"),
    num_samples: int = 50,
) -> Dict[str, QualityResult]:
    """
    Run all requested quality benchmarks and return a dict keyed by benchmark name.
    """
    _bench_map: Dict[str, Callable] = {
        "mmlu": lambda: evaluate_mmlu(model, tokenizer, model_id, quantization, num_samples),
        "humaneval": lambda: evaluate_humaneval(model, tokenizer, model_id, quantization, num_samples),
        "truthfulqa": lambda: evaluate_truthfulqa(model, tokenizer, model_id, quantization, num_samples),
    }

    results: Dict[str, QualityResult] = {}
    for name in benchmarks:
        fn = _bench_map.get(name)
        if fn is None:
            logger.warning("Unknown benchmark: %s", name)
            continue
        logger.info("Running %s on %s …", name.upper(), model_id)
        results[name] = fn()
        logger.info("  → %s = %s", name, results[name].pct)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_generate_fn(model: Any, tokenizer: Any) -> Callable:
    """Return a unified generate(prompt, max_tokens, stop=[]) → str function."""
    cls = type(model).__name__

    if "Llama" in cls and "Causal" not in cls:
        # llama.cpp
        def _gen(prompt: str, max_tokens: int = 128, stop: List[str] = None) -> str:
            out = model(prompt, max_tokens=max_tokens, stop=stop or [], echo=False)
            return out["choices"][0]["text"]
        return _gen

    # HF transformers
    import torch

    def _gen(prompt: str, max_tokens: int = 128, stop: List[str] = None) -> str:
        device = next(model.parameters()).device
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0][enc["input_ids"].shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)

        # Honour stop sequences
        if stop:
            for s in stop:
                if s in text:
                    text = text[:text.index(s)]
        return text

    return _gen


def _safe_exec(code: str, timeout: float = 10.0) -> bool:
    """
    Execute Python code in a subprocess with a wall-clock timeout.
    Returns True if it exits with code 0, False otherwise.
    """
    import subprocess, sys, tempfile, os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name

    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def _err_result(model_id: str, quantization: str, bench: str, msg: str) -> QualityResult:
    return QualityResult(
        model_id=model_id,
        quantization=quantization,
        benchmark=bench,
        score=0.0,
        correct=0,
        total=0,
        elapsed_sec=0.0,
        error=msg,
    )
