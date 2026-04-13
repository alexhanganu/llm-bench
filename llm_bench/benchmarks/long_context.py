"""
Long-context benchmark for llm-bench.

Implements the "Needle in a Haystack" test:
  1. Stuff a long filler document to a target context length
  2. Insert a "needle" (short factual statement) at varying depth positions
  3. Ask the model to retrieve the needle
  4. Score recall across (context_length × depth) grid

This reveals whether a model truly uses its claimed context window.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Filler text (Paul Graham essay excerpt, public domain style placeholder)
# Replace with a large public-domain corpus in production.
# ─────────────────────────────────────────────────────────────────────────────

_FILLER_PARAGRAPH = (
    "The most important thing about startups is that you need to make something people want. "
    "The way to find out if you've made something people want is to ship it and see. "
    "Many founders spend too long building the product before talking to users. "
    "The right approach is to do things that don't scale at first, then figure out how to scale them. "
    "Investors fund the team as much as the idea, because ideas change but execution ability persists. "
    "Growth is the most important metric in the early stage; if you're growing fast everything else works out. "
    "The hardest part of starting a company is recruiting good people who share your vision. "
    "You should be uncomfortably determined; most founders give up just before they would have succeeded. "
)

_NEEDLE_TEMPLATE = (
    "The secret passphrase for this document is: {passphrase}. "
    "Remember this and recall it when asked."
)

_QUESTION_TEMPLATE = "What is the secret passphrase mentioned in the document?"

_PASSPHRASES = [
    "ALPHA-TANGO-7749",
    "BLUE-FALCON-2031",
    "ZETA-PRIME-8842",
    "CRIMSON-TIDE-4417",
    "OMEGA-ZERO-1729",
    "SIERRA-NEVADA-5503",
    "DELTA-ECHO-9981",
    "VIOLET-STORM-3366",
]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NeedleResult:
    """Single needle retrieval result at one (context_len, depth) point."""
    context_tokens: int
    depth_pct: float       # 0.0 = start, 1.0 = end
    passphrase: str
    prediction: str
    correct: bool
    elapsed_sec: float


@dataclass
class LongContextResult:
    """Aggregate result across the full needle-in-haystack grid."""
    model_id: str
    quantization: str
    context_lengths: List[int]
    depth_levels: List[float]
    results: List[NeedleResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def overall_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.correct for r in self.results) / len(self.results)

    def accuracy_by_context(self) -> Dict[int, float]:
        acc: Dict[int, List[bool]] = {}
        for r in self.results:
            acc.setdefault(r.context_tokens, []).append(r.correct)
        return {k: sum(v) / len(v) for k, v in acc.items()}

    def accuracy_by_depth(self) -> Dict[float, float]:
        acc: Dict[float, List[bool]] = {}
        for r in self.results:
            acc.setdefault(r.depth_pct, []).append(r.correct)
        return {k: sum(v) / len(v) for k, v in acc.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "quantization": self.quantization,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "accuracy_by_context": {
                str(k): round(v, 4) for k, v in self.accuracy_by_context().items()
            },
            "accuracy_by_depth": {
                str(k): round(v, 4) for k, v in self.accuracy_by_depth().items()
            },
            "n_tests": len(self.results),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark function
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_long_context(
    model: Any,
    tokenizer: Any,
    model_id: str = "unknown",
    quantization: str = "4bit",
    context_lengths: List[int] = (4_096, 8_192, 16_384, 32_768),
    depth_levels: List[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    seed: int = 42,
) -> LongContextResult:
    """
    Run the needle-in-haystack test across a grid of context sizes and depths.

    Parameters
    ----------
    context_lengths : token counts to test (skipped if > model's max context)
    depth_levels    : 0.0 = needle at very start, 1.0 = very end
    """
    from llm_bench.models.registry import get_model

    try:
        meta = get_model(model_id)
        max_ctx = meta.get("context", 4096)
    except KeyError:
        max_ctx = 32_768

    rng = random.Random(seed)
    generate_fn = _build_generate_fn(model, tokenizer)
    result = LongContextResult(
        model_id=model_id,
        quantization=quantization,
        context_lengths=list(context_lengths),
        depth_levels=list(depth_levels),
    )

    for ctx_len in context_lengths:
        if ctx_len > max_ctx:
            logger.info("Skipping ctx_len=%d (model max=%d)", ctx_len, max_ctx)
            continue

        for depth in depth_levels:
            passphrase = rng.choice(_PASSPHRASES)
            needle = _NEEDLE_TEMPLATE.format(passphrase=passphrase)

            prompt = _build_prompt(
                tokenizer=tokenizer,
                target_tokens=ctx_len,
                needle=needle,
                depth=depth,
            )

            question = f"\n\n{_QUESTION_TEMPLATE}\nAnswer:"

            t0 = time.perf_counter()
            try:
                prediction = generate_fn(prompt + question, max_tokens=32)
            except Exception as exc:
                logger.warning("Generation failed at ctx=%d depth=%.2f: %s", ctx_len, depth, exc)
                prediction = ""
            elapsed = time.perf_counter() - t0

            correct = passphrase.lower() in prediction.lower()

            result.results.append(NeedleResult(
                context_tokens=ctx_len,
                depth_pct=depth,
                passphrase=passphrase,
                prediction=prediction.strip(),
                correct=correct,
                elapsed_sec=round(elapsed, 3),
            ))

            logger.debug(
                "ctx=%6d depth=%.2f  correct=%s  pred=%r",
                ctx_len, depth, correct, prediction[:60],
            )

    logger.info(
        "Long-context benchmark complete: %.1f%% overall (%d tests)",
        result.overall_accuracy * 100,
        len(result.results),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(
    tokenizer: Any,
    target_tokens: int,
    needle: str,
    depth: float,
    filler: str = _FILLER_PARAGRAPH,
) -> str:
    """
    Build a document of approximately `target_tokens` tokens with the needle
    inserted at fractional position `depth` (0=start, 1=end).
    """
    # Estimate tokens needed (rough: 1 token ≈ 4 chars for English)
    chars_needed = target_tokens * 4
    repeated = (filler * (chars_needed // len(filler) + 1))[:chars_needed]

    # Insert needle at depth
    insert_at = int(len(repeated) * depth)
    doc = repeated[:insert_at] + " " + needle + " " + repeated[insert_at:]

    return (
        "Read the following document carefully and answer the question at the end.\n\n"
        f"DOCUMENT:\n{doc}\n"
    )


def _build_generate_fn(model: Any, tokenizer: Any) -> Callable:
    """Unified generate(prompt, max_tokens, stop) → str."""
    cls = type(model).__name__

    if "Llama" in cls and "Causal" not in cls:
        def _gen(prompt: str, max_tokens: int = 64, stop=None) -> str:
            out = model(prompt, max_tokens=max_tokens, stop=stop or [], echo=False)
            return out["choices"][0]["text"]
        return _gen

    import torch

    def _gen(prompt: str, max_tokens: int = 64, stop=None) -> str:
        device = next(model.parameters()).device
        enc = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    return _gen


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap data for Streamlit / Plotly
# ─────────────────────────────────────────────────────────────────────────────

def results_to_heatmap(lc_result: LongContextResult):
    """
    Convert LongContextResult into a 2-D grid suitable for plotly heatmap.

    Returns (z, x_labels, y_labels) where:
      z[i][j] = accuracy at depth_levels[i], context_lengths[j]
    """
    import pandas as pd

    rows = []
    for r in lc_result.results:
        rows.append({
            "depth": f"{int(r.depth_pct*100)}%",
            "context_k": f"{r.context_tokens // 1024}K",
            "correct": int(r.correct),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index="depth", columns="context_k", values="correct", aggfunc="mean"
    )
    return pivot
