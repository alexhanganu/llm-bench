#!/usr/bin/env python3
"""
scripts/precompute.py
---------------------
Run all benchmarks on the current machine and write results to
data/precomputed/<hardware_tag>.json.

Usage:
    python scripts/precompute.py --hardware rtx3090 --models llama-3.1-8b qwen2.5-7b
    python scripts/precompute.py --hardware rtx3090 --all-small   # all ≤8GB VRAM models
    python scripts/precompute.py --hardware rtx3090 --dry-run     # validate config only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-compute llm-bench results")
    p.add_argument("--hardware", required=True,
                   help="Hardware tag written to the output file (e.g. rtx3090)")
    p.add_argument("--models", nargs="*", default=None,
                   help="Specific model IDs to benchmark")
    p.add_argument("--all-small", action="store_true",
                   help="Benchmark all models fitting in ≤8 GB VRAM at 4-bit")
    p.add_argument("--quant", default="4bit",
                   choices=["fp16", "8bit", "4bit"],
                   help="Quantization level")
    p.add_argument("--benchmarks", nargs="*",
                   default=["speed", "mmlu"],
                   choices=["speed", "mmlu", "humaneval", "truthfulqa"],
                   help="Which benchmarks to run")
    p.add_argument("--mmlu-samples", type=int, default=100)
    p.add_argument("--speed-tokens", type=int, default=256)
    p.add_argument("--output-dir", default=str(ROOT / "data" / "precomputed"),
                   help="Directory to write results JSON")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate and print plan without running benchmarks")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from llm_bench.models.registry import MODELS, list_models_by_vram

    # Determine model list
    if args.models:
        model_ids = args.models
        for mid in model_ids:
            if mid not in MODELS:
                logger.error("Unknown model: %s", mid)
                sys.exit(1)
    elif args.all_small:
        model_ids = list(list_models_by_vram(max_vram_gb=8, quantization="4bit").keys())
        logger.info("Auto-selected %d models fitting in 8 GB VRAM", len(model_ids))
    else:
        logger.error("Specify --models or --all-small")
        sys.exit(1)

    logger.info("Plan: %d models × %s × benchmarks=%s",
                len(model_ids), args.quant, args.benchmarks)
    for mid in model_ids:
        logger.info("  %s — %s", mid, MODELS[mid]["name"])

    if args.dry_run:
        logger.info("Dry-run complete. No models loaded.")
        return

    # ── Run benchmarks ──────────────────────────────────────────────────────
    from llm_bench.models.loader import ModelLoader, Quantization
    from llm_bench.benchmarks.speed import benchmark_speed
    from llm_bench.benchmarks.quality import run_quality_suite

    quant_map = {"fp16": Quantization.FP16, "8bit": Quantization.INT8, "4bit": Quantization.INT4}
    loader = ModelLoader()

    speed_results = []
    quality_results: dict = {b: [] for b in ("mmlu", "humaneval", "truthfulqa")}

    for model_id in model_ids:
        logger.info("=" * 60)
        logger.info("Benchmarking: %s (%s)", MODELS[model_id]["name"], args.quant)

        load_result = loader.load(model_id, quantization=quant_map[args.quant])
        if not load_result.success:
            logger.error("  ✗ Load failed: %s", load_result.error)
            continue

        # Speed
        if "speed" in args.benchmarks:
            sr = benchmark_speed(
                load_result.model, load_result.tokenizer,
                model_id=model_id, quantization=args.quant,
                num_tokens=args.speed_tokens,
            )
            logger.info("  ⚡ %.1f tok/s  TTFT=%.0f ms  VRAM Δ=%.2f GB",
                        sr.tokens_per_second, sr.time_to_first_token_ms, sr.memory_delta_gb)
            speed_results.append({
                "model_id": model_id,
                "quantization": args.quant,
                "tokens_per_sec": sr.tokens_per_second,
                "ttft_ms": sr.time_to_first_token_ms,
                "memory_gb": sr.memory_delta_gb,
            })

        # Quality
        q_benches = [b for b in args.benchmarks if b != "speed"]
        if q_benches:
            qrs = run_quality_suite(
                load_result.model, load_result.tokenizer,
                model_id, args.quant,
                benchmarks=q_benches,
                num_samples=args.mmlu_samples,
            )
            for bench, qr in qrs.items():
                logger.info("  🎯 %s = %.1f%%", bench.upper(), qr.score * 100)
                quality_results[bench].append({
                    "model_id": model_id,
                    "quantization": args.quant,
                    "score": round(qr.score, 4),
                    "correct": qr.correct,
                    "total": qr.total,
                })

        loader.unload(load_result)

    # ── Write output ────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{args.hardware}.json"
    payload = {
        "_meta": {
            "hardware_tag": args.hardware,
            "quantization": args.quant,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models_benchmarked": model_ids,
        },
        "speed_results": speed_results,
        "quality_results": {k: v for k, v in quality_results.items() if v},
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
