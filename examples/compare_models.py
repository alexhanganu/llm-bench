"""
examples/compare_models.py
--------------------------
Compare three models side-by-side on speed + MMLU quality.

Usage:
    python examples/compare_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


MODELS_TO_COMPARE = [
    ("llama-3.1-8b",   "4bit"),
    ("qwen2.5-7b",     "4bit"),
    ("mistral-7b-v0.3","4bit"),
]


def main():
    from llm_bench.models.loader import ModelLoader, Quantization
    from llm_bench.benchmarks.speed import benchmark_speed
    from llm_bench.benchmarks.quality import evaluate_mmlu
    from llm_bench.results.database import ResultsDB
    from llm_bench.models.registry import MODELS

    quant_map = {"fp16": Quantization.FP16, "8bit": Quantization.INT8, "4bit": Quantization.INT4}

    loader = ModelLoader()
    db = ResultsDB()
    results = []

    for model_id, quant in MODELS_TO_COMPARE:
        print(f"\n{'='*55}")
        print(f"  {MODELS[model_id]['name']} ({quant})")
        print(f"{'='*55}")

        load_result = loader.load(model_id, quantization=quant_map[quant])
        if not load_result.success:
            print(f"  ✗ Load failed: {load_result.error}")
            continue

        # Speed
        print("  Benchmarking speed…", end=" ", flush=True)
        sr = benchmark_speed(load_result.model, load_result.tokenizer,
                             model_id=model_id, quantization=quant)
        print(f"{sr.tokens_per_second:.1f} tok/s")

        # Quality
        print("  Benchmarking MMLU (50 samples)…", end=" ", flush=True)
        qr = evaluate_mmlu(load_result.model, load_result.tokenizer,
                           model_id=model_id, quantization=quant, num_samples=50)
        print(f"{qr.pct}")

        # Save to DB
        db.upsert_speed(model_id, quant, load_result.backend.value,
                        sr.tokens_per_second, sr.time_to_first_token_ms,
                        sr.memory_delta_gb, sr.total_time_sec)
        db.upsert_quality(model_id, quant, "mmlu",
                          qr.score, qr.correct, qr.total, qr.elapsed_sec)

        results.append({
            "model": MODELS[model_id]["name"],
            "quant": quant,
            "tok_s": sr.tokens_per_second,
            "ttft_ms": sr.time_to_first_token_ms,
            "vram_gb": sr.memory_delta_gb,
            "mmlu": qr.score,
        })

        loader.unload(load_result)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  {'Model':<28} {'Quant':<6} {'Tok/s':>7} {'TTFT':>8} {'VRAM':>7} {'MMLU':>7}")
    print(f"  {'-'*28} {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")
    for r in sorted(results, key=lambda x: x["tok_s"], reverse=True):
        print(
            f"  {r['model']:<28} {r['quant']:<6} "
            f"{r['tok_s']:>6.1f} {r['ttft_ms']:>7.0f}ms "
            f"{r['vram_gb']:>6.1f}G {r['mmlu']*100:>6.1f}%"
        )
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
