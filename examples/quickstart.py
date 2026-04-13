"""
examples/quickstart.py
----------------------
Run a single speed benchmark on one model.
Assumes the model is already downloaded.

Usage:
    python examples/quickstart.py --model llama-3.1-8b --quant 4bit
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama-3.1-8b")
    parser.add_argument("--quant", default="4bit", choices=["fp16", "8bit", "4bit"])
    args = parser.parse_args()

    from llm_bench.models.registry import get_model
    from llm_bench.models.loader import ModelLoader, Quantization
    from llm_bench.benchmarks.speed import benchmark_speed
    from llm_bench.utils.hardware_detect import detect_hardware

    print("=" * 55)
    print("  llm-bench quickstart")
    print("=" * 55)

    # Show hardware
    hw = detect_hardware()
    print(hw.summary())
    print()

    meta = get_model(args.model)
    print(f"Model : {meta['name']}")
    print(f"Quant : {args.quant}")
    print()

    quant_map = {"fp16": Quantization.FP16, "8bit": Quantization.INT8, "4bit": Quantization.INT4}

    print("Loading model…")
    loader = ModelLoader()
    result = loader.load(args.model, quantization=quant_map[args.quant])

    if not result.success:
        print(f"ERROR: {result.error}")
        sys.exit(1)

    print(f"Loaded in {result.load_time_sec:.1f}s  (VRAM delta: {result.vram_gb:.2f} GB)")
    print()

    print("Running speed benchmark…")
    sr = benchmark_speed(
        result.model, result.tokenizer,
        model_id=args.model, quantization=args.quant,
    )

    print(f"  Speed (tok/s)  : {sr.tokens_per_second:.1f}")
    print(f"  TTFT           : {sr.time_to_first_token_ms:.0f} ms")
    print(f"  VRAM delta     : {sr.memory_delta_gb:.2f} GB")
    print(f"  Tokens gen.    : {sr.generated_tokens}")

    loader.unload(result)
    print("\nDone.")


if __name__ == "__main__":
    main()
