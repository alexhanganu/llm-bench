# Benchmark Guide

## Speed Benchmark

Measures **tokens per second** (throughput) and **time to first token** (TTFT / latency).

### Methodology

1. **Warmup pass** — generate 16 tokens to warm the GPU kernels (not timed)
2. **TTFT** — time a single-token generation from the same prompt
3. **Throughput** — generate 256 tokens, record wall-clock time

```python
from llm_bench.benchmarks.speed import benchmark_speed

result = benchmark_speed(model, tokenizer, model_id="llama-3.1-8b", num_tokens=256)
print(result.tokens_per_second)  # e.g. 87.4
print(result.time_to_first_token_ms)  # e.g. 38 ms
```

**Factors affecting speed:**
- Quantization (4-bit is fastest)
- Context length (longer = slower per token)
- Batch size (llm-bench uses batch=1 / single-user)
- GPU memory bandwidth (the primary bottleneck for inference)

---

## MMLU — Massive Multitask Language Understanding

57 tasks across STEM, humanities, social sciences, law, medicine, and more.
Each question is 4-choice multiple choice.

**Score interpretation:**
- < 50% = below random chance (something is wrong)
- 50–60% = weak
- 60–70% = decent — good for most casual use
- 70–80% = strong — suitable for complex tasks
- 80%+ = excellent — frontier-level open-source

```python
from llm_bench.benchmarks.quality import evaluate_mmlu
result = evaluate_mmlu(model, tokenizer, num_samples=100)
print(result.pct)  # e.g. "72.1%"
```

---

## HumanEval — Code Generation (pass@1)

164 Python programming problems. The model must write a function body that
passes a hidden test suite. Score = fraction of problems solved.

**Score interpretation:**
- < 30% = poor coding ability
- 30–60% = capable of simple functions
- 60–80% = handles most real-world Python tasks
- 80%+ = strong coding assistant

```python
from llm_bench.benchmarks.quality import evaluate_humaneval
result = evaluate_humaneval(model, tokenizer, num_samples=50)
print(result.pct)
```

> ⚠️ HumanEval executes generated Python code in a subprocess with a timeout.
> This is inherently unsafe. Run only on trusted machines.

---

## TruthfulQA

817 questions designed to elicit false beliefs. Tests factual accuracy and
resistance to hallucination.

**Score interpretation:**
- ~50% = human baseline
- 60%+ = meaningfully truthful

---

## Needle in a Haystack (Long-context)

Tests whether a model actually uses its claimed context window. A "needle"
(short factual statement) is embedded in a long filler document at varying
depths, then the model is asked to retrieve it.

```python
from llm_bench.benchmarks.long_context import benchmark_long_context

result = benchmark_long_context(
    model, tokenizer,
    model_id="mistral-nemo-12b",
    context_lengths=[4096, 16384, 32768],
    depth_levels=[0.1, 0.5, 0.9],
)
print(f"Overall: {result.overall_accuracy:.0%}")
print(result.accuracy_by_context())  # {4096: 1.0, 16384: 0.8, 32768: 0.4}
```

The results render as a heatmap in the dashboard's Quality tab.

---

## Running the Full Suite

```bash
# Full benchmark (loads model once, runs all benchmarks)
llm-bench benchmark \
  --model qwen2.5-7b \
  --quant 4bit \
  --benchmarks speed mmlu humaneval \
  --num-samples 100 \
  --output results/qwen2.5-7b.json
```

Or via the Streamlit Live Benchmark tab.
