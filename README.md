# 🦙 llm-bench

> **Compare 20+ local LLMs on your hardware — see speed, quality, and memory before downloading.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/your-org/llm-bench?style=social)](https://github.com/your-org/llm-bench)

Stop wasting hours downloading models that don't fit your GPU or use case.
`llm-bench` benchmarks speed, quality, and memory — and shows you the results
*before* you commit to a download.

---

## ✨ Features

| Feature | Details |
|---|---|
| **25+ models** | Llama, Qwen, Mistral, DeepSeek, Phi, Gemma, Mixtral, CodeLlama, StarCoder2… |
| **Speed benchmark** | Tokens/second, time-to-first-token, throughput at multiple quantizations |
| **Quality benchmarks** | MMLU, HumanEval (pass@1), TruthfulQA |
| **Memory profiling** | VRAM usage at load + during inference, across FP16/8-bit/4-bit |
| **Hardware-aware recs** | "Your RTX 3090 can run these 12 models at 4-bit" |
| **Pre-computed results** | Browse results offline without downloading a single model |
| **Interactive dashboard** | Plotly charts — scatter, radar, bar, table, side-by-side |
| **CLI** | `llm-bench benchmark`, `llm-bench hardware`, `llm-bench list` |
| **SQLite caching** | Results persist; re-run only when you update a model |

---

## 🚀 Quick Start

```bash
# Install
pip install llm-bench          # CPU + dashboard only
pip install "llm-bench[gpu]"   # + GPU inference (torch, transformers, bitsandbytes)
pip install "llm-bench[gguf]"  # + llama.cpp / GGUF support

# Launch the dashboard (uses pre-computed results — no GPU needed)
llm-bench dashboard
# → opens http://localhost:8501

# Check your hardware
llm-bench hardware

# List compatible models for your GPU
llm-bench list --max-vram 24

# Run a benchmark
llm-bench benchmark --model llama-3.1-8b --quant 4bit --benchmarks speed,mmlu
```

---

## 📊 Dashboard

```bash
streamlit run app.py
```

Five tabs:
- **Overview** — speed vs quality scatter + data table
- **Speed** — bar charts, TTFT comparison
- **Quality** — MMLU radar, HumanEval scores
- **Memory** — VRAM usage by model & quantization
- **Live Benchmark** — run on your own GPU

---

## 🖥️ Hardware Recommendations

```
$ llm-bench hardware

🖥️  Hardware Profile
CPU  : AMD Ryzen 9 5900X (12C / 24T)
RAM  : 64.0 GB total  (48.2 GB free)
GPU 0: NVIDIA GeForce RTX 3090  24.0 GB VRAM  (22.1 GB free)
CUDA : 12.1  PyTorch 2.3.0

💡 Your RTX 3090 can handle most 7B–30B models with ease

Recommended models:
  • Llama 3.1 8B Instruct (4bit) — widely used, fast inference
  • Qwen 2.5 7B Instruct (4bit) — great at coding, multilingual
  • Phi 3.5 Mini Instruct (4bit) — fast inference
  • Mistral 7B v0.3 Instruct (4bit) — widely used, fast inference
  • Qwen 2.5 14B Instruct (4bit) — balanced
  • DeepSeek R1 Distill 7B (4bit) — strong reasoning
```

---

## 📈 Pre-computed Results (RTX 3090, 4-bit)

| Model | Tok/s | TTFT | VRAM | MMLU |
|---|---:|---:|---:|---:|
| Llama 3.2 3B | 148.6 | 22ms | 2.4 GB | 58.9% |
| Phi 3.5 Mini | 132.4 | 26ms | 2.9 GB | 69.3% |
| Mistral 7B v0.3 | 95.1 | 34ms | 4.9 GB | 64.1% |
| Qwen 2.5 7B | 91.3 | 36ms | 5.1 GB | 72.1% |
| Llama 3.1 8B | 87.4 | 38ms | 5.8 GB | 68.2% |
| DeepSeek R1 7B | 78.3 | 41ms | 5.4 GB | 71.2% |
| Gemma 2 9B | 72.8 | 44ms | 5.9 GB | 71.8% |
| Qwen 2.5 14B | 39.1 | 72ms | 8.8 GB | 78.9% |

*Pre-computed results for RTX 3090. Run `llm-bench benchmark` to generate results for your hardware.*

---

## 🏗️ Architecture

```
llm-bench/
├── app.py                        # Streamlit dashboard
├── llm_bench/
│   ├── models/
│   │   ├── registry.py           # 25+ model definitions
│   │   └── loader.py             # HF transformers + llama.cpp loader
│   ├── benchmarks/
│   │   ├── speed.py              # Tokens/sec, TTFT, memory delta
│   │   ├── quality.py            # MMLU, HumanEval, TruthfulQA
│   │   └── memory.py             # VRAM profiler + context manager
│   ├── results/
│   │   └── database.py           # SQLite store with upsert
│   ├── utils/
│   │   └── hardware_detect.py    # GPU/CPU/RAM detection + recommendations
│   └── cli.py                    # Click CLI
└── data/precomputed/             # Pre-run JSON results (offline use)
```

---

## 🔧 Python API

```python
from llm_bench.models.registry import MODELS, list_models_by_vram, recommend_for_hardware
from llm_bench.models.loader import ModelLoader, Quantization
from llm_bench.benchmarks.speed import benchmark_speed
from llm_bench.benchmarks.quality import evaluate_mmlu

# Which 7B-ish models fit in 8 GB VRAM at 4-bit?
models = list_models_by_vram(max_vram_gb=8, quantization="4bit")

# Load and benchmark
loader = ModelLoader()
result = loader.load("qwen2.5-7b", quantization=Quantization.INT4)

speed = benchmark_speed(result.model, result.tokenizer, model_id="qwen2.5-7b")
print(f"{speed.tokens_per_second:.1f} tok/s")

quality = evaluate_mmlu(result.model, result.tokenizer, num_samples=100)
print(f"MMLU: {quality.pct}")

loader.unload(result)
```

---

## 🤝 Contributing

1. Add a model → edit `llm_bench/models/registry.py`
2. Add a benchmark → add a module in `llm_bench/benchmarks/`
3. Run pre-compute → `python scripts/precompute.py --hardware rtx3090`
4. Open a PR!

---

## 📜 License

MIT. Pre-computed results are provided as-is; accuracy numbers reflect a specific hardware configuration and may differ on yours.
