# Hardware Guide

## GPU Tiers

### 🔥 High-end (40+ GB VRAM) — A100, H100, 2× RTX 3090

Run any model in the registry. FP16 precision is viable for 70B models.

| Recommended | Quantization | Why |
|---|---|---|
| Llama 3.1 70B | fp16 or 4bit | Best overall open-source model |
| Qwen 2.5 72B | 4bit | Top coding + math performance |
| Mixtral 8x7B | fp16 | Excellent MoE architecture |
| Command-R 35B | 4bit | Best RAG / long-context retrieval |

---

### ⚡ Mid-range (20–40 GB VRAM) — RTX 3090, RTX 4090, A5000

Best balance of quality and speed. The sweet spot for local LLM use.

| Recommended | Quantization | Why |
|---|---|---|
| Qwen 2.5 14B | 4bit | ~79% MMLU, fits in 9 GB |
| Phi-4 | 4bit | 80%+ MMLU in a 14B model |
| DeepSeek R1 14B | 4bit | Best reasoning/math at this tier |
| Mistral Nemo 12B | 4bit | 128K context, Apache 2.0 |

---

### ✅ Consumer GPU (6–20 GB VRAM) — RTX 3060/3070/3080, 4060/4070

7B–8B models at 4-bit are the sweet spot. Expect 60–100+ tok/s.

| Recommended | Quantization | VRAM | Tok/s (est.) |
|---|---|---|---|
| Mistral 7B v0.3 | 4bit | 5.0 GB | 95 |
| Qwen 2.5 7B | 4bit | 5.1 GB | 91 |
| Llama 3.1 8B | 4bit | 5.8 GB | 87 |
| DeepSeek R1 7B | 4bit | 5.4 GB | 78 |
| Phi 3.5 Mini | 4bit | 2.9 GB | 132 |

---

### 🖥️ CPU-only

Expect 3–15 tok/s. Prefer 3B–7B models with GGUF quantization.

```bash
# Install llama.cpp Python bindings (CPU build)
pip install llama-cpp-python

# Run a GGUF model
llm-bench benchmark --model phi-3.5-mini --quant gguf-q4
```

---

## Quantization Explained

| Level | Speed | Quality loss | VRAM vs FP16 |
|---|---|---|---|
| **FP16** | baseline | none | 1× |
| **8-bit** | ~1.3× | negligible | ~0.5× |
| **4-bit (NF4)** | ~1.8× | small (~1–3% MMLU) | ~0.28× |
| **GGUF Q4_K_M** | ~2× | small | ~0.28× |

**Recommendation**: Start with 4-bit. Only go to 8-bit or FP16 if you notice
degraded output quality on your specific task.

---

## Multi-GPU Setup

llm-bench supports multi-GPU via `device_map="auto"` in transformers.
No extra configuration needed — it distributes layers automatically.

```bash
# Verify your GPUs are visible
llm-bench hardware
# GPU 0: NVIDIA A100 80GB  80.0 GB VRAM (78.2 GB free)
# GPU 1: NVIDIA A100 80GB  80.0 GB VRAM (78.9 GB free)
```

---

## Apple Silicon (M1/M2/M3)

Metal Performance Shaders (MPS) backend is supported via llama.cpp.
Unified memory means VRAM = system RAM — M2 Max 96GB can run 70B at 4-bit.

```bash
pip install "llama-cpp-python[metal]"
llm-bench benchmark --model llama-3.1-70b --quant gguf-q4
```

Expected throughput on M2 Max:
- Llama 3.1 8B (4bit): ~68 tok/s
- Llama 3.1 70B (4bit): ~12 tok/s
