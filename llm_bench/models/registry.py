"""
Model registry for llm-bench.
Defines metadata for all supported local LLMs.
"""

from typing import Dict, Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# Each entry: model_id → metadata dict
# ─────────────────────────────────────────────────────────────────────────────

MODELS: Dict[str, Dict[str, Any]] = {

    # ── Llama 3.x ──────────────────────────────────────────────────────────
    "llama-3.1-8b": {
        "name": "Llama 3.1 8B Instruct",
        "family": "llama",
        "developer": "Meta",
        "params": "8B",
        "context": 128_000,
        "architecture": "decoder-only",
        "hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
        "gguf_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "gguf_file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 16, "8bit": 8, "4bit": 6},
        "min_ram_gb": {"cpu": 16},
        "license": "Llama 3.1 Community License",
        "strengths": ["general", "reasoning", "coding"],
        "tags": ["popular", "fast"],
    },
    "llama-3.1-70b": {
        "name": "Llama 3.1 70B Instruct",
        "family": "llama",
        "developer": "Meta",
        "params": "70B",
        "context": 128_000,
        "architecture": "decoder-only",
        "hf_repo": "meta-llama/Llama-3.1-70B-Instruct",
        "gguf_repo": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "gguf_file": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 140, "8bit": 70, "4bit": 40},
        "min_ram_gb": {"cpu": 80},
        "license": "Llama 3.1 Community License",
        "strengths": ["general", "reasoning", "long-context"],
        "tags": ["high-quality"],
    },
    "llama-3.2-3b": {
        "name": "Llama 3.2 3B Instruct",
        "family": "llama",
        "developer": "Meta",
        "params": "3B",
        "context": 128_000,
        "architecture": "decoder-only",
        "hf_repo": "meta-llama/Llama-3.2-3B-Instruct",
        "gguf_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "gguf_file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 6, "8bit": 3, "4bit": 2},
        "min_ram_gb": {"cpu": 6},
        "license": "Llama 3.2 Community License",
        "strengths": ["edge", "fast", "low-memory"],
        "tags": ["fast", "small"],
    },

    # ── Qwen 2.x ───────────────────────────────────────────────────────────
    "qwen2.5-7b": {
        "name": "Qwen 2.5 7B Instruct",
        "family": "qwen",
        "developer": "Alibaba",
        "params": "7B",
        "context": 32_768,
        "architecture": "decoder-only",
        "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
        "gguf_repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "gguf_file": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "min_vram_gb": {"fp16": 14, "8bit": 7, "4bit": 5},
        "min_ram_gb": {"cpu": 14},
        "license": "Apache 2.0",
        "strengths": ["coding", "math", "multilingual"],
        "tags": ["popular", "multilingual"],
    },
    "qwen2.5-14b": {
        "name": "Qwen 2.5 14B Instruct",
        "family": "qwen",
        "developer": "Alibaba",
        "params": "14B",
        "context": 32_768,
        "architecture": "decoder-only",
        "hf_repo": "Qwen/Qwen2.5-14B-Instruct",
        "gguf_repo": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "gguf_file": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "min_vram_gb": {"fp16": 28, "8bit": 14, "4bit": 8},
        "min_ram_gb": {"cpu": 28},
        "license": "Apache 2.0",
        "strengths": ["coding", "math", "reasoning"],
        "tags": ["balanced"],
    },
    "qwen2.5-72b": {
        "name": "Qwen 2.5 72B Instruct",
        "family": "qwen",
        "developer": "Alibaba",
        "params": "72B",
        "context": 32_768,
        "architecture": "decoder-only",
        "hf_repo": "Qwen/Qwen2.5-72B-Instruct",
        "gguf_repo": "Qwen/Qwen2.5-72B-Instruct-GGUF",
        "gguf_file": "qwen2.5-72b-instruct-q4_k_m.gguf",
        "min_vram_gb": {"fp16": 144, "8bit": 72, "4bit": 40},
        "min_ram_gb": {"cpu": 80},
        "license": "Qwen License",
        "strengths": ["coding", "math", "reasoning"],
        "tags": ["high-quality"],
    },
    "qwen2.5-coder-7b": {
        "name": "Qwen 2.5 Coder 7B",
        "family": "qwen",
        "developer": "Alibaba",
        "params": "7B",
        "context": 32_768,
        "architecture": "decoder-only",
        "hf_repo": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "gguf_repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "gguf_file": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "min_vram_gb": {"fp16": 14, "8bit": 7, "4bit": 5},
        "min_ram_gb": {"cpu": 14},
        "license": "Apache 2.0",
        "strengths": ["coding", "debugging", "code-completion"],
        "tags": ["coding-specialist"],
    },

    # ── Mistral / Mixtral ──────────────────────────────────────────────────
    "mistral-7b-v0.3": {
        "name": "Mistral 7B v0.3 Instruct",
        "family": "mistral",
        "developer": "Mistral AI",
        "params": "7B",
        "context": 32_768,
        "architecture": "decoder-only",
        "hf_repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "gguf_repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "gguf_file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 14, "8bit": 7, "4bit": 5},
        "min_ram_gb": {"cpu": 14},
        "license": "Apache 2.0",
        "strengths": ["general", "fast", "efficient"],
        "tags": ["popular", "fast"],
    },
    "mistral-nemo-12b": {
        "name": "Mistral Nemo 12B Instruct",
        "family": "mistral",
        "developer": "Mistral AI",
        "params": "12B",
        "context": 128_000,
        "architecture": "decoder-only",
        "hf_repo": "mistralai/Mistral-Nemo-Instruct-2407",
        "gguf_repo": "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        "gguf_file": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 24, "8bit": 12, "4bit": 7},
        "min_ram_gb": {"cpu": 24},
        "license": "Apache 2.0",
        "strengths": ["long-context", "reasoning"],
        "tags": ["balanced", "long-context"],
    },
    "mixtral-8x7b": {
        "name": "Mixtral 8x7B Instruct",
        "family": "mistral",
        "developer": "Mistral AI",
        "params": "47B (MoE)",
        "context": 32_768,
        "architecture": "mixture-of-experts",
        "hf_repo": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "gguf_repo": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "gguf_file": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 90, "8bit": 48, "4bit": 26},
        "min_ram_gb": {"cpu": 48},
        "license": "Apache 2.0",
        "strengths": ["reasoning", "coding", "multilingual"],
        "tags": ["moe", "popular"],
    },

    # ── DeepSeek ───────────────────────────────────────────────────────────
    "deepseek-r1-7b": {
        "name": "DeepSeek R1 Distill Qwen 7B",
        "family": "deepseek",
        "developer": "DeepSeek",
        "params": "7B",
        "context": 32_768,
        "architecture": "decoder-only",
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "gguf_repo": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "gguf_file": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 14, "8bit": 7, "4bit": 5},
        "min_ram_gb": {"cpu": 14},
        "license": "MIT",
        "strengths": ["reasoning", "math", "coding"],
        "tags": ["reasoning-specialist", "popular"],
    },
    "deepseek-r1-14b": {
        "name": "DeepSeek R1 Distill Qwen 14B",
        "family": "deepseek",
        "developer": "DeepSeek",
        "params": "14B",
        "context": 32_768,
        "architecture": "decoder-only",
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "gguf_repo": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "gguf_file": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 28, "8bit": 14, "4bit": 8},
        "min_ram_gb": {"cpu": 28},
        "license": "MIT",
        "strengths": ["reasoning", "math", "coding"],
        "tags": ["reasoning-specialist"],
    },
    "deepseek-coder-v2-16b": {
        "name": "DeepSeek Coder V2 16B",
        "family": "deepseek",
        "developer": "DeepSeek",
        "params": "16B (MoE)",
        "context": 163_840,
        "architecture": "mixture-of-experts",
        "hf_repo": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "gguf_repo": "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "gguf_file": "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 32, "8bit": 16, "4bit": 10},
        "min_ram_gb": {"cpu": 32},
        "license": "DeepSeek License",
        "strengths": ["coding", "long-context"],
        "tags": ["coding-specialist", "moe"],
    },

    # ── Phi-3 / Phi-4 ──────────────────────────────────────────────────────
    "phi-3.5-mini": {
        "name": "Phi 3.5 Mini Instruct",
        "family": "phi",
        "developer": "Microsoft",
        "params": "3.8B",
        "context": 128_000,
        "architecture": "decoder-only",
        "hf_repo": "microsoft/Phi-3.5-mini-instruct",
        "gguf_repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "gguf_file": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 8, "8bit": 4, "4bit": 3},
        "min_ram_gb": {"cpu": 8},
        "license": "MIT",
        "strengths": ["fast", "coding", "reasoning"],
        "tags": ["small", "fast", "popular"],
    },
    "phi-4": {
        "name": "Phi 4 (14B)",
        "family": "phi",
        "developer": "Microsoft",
        "params": "14B",
        "context": 16_384,
        "architecture": "decoder-only",
        "hf_repo": "microsoft/phi-4",
        "gguf_repo": "bartowski/phi-4-GGUF",
        "gguf_file": "phi-4-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 28, "8bit": 14, "4bit": 8},
        "min_ram_gb": {"cpu": 28},
        "license": "MIT",
        "strengths": ["reasoning", "math", "coding"],
        "tags": ["high-quality", "popular"],
    },

    # ── Gemma ──────────────────────────────────────────────────────────────
    "gemma-2-9b": {
        "name": "Gemma 2 9B Instruct",
        "family": "gemma",
        "developer": "Google",
        "params": "9B",
        "context": 8_192,
        "architecture": "decoder-only",
        "hf_repo": "google/gemma-2-9b-it",
        "gguf_repo": "bartowski/gemma-2-9b-it-GGUF",
        "gguf_file": "gemma-2-9b-it-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 18, "8bit": 9, "4bit": 6},
        "min_ram_gb": {"cpu": 18},
        "license": "Gemma License",
        "strengths": ["general", "reasoning"],
        "tags": ["popular"],
    },
    "gemma-2-27b": {
        "name": "Gemma 2 27B Instruct",
        "family": "gemma",
        "developer": "Google",
        "params": "27B",
        "context": 8_192,
        "architecture": "decoder-only",
        "hf_repo": "google/gemma-2-27b-it",
        "gguf_repo": "bartowski/gemma-2-27b-it-GGUF",
        "gguf_file": "gemma-2-27b-it-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 54, "8bit": 27, "4bit": 16},
        "min_ram_gb": {"cpu": 54},
        "license": "Gemma License",
        "strengths": ["general", "reasoning", "high-quality"],
        "tags": ["high-quality"],
    },

    # ── Command-R ──────────────────────────────────────────────────────────
    "command-r-35b": {
        "name": "Command R (35B)",
        "family": "command-r",
        "developer": "Cohere",
        "params": "35B",
        "context": 128_000,
        "architecture": "decoder-only",
        "hf_repo": "CohereForAI/c4ai-command-r-v01",
        "gguf_repo": "bartowski/c4ai-command-r-v01-GGUF",
        "gguf_file": "c4ai-command-r-v01-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 70, "8bit": 35, "4bit": 20},
        "min_ram_gb": {"cpu": 70},
        "license": "CC BY-NC 4.0",
        "strengths": ["rag", "long-context", "reasoning"],
        "tags": ["rag-specialist", "long-context"],
    },

    # ── Yi ─────────────────────────────────────────────────────────────────
    "yi-1.5-9b": {
        "name": "Yi 1.5 9B Chat",
        "family": "yi",
        "developer": "01.AI",
        "params": "9B",
        "context": 4_096,
        "architecture": "decoder-only",
        "hf_repo": "01-ai/Yi-1.5-9B-Chat",
        "gguf_repo": "bartowski/Yi-1.5-9B-Chat-GGUF",
        "gguf_file": "Yi-1.5-9B-Chat-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 18, "8bit": 9, "4bit": 6},
        "min_ram_gb": {"cpu": 18},
        "license": "Apache 2.0",
        "strengths": ["multilingual", "general"],
        "tags": ["multilingual"],
    },

    # ── Vicuna / Openchat ──────────────────────────────────────────────────
    "openchat-3.5": {
        "name": "OpenChat 3.5 (7B)",
        "family": "openchat",
        "developer": "OpenChat",
        "params": "7B",
        "context": 8_192,
        "architecture": "decoder-only",
        "hf_repo": "openchat/openchat-3.5-0106",
        "gguf_repo": "TheBloke/openchat-3.5-0106-GGUF",
        "gguf_file": "openchat-3.5-0106.Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 14, "8bit": 7, "4bit": 5},
        "min_ram_gb": {"cpu": 14},
        "license": "Apache 2.0",
        "strengths": ["chat", "instruction-following"],
        "tags": ["fast"],
    },

    # ── Neural Chat / Zephyr ───────────────────────────────────────────────
    "zephyr-7b-beta": {
        "name": "Zephyr 7B Beta",
        "family": "zephyr",
        "developer": "HuggingFace H4",
        "params": "7B",
        "context": 32_768,
        "architecture": "decoder-only",
        "hf_repo": "HuggingFaceH4/zephyr-7b-beta",
        "gguf_repo": "TheBloke/zephyr-7B-beta-GGUF",
        "gguf_file": "zephyr-7b-beta.Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 14, "8bit": 7, "4bit": 5},
        "min_ram_gb": {"cpu": 14},
        "license": "MIT",
        "strengths": ["chat", "instruction-following"],
        "tags": ["fast"],
    },

    # ── CodeLlama ──────────────────────────────────────────────────────────
    "codellama-13b": {
        "name": "Code Llama 13B Instruct",
        "family": "llama",
        "developer": "Meta",
        "params": "13B",
        "context": 16_384,
        "architecture": "decoder-only",
        "hf_repo": "codellama/CodeLlama-13b-Instruct-hf",
        "gguf_repo": "TheBloke/CodeLlama-13B-Instruct-GGUF",
        "gguf_file": "codellama-13b-instruct.Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 26, "8bit": 13, "4bit": 8},
        "min_ram_gb": {"cpu": 26},
        "license": "Llama 2 Community License",
        "strengths": ["coding", "infill", "debugging"],
        "tags": ["coding-specialist"],
    },

    # ── StarCoder2 ─────────────────────────────────────────────────────────
    "starcoder2-15b": {
        "name": "StarCoder2 15B",
        "family": "starcoder",
        "developer": "BigCode",
        "params": "15B",
        "context": 16_384,
        "architecture": "decoder-only",
        "hf_repo": "bigcode/starcoder2-15b-instruct-v0.1",
        "gguf_repo": "bartowski/starcoder2-15b-instruct-v0.1-GGUF",
        "gguf_file": "starcoder2-15b-instruct-v0.1-Q4_K_M.gguf",
        "min_vram_gb": {"fp16": 30, "8bit": 15, "4bit": 9},
        "min_ram_gb": {"cpu": 30},
        "license": "BigCode Open RAIL-M v1",
        "strengths": ["coding", "code-completion", "fill-in-middle"],
        "tags": ["coding-specialist"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def get_model(model_id: str) -> Dict[str, Any]:
    """Return metadata for a model by ID. Raises KeyError if not found."""
    if model_id not in MODELS:
        raise KeyError(f"Model '{model_id}' not found. Available: {list(MODELS.keys())}")
    return MODELS[model_id]


def list_models_by_family(family: str) -> Dict[str, Dict[str, Any]]:
    """Return all models belonging to a given family."""
    return {k: v for k, v in MODELS.items() if v["family"] == family}


def list_models_by_vram(max_vram_gb: float, quantization: str = "4bit") -> Dict[str, Dict[str, Any]]:
    """
    Return models whose minimum VRAM (for the given quantization) fits within
    max_vram_gb.  CPU-only models are included when max_vram_gb == 0.
    """
    result = {}
    for model_id, meta in MODELS.items():
        vram_map = meta.get("min_vram_gb", {})
        required = vram_map.get(quantization)
        if required is None:
            continue
        if required <= max_vram_gb:
            result[model_id] = meta
    return result


def list_models_by_tag(tag: str) -> Dict[str, Dict[str, Any]]:
    """Return all models that carry a specific tag."""
    return {k: v for k, v in MODELS.items() if tag in v.get("tags", [])}


def get_families() -> list:
    """Return sorted list of unique model families."""
    return sorted({v["family"] for v in MODELS.values()})


def get_all_tags() -> list:
    """Return sorted list of all unique tags across models."""
    tags: set = set()
    for meta in MODELS.values():
        tags.update(meta.get("tags", []))
    return sorted(tags)


def recommend_for_hardware(vram_gb: float, ram_gb: float) -> Dict[str, Dict[str, Any]]:
    """
    Given available VRAM and system RAM, return a prioritised dict of
    recommended models with their suggested quantization level.
    """
    recommendations: Dict[str, Dict[str, Any]] = {}

    for model_id, meta in MODELS.items():
        vram_map = meta.get("min_vram_gb", {})
        ram_map = meta.get("min_ram_gb", {})

        best_quant: Optional[str] = None
        for quant in ("fp16", "8bit", "4bit"):
            req_vram = vram_map.get(quant, float("inf"))
            if vram_gb >= req_vram:
                best_quant = quant
                break

        # Fall back to CPU if no VRAM option fits
        if best_quant is None:
            cpu_ram = ram_map.get("cpu", float("inf"))
            if ram_gb >= cpu_ram:
                best_quant = "cpu-4bit"

        if best_quant:
            recommendations[model_id] = {**meta, "recommended_quant": best_quant}

    return recommendations
