"""
Model loader for llm-bench.
Supports Hugging Face transformers (FP16, 8-bit, 4-bit via bitsandbytes)
and llama.cpp (GGUF) via llama-cpp-python.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums / constants
# ─────────────────────────────────────────────────────────────────────────────

class Backend(str, Enum):
    TRANSFORMERS = "transformers"
    LLAMACPP = "llamacpp"


class Quantization(str, Enum):
    FP16 = "fp16"
    INT8 = "8bit"
    INT4 = "4bit"
    GGUF_Q4 = "gguf-q4"
    GGUF_Q8 = "gguf-q8"
    CPU = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadResult:
    model_id: str
    backend: Backend
    quantization: str
    load_time_sec: float
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    success: bool = True
    error: Optional[str] = None
    model: Any = field(default=None, repr=False)
    tokenizer: Any = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backend": self.backend.value,
            "quantization": self.quantization,
            "load_time_sec": round(self.load_time_sec, 2),
            "vram_gb": round(self.vram_gb, 2),
            "ram_gb": round(self.ram_gb, 2),
            "success": self.success,
            "error": self.error,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loader class
# ─────────────────────────────────────────────────────────────────────────────

class ModelLoader:
    """
    Load local LLMs via either Hugging Face transformers or llama.cpp.

    Usage
    -----
    loader = ModelLoader(cache_dir="~/.cache/llm-bench")
    result = loader.load("llama-3.1-8b", quantization=Quantization.INT4)
    model, tokenizer = result.model, result.tokenizer
    """

    def __init__(
        self,
        cache_dir: str | Path = "~/.cache/llm-bench/models",
        device: Optional[str] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._device = device  # None → auto-detect

    # ── Public API ─────────────────────────────────────────────────────────

    def load(
        self,
        model_id: str,
        quantization: Quantization = Quantization.INT4,
        backend: Optional[Backend] = None,
        gguf_path: Optional[Path] = None,
    ) -> LoadResult:
        """
        Load a model. Auto-selects backend if not specified:
          - GGUF_* quant  →  llama.cpp
          - everything else  →  transformers
        """
        from llm_bench.models.registry import get_model

        meta = get_model(model_id)

        if backend is None:
            backend = (
                Backend.LLAMACPP
                if quantization in (Quantization.GGUF_Q4, Quantization.GGUF_Q8)
                else Backend.TRANSFORMERS
            )

        logger.info(
            "Loading %s | backend=%s | quant=%s",
            model_id, backend.value, quantization.value,
        )

        start = time.perf_counter()
        try:
            if backend == Backend.TRANSFORMERS:
                result = self._load_transformers(model_id, meta, quantization)
            else:
                result = self._load_llamacpp(model_id, meta, quantization, gguf_path)
            result.load_time_sec = time.perf_counter() - start
            return result
        except Exception as exc:
            logger.exception("Failed to load %s: %s", model_id, exc)
            return LoadResult(
                model_id=model_id,
                backend=backend,
                quantization=quantization.value,
                load_time_sec=time.perf_counter() - start,
                success=False,
                error=str(exc),
            )

    def unload(self, result: LoadResult) -> None:
        """Release GPU/CPU memory held by a loaded model."""
        try:
            import torch
            if result.model is not None:
                del result.model
            if result.tokenizer is not None:
                del result.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc; gc.collect()
            logger.info("Unloaded %s", result.model_id)
        except Exception as exc:
            logger.warning("Error during unload: %s", exc)

    # ── Transformers backend ───────────────────────────────────────────────

    def _load_transformers(
        self,
        model_id: str,
        meta: Dict[str, Any],
        quantization: Quantization,
    ) -> LoadResult:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        hf_repo = meta["hf_repo"]
        device = self._resolve_device()

        # ── Quantization config ──────────────────────────────────────────
        bnb_config = None
        dtype = torch.float16

        if quantization == Quantization.INT8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == Quantization.INT4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == Quantization.CPU:
            dtype = torch.float32

        # ── Record baseline VRAM ─────────────────────────────────────────
        vram_before = _gpu_allocated_gb() if torch.cuda.is_available() else 0.0

        # ── Load tokenizer ───────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(
            hf_repo,
            trust_remote_code=True,
            cache_dir=self.cache_dir / "hf",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── Load model ───────────────────────────────────────────────────
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "cache_dir": self.cache_dir / "hf",
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        elif device == "cpu":
            model_kwargs["torch_dtype"] = torch.float32
        else:
            model_kwargs["torch_dtype"] = dtype
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(hf_repo, **model_kwargs)
        model.eval()

        vram_after = _gpu_allocated_gb() if torch.cuda.is_available() else 0.0

        return LoadResult(
            model_id=model_id,
            backend=Backend.TRANSFORMERS,
            quantization=quantization.value,
            load_time_sec=0.0,  # filled in by caller
            vram_gb=max(0.0, vram_after - vram_before),
            model=model,
            tokenizer=tokenizer,
        )

    # ── llama.cpp backend ──────────────────────────────────────────────────

    def _load_llamacpp(
        self,
        model_id: str,
        meta: Dict[str, Any],
        quantization: Quantization,
        gguf_path: Optional[Path],
    ) -> LoadResult:
        from llama_cpp import Llama

        # Resolve GGUF file path
        if gguf_path is None:
            gguf_path = self._get_gguf_path(meta)

        if not gguf_path.exists():
            raise FileNotFoundError(
                f"GGUF file not found: {gguf_path}\n"
                f"Download it with: llm-bench download {model_id}"
            )

        n_gpu_layers = -1  # offload all layers to GPU by default

        try:
            import torch
            if not torch.cuda.is_available():
                n_gpu_layers = 0
        except ImportError:
            n_gpu_layers = 0

        model = Llama(
            model_path=str(gguf_path),
            n_ctx=meta.get("context", 4096),
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        return LoadResult(
            model_id=model_id,
            backend=Backend.LLAMACPP,
            quantization=quantization.value,
            load_time_sec=0.0,
            model=model,
            tokenizer=None,  # llama.cpp handles tokenization internally
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _get_gguf_path(self, meta: Dict[str, Any]) -> Path:
        gguf_file = meta.get("gguf_file", "model.gguf")
        return self.cache_dir / "gguf" / gguf_file


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_allocated_gb() -> float:
    """Return currently allocated GPU memory in GB."""
    try:
        import torch
        return torch.cuda.memory_allocated() / 1e9
    except Exception:
        return 0.0
