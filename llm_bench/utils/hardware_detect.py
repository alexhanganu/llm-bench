"""
Hardware detection for llm-bench.

Auto-detects CPU, GPU (VRAM), and system RAM.
Produces model recommendations based on available resources.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    driver_version: str = ""
    cuda_capability: str = ""


@dataclass
class HardwareProfile:
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    gpus: List[GPUInfo] = field(default_factory=list)
    platform_str: str = ""
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""

    # ── Derived properties ─────────────────────────────────────────────────

    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    @property
    def total_vram_gb(self) -> float:
        return sum(g.vram_total_gb for g in self.gpus)

    @property
    def free_vram_gb(self) -> float:
        return sum(g.vram_free_gb for g in self.gpus)

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        return self.gpus[0] if self.gpus else None

    def summary(self) -> str:
        lines = [
            f"CPU  : {self.cpu_name} ({self.cpu_cores}C / {self.cpu_threads}T)",
            f"RAM  : {self.ram_total_gb:.1f} GB total  ({self.ram_available_gb:.1f} GB free)",
        ]
        if self.gpus:
            for g in self.gpus:
                lines.append(
                    f"GPU {g.index}: {g.name}  {g.vram_total_gb:.1f} GB VRAM  "
                    f"({g.vram_free_gb:.1f} GB free)"
                )
        else:
            lines.append("GPU  : None detected – CPU inference only")
        if self.cuda_version:
            lines.append(f"CUDA : {self.cuda_version}  PyTorch {self.torch_version}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["has_gpu"] = self.has_gpu
        d["total_vram_gb"] = round(self.total_vram_gb, 2)
        d["free_vram_gb"] = round(self.free_vram_gb, 2)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_hardware() -> HardwareProfile:
    """Probe the current system and return a HardwareProfile."""
    import psutil

    # ── CPU / RAM ──────────────────────────────────────────────────────────
    cpu_name = _cpu_name()
    cpu_cores = psutil.cpu_count(logical=False) or 1
    cpu_threads = psutil.cpu_count(logical=True) or 1
    mem = psutil.virtual_memory()
    ram_total = mem.total / 1e9
    ram_avail = mem.available / 1e9

    # ── GPUs ───────────────────────────────────────────────────────────────
    gpus: List[GPUInfo] = _detect_gpus()

    # ── Versions ───────────────────────────────────────────────────────────
    import sys
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    torch_ver = cuda_ver = ""
    try:
        import torch
        torch_ver = torch.__version__
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda or ""
    except ImportError:
        pass

    return HardwareProfile(
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        ram_total_gb=round(ram_total, 2),
        ram_available_gb=round(ram_avail, 2),
        gpus=gpus,
        platform_str=platform.platform(),
        python_version=py_ver,
        torch_version=torch_ver,
        cuda_version=cuda_ver,
    )


def _detect_gpus() -> List[GPUInfo]:
    gpus: List[GPUInfo] = []

    # Try PyTorch / CUDA first
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_b, total_b = torch.cuda.mem_get_info(i)
                gpus.append(GPUInfo(
                    index=i,
                    name=props.name,
                    vram_total_gb=round(total_b / 1e9, 2),
                    vram_free_gb=round(free_b / 1e9, 2),
                    cuda_capability=f"{props.major}.{props.minor}",
                ))
            return gpus
    except Exception:
        pass

    # Fall back to nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode()
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            idx, name, mem_total, mem_free, drv = parts[:5]
            gpus.append(GPUInfo(
                index=int(idx),
                name=name,
                vram_total_gb=round(int(mem_total) / 1024, 2),
                vram_free_gb=round(int(mem_free) / 1024, 2),
                driver_version=drv,
            ))
    except Exception:
        pass

    # Apple Silicon / Metal
    if not gpus:
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS doesn't expose VRAM directly; use system RAM as proxy
                import psutil
                mem = psutil.virtual_memory()
                gpus.append(GPUInfo(
                    index=0,
                    name="Apple Silicon (MPS)",
                    vram_total_gb=round(mem.total / 1e9, 2),
                    vram_free_gb=round(mem.available / 1e9, 2),
                ))
        except Exception:
            pass

    return gpus


def _cpu_name() -> str:
    """Return a human-readable CPU name."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    elif system == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], timeout=3
            ).decode().strip()
        except Exception:
            pass
    elif system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                  r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return name
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation engine
# ─────────────────────────────────────────────────────────────────────────────

def get_model_recommendations(
    hw: Optional[HardwareProfile] = None,
) -> Dict[str, Any]:
    """
    Return structured recommendations based on detected hardware.

    Returns
    -------
    {
        "hardware": HardwareProfile,
        "tier": "gpu-high" | "gpu-mid" | "gpu-low" | "cpu",
        "headline": str,
        "recommended_models": [{"model_id": ..., "quant": ..., "reason": ...}],
    }
    """
    from llm_bench.models.registry import recommend_for_hardware

    if hw is None:
        hw = detect_hardware()

    vram = hw.free_vram_gb
    ram = hw.ram_available_gb

    # Classify hardware tier
    if vram >= 40:
        tier = "gpu-high"
        headline = f"🔥 {hw.primary_gpu.name if hw.primary_gpu else 'Your GPU'} can run the biggest open-source models"
    elif vram >= 20:
        tier = "gpu-mid"
        headline = f"⚡ {hw.primary_gpu.name if hw.primary_gpu else 'Your GPU'} handles most 7B–30B models with ease"
    elif vram >= 6:
        tier = "gpu-low"
        headline = f"✅ {hw.primary_gpu.name if hw.primary_gpu else 'Your GPU'} is great for 7B–8B models"
    else:
        tier = "cpu"
        headline = "🖥️ CPU-only mode — 3B–7B quantised models will work"

    # Get compatible models
    compatible = recommend_for_hardware(vram_gb=vram, ram_gb=ram)

    # Select top picks with reasons
    priority_order = ["popular", "fast", "high-quality", "coding-specialist", "reasoning-specialist"]
    recommendations = []
    seen_params = set()

    for model_id, meta in compatible.items():
        param_size = meta.get("params", "")
        if param_size in seen_params:
            continue
        seen_params.add(param_size)

        tags = meta.get("tags", [])
        strengths = meta.get("strengths", [])
        quant = meta.get("recommended_quant", "4bit")

        reason_parts = []
        if "popular" in tags:
            reason_parts.append("widely used")
        if "fast" in tags:
            reason_parts.append("fast inference")
        if "coding" in strengths:
            reason_parts.append("great at coding")
        if "reasoning" in strengths:
            reason_parts.append("strong reasoning")
        if "multilingual" in strengths:
            reason_parts.append("multilingual")

        reason = ", ".join(reason_parts) if reason_parts else "good general model"

        recommendations.append({
            "model_id": model_id,
            "name": meta["name"],
            "quant": quant,
            "reason": reason,
            "params": param_size,
        })

        if len(recommendations) >= 8:
            break

    return {
        "hardware": hw,
        "tier": tier,
        "headline": headline,
        "recommended_models": recommendations,
    }
