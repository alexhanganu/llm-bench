"""
Model downloader for llm-bench.

Supports:
  - Hugging Face Hub (full model repos via huggingface_hub)
  - Single GGUF files from HF repos via HTTP with progress bars
  - Resume interrupted downloads

Usage
-----
    from llm_bench.utils.download import download_model, download_gguf

    # Download GGUF (recommended – much smaller)
    path = download_gguf("llama-3.1-8b")

    # Download full HF repo (for transformers loading)
    snapshot = download_model("qwen2.5-7b")
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE = Path("~/.cache/llm-bench").expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def download_model(
    model_id: str,
    cache_dir: str | Path = _DEFAULT_CACHE / "hf",
    force: bool = False,
    token: Optional[str] = None,
) -> Path:
    """
    Download a full Hugging Face model repo (for transformers loading).

    Returns the local snapshot directory path.
    """
    from llm_bench.models.registry import get_model

    meta = get_model(model_id)
    hf_repo = meta["hf_repo"]
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading HF repo: %s → %s", hf_repo, cache)

    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=hf_repo,
            cache_dir=str(cache),
            token=token or os.environ.get("HF_TOKEN"),
            ignore_patterns=["*.bin", "*.pt"],  # prefer safetensors
        )
        return Path(local_dir)

    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required for full model downloads.\n"
            "Install it:  pip install huggingface_hub"
        )


def download_gguf(
    model_id: str,
    cache_dir: str | Path = _DEFAULT_CACHE / "gguf",
    force: bool = False,
    token: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """
    Download a single pre-quantised GGUF file from Hugging Face.

    Parameters
    ----------
    model_id          : registry model ID (e.g. "llama-3.1-8b")
    cache_dir         : local directory to save the file
    force             : re-download even if file already exists
    token             : HF token (also reads HF_TOKEN env var)
    progress_callback : called with (bytes_downloaded, total_bytes)

    Returns the local Path to the downloaded .gguf file.
    """
    from llm_bench.models.registry import get_model

    meta = get_model(model_id)
    gguf_repo = meta.get("gguf_repo")
    gguf_file = meta.get("gguf_file")

    if not gguf_repo or not gguf_file:
        raise ValueError(f"Model '{model_id}' does not have a GGUF entry in the registry.")

    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / gguf_file

    if dest.exists() and not force:
        size_gb = dest.stat().st_size / 1e9
        logger.info("GGUF already cached (%.2f GB): %s", size_gb, dest)
        return dest

    # Build URL
    hf_token = token or os.environ.get("HF_TOKEN", "")
    url = f"https://huggingface.co/{gguf_repo}/resolve/main/{gguf_file}"

    logger.info("Downloading GGUF: %s", url)
    _download_file(url, dest, hf_token=hf_token, progress_callback=progress_callback)
    return dest


def check_model_cached(model_id: str, backend: str = "gguf") -> bool:
    """Return True if the model is already downloaded locally."""
    from llm_bench.models.registry import get_model

    meta = get_model(model_id)
    if backend == "gguf":
        gguf_file = meta.get("gguf_file")
        if not gguf_file:
            return False
        dest = _DEFAULT_CACHE / "gguf" / gguf_file
        return dest.exists()
    else:
        # HF snapshot: look for config.json as a proxy
        hf_repo = meta["hf_repo"]
        model_name = hf_repo.split("/")[-1]
        hf_cache = _DEFAULT_CACHE / "hf"
        return any(hf_cache.rglob(f"{model_name}*/config.json"))


def get_download_size(model_id: str, quantization: str = "4bit") -> Optional[float]:
    """
    Return approximate download size in GB for the given quant level.
    Derived from min_vram_gb as a rough proxy (actual files are slightly smaller).
    """
    from llm_bench.models.registry import get_model

    meta = get_model(model_id)
    vram = meta.get("min_vram_gb", {}).get(quantization)
    if vram is None:
        return None
    # GGUF files are typically ~85-90% of VRAM requirement
    return round(vram * 0.88, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Rich-based CLI downloader (used by llm-bench download command)
# ─────────────────────────────────────────────────────────────────────────────

def download_with_progress(model_id: str, backend: str = "gguf") -> Path:
    """
    Download a model with a rich progress bar in the terminal.
    Suitable for use from the CLI.
    """
    try:
        from rich.progress import (
            Progress, BarColumn, DownloadColumn,
            TransferSpeedColumn, TimeRemainingColumn, TextColumn,
        )
        from rich.console import Console

        console = Console()
        from llm_bench.models.registry import get_model, MODELS

        meta = get_model(model_id)
        size_gb = get_download_size(model_id) or "?"
        console.print(
            f"\n[bold cyan]Downloading[/bold cyan] {meta['name']} "
            f"[dim]({size_gb} GB approx)[/dim]"
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(meta.get("gguf_file", model_id), total=None)

            def _cb(downloaded: int, total: int) -> None:
                progress.update(task, completed=downloaded, total=total or None)

            if backend == "gguf":
                path = download_gguf(model_id, progress_callback=_cb)
            else:
                path = download_model(model_id)

        console.print(f"[green]✓ Saved to {path}[/green]")
        return path

    except ImportError:
        # Fallback without rich
        logger.info("Downloading %s (no rich progress bar)…", model_id)
        if backend == "gguf":
            return download_gguf(model_id)
        return download_model(model_id)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_file(
    url: str,
    dest: Path,
    hf_token: str = "",
    chunk_size: int = 1 << 20,  # 1 MB
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Stream-download a file to dest with optional resume support.
    Uses a .part suffix during download to avoid corrupt partial files.
    """
    import urllib.request

    part = dest.with_suffix(dest.suffix + ".part")
    resume_pos = part.stat().st_size if part.exists() else 0

    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    if resume_pos:
        headers["Range"] = f"bytes={resume_pos}-"
        logger.info("Resuming download from byte %d", resume_pos)

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0)) + resume_pos
        mode = "ab" if resume_pos else "wb"
        downloaded = resume_pos

        with open(part, mode) as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(downloaded, total)

    # Atomic rename
    shutil.move(str(part), str(dest))
    logger.info("Download complete: %s (%.2f GB)", dest.name, dest.stat().st_size / 1e9)
