"""
llm-bench CLI
Usage:
  llm-bench benchmark --model llama-3.1-8b --quant 4bit
  llm-bench download  --model qwen2.5-7b
  llm-bench hardware
  llm-bench list
"""

import click
import json
import sys

from llm_bench.models.registry import MODELS


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """🦙 llm-bench — Compare local LLMs on your hardware."""


@cli.command()
@click.option("--model", "-m", required=True, help="Model ID (e.g. llama-3.1-8b)")
@click.option("--quant", "-q", default="4bit",
              type=click.Choice(["fp16", "8bit", "4bit", "gguf-q4"]))
@click.option("--benchmarks", "-b", default="speed,mmlu",
              help="Comma-separated benchmarks: speed,mmlu,humaneval,truthfulqa")
@click.option("--num-samples", default=50, help="Samples for quality benchmarks")
@click.option("--output", "-o", default=None, help="Save results to JSON file")
def benchmark(model, quant, benchmarks, num_samples, output):
    """Run benchmarks on a specified model."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    if model not in MODELS:
        console.print(f"[red]Model '{model}' not found.[/red]")
        console.print(f"Available: {', '.join(list(MODELS.keys())[:5])}…")
        sys.exit(1)

    console.print(f"[bold cyan]🦙 llm-bench[/bold cyan] | {MODELS[model]['name']} | {quant}")
    console.print()

    from llm_bench.models.loader import ModelLoader, Quantization
    quant_map = {"fp16": Quantization.FP16, "8bit": Quantization.INT8,
                 "4bit": Quantization.INT4, "gguf-q4": Quantization.GGUF_Q4}

    loader = ModelLoader()
    with console.status(f"Loading {model}…"):
        load_result = loader.load(model, quantization=quant_map[quant])

    if not load_result.success:
        console.print(f"[red]Failed to load: {load_result.error}[/red]")
        sys.exit(1)

    results = {"model_id": model, "quantization": quant}
    bench_list = [b.strip() for b in benchmarks.split(",")]

    if "speed" in bench_list:
        from llm_bench.benchmarks.speed import benchmark_speed
        with console.status("Running speed benchmark…"):
            sr = benchmark_speed(load_result.model, load_result.tokenizer,
                                 model_id=model, quantization=quant)
        results["speed"] = sr.to_dict()
        console.print(f"  ⚡ Speed: [green]{sr.tokens_per_second:.1f}[/green] tok/s  "
                      f"TTFT: {sr.time_to_first_token_ms:.0f}ms  "
                      f"VRAM Δ: {sr.memory_delta_gb:.2f} GB")

    quality_benches = [b for b in bench_list if b in ("mmlu", "humaneval", "truthfulqa")]
    if quality_benches:
        from llm_bench.benchmarks.quality import run_quality_suite
        with console.status(f"Running quality benchmarks: {quality_benches}…"):
            qrs = run_quality_suite(load_result.model, load_result.tokenizer,
                                    model, quant, quality_benches, num_samples)
        for name, qr in qrs.items():
            results[name] = qr.to_dict()
            console.print(f"  🎯 {name.upper()}: [green]{qr.pct}[/green] "
                          f"({qr.correct}/{qr.total})")

    loader.unload(load_result)

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[dim]Results saved to {output}[/dim]")


@cli.command()
def hardware():
    """Detect and display hardware info."""
    from llm_bench.utils.hardware_detect import detect_hardware, get_model_recommendations
    from rich.console import Console

    console = Console()
    hw = detect_hardware()
    console.print("\n[bold]🖥️  Hardware Profile[/bold]")
    console.print(hw.summary())

    recs = get_model_recommendations(hw)
    console.print(f"\n[bold]💡 {recs['headline']}[/bold]")
    console.print("\nRecommended models:")
    for r in recs["recommended_models"][:6]:
        console.print(f"  • [cyan]{r['name']}[/cyan] ({r['quant']}) — {r['reason']}")


@cli.command("list")
@click.option("--family", default=None)
@click.option("--max-vram", default=None, type=float, help="Filter by max VRAM (GB)")
def list_models(family, max_vram):
    """List all available models."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    models = MODELS

    if family:
        models = {k: v for k, v in models.items() if v["family"] == family}
    if max_vram is not None:
        from llm_bench.models.registry import list_models_by_vram
        models = {k: v for k, v in list_models_by_vram(max_vram).items() if k in models}

    table = Table(title=f"🦙 Available Models ({len(models)})")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Params", justify="right")
    table.add_column("Context")
    table.add_column("VRAM 4-bit", justify="right")
    table.add_column("License")

    for mid, meta in models.items():
        table.add_row(
            mid,
            meta["name"],
            meta["params"],
            f"{meta['context']//1024}K",
            f"{meta['min_vram_gb'].get('4bit','?')} GB",
            meta.get("license", "?"),
        )

    console.print(table)


@cli.command()
@click.argument("model_id")
@click.option("--backend", default="gguf", type=click.Choice(["gguf", "hf"]),
              help="Download format: gguf (single file) or hf (full repo)")
@click.option("--force", is_flag=True, help="Re-download even if cached")
@click.option("--token", default=None, envvar="HF_TOKEN", help="HuggingFace token")
def download(model_id, backend, force, token):
    """Download a model (GGUF or full HF repo)."""
    from llm_bench.utils.download import download_with_progress, check_model_cached
    from rich.console import Console

    console = Console()

    if model_id not in MODELS:
        console.print(f"[red]Unknown model: {model_id}[/red]")
        sys.exit(1)

    if not force and check_model_cached(model_id, backend=backend):
        console.print(f"[yellow]Already cached. Use --force to re-download.[/yellow]")
        return

    path = download_with_progress(model_id, backend=backend)
    console.print(f"[green]✓ {path}[/green]")


@cli.command()
def dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess, sys
    from pathlib import Path
    app = Path(__file__).parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)])


if __name__ == "__main__":
    cli()
