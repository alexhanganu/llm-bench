# Contributing to llm-bench

## Adding a New Model

Edit `llm_bench/models/registry.py` and add an entry to `MODELS`:

```python
"your-model-id": {
    "name": "Human Readable Name",
    "family": "llama",           # or qwen, mistral, phi, gemma, deepseek, …
    "developer": "Org Name",
    "params": "7B",
    "context": 32768,
    "hf_repo": "org/repo-name",
    "gguf_repo": "bartowski/repo-name-GGUF",
    "gguf_file": "model-Q4_K_M.gguf",
    "min_vram_gb": {"fp16": 14, "8bit": 7, "4bit": 5},
    "min_ram_gb":  {"cpu": 14},
    "license": "Apache 2.0",
    "strengths": ["coding", "reasoning"],
    "tags": ["popular"],
},
```

Also add the entry to `configs/models.yaml` in the same format.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 35 tests must pass before opening a PR.

## Adding Pre-computed Results

Run benchmarks on your hardware and submit the JSON:

```bash
python scripts/precompute.py \
    --hardware rtx3080 \
    --all-small \
    --benchmarks speed mmlu \
    --output-dir data/precomputed
```

Then open a PR adding `data/precomputed/rtx3080.json`.

## Code Style

- Black formatting (`black .`)
- Type hints on all public functions
- Docstrings on all public classes/functions
