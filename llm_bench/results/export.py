"""
Export benchmark results to JSON and CSV.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def export_json(results: List[Dict[str, Any]], path: str | Path, indent: int = 2) -> Path:
    """Write a list of result dicts to a JSON file."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(results, f, indent=indent, default=str)
    return dest


def export_csv(results: List[Dict[str, Any]], path: str | Path) -> Path:
    """Write a list of result dicts to a CSV file."""
    if not results:
        raise ValueError("results list is empty")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(dest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return dest


def results_to_markdown_table(results: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    """Render results as a GitHub-flavoured Markdown table."""
    if not results:
        return "_No results_"
    cols = columns or list(results[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = []
    for r in results:
        row = "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        rows.append(row)
    return "\n".join([header, sep] + rows)
