"""
Shared pytest fixtures for llm-bench tests.
"""
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
