# scripts/paths.py
from pathlib import Path

# Project root = parent of 'scripts/'
ROOT = Path(__file__).resolve().parents[1]

def project_path(*parts: str) -> Path:
    """Join path parts to the project root."""
    return ROOT.joinpath(*parts)
