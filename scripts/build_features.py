"""Phase 3 entry point: build and cache the leakage-safe feature table."""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.features import build_features

if __name__ == "__main__":
    build_features()