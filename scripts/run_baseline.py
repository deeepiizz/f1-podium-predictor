"""Phase 2 entry point: labels + grid-only baseline metrics."""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.baseline import run

if __name__ == "__main__":
    run()