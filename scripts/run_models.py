"""Phase 4 entry point: temporal-CV modelling and comparison to the baseline."""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.model import run

if __name__ == "__main__":
    run()