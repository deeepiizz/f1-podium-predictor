"""Phase 5 entry point: SHAP interpretation of all trained models."""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.interpret import run

if __name__ == "__main__":
    run()