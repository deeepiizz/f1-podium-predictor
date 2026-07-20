"""Phase 1 entry point: build and cache the driver-race table."""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_loader import build_driver_race_table

if __name__ == "__main__":
    build_driver_race_table()