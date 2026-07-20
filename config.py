"""
Central configuration for the F1 podium predictor.

All scope decisions live here so the project has a single source of truth:

  * TARGET     : podium = finishing position <= 3  (binary classification)
  * SEASONS    : 2018-2024. Recent enough to be relevant, long enough to give
                 ~150 races of training data.
  * DNF policy : retirements are kept in and labelled non-podium. Random
                 mechanical failures are irreducible noise, and we say so.
"""

from pathlib import Path

# --- Scope ------------------------------------------------------------------
SEASONS = list(range(2018, 2026))   # 2018, 2019, ..., 2025 (inclusive)
PODIUM_CUTOFF = 3                    # top-N finishing position counts as podium

# --- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "fastf1_cache"                     # FastF1's HTTP cache
DRIVER_RACE_PARQUET = DATA_DIR / "driver_race.parquet"    # our assembled table

# Ensure directories exist on import
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)