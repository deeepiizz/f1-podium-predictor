"""
Phase 1 - Data assembly.

Build one tidy table with ONE ROW PER DRIVER-RACE, for every race in the
configured season range. We pull two things from the Jolpica API (via FastF1's
Ergast wrapper): race results and qualifying.

Two real-world API wrinkles handled here:
  * Jolpica caps each response at 100 rows, so a full season spans several
    pages -> _iter_pages walks every page.
  * Jolpica is volunteer-run and rate-limits bursts (HTTP 429) -> every request
    goes through _call_with_retry, which backs off and retries.

Championship standings "going into" each race are derived later in Phase 3
(cumulative points, shifted) rather than fetched, to avoid extra API calls.

Run this via  scripts/build_dataset.py.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

import pandas as pd
import fastf1
import fastf1.exceptions as f1exc
from fastf1.ergast import Ergast

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


RESULT_COLS = [
    "number", "position", "positionText", "points", "grid", "laps", "status",
    "driverId", "driverCode", "givenName", "familyName", "driverNationality",
    "constructorId", "constructorName",
]
QUALI_COLS = [
    "position", "driverId", "Q1", "Q2", "Q3",
]
# Q1/Q2/Q3 are legitimately absent for some sessions (drivers knocked out early),
# so their absence should not raise a warning.
QUALI_OPTIONAL = {"Q1", "Q2", "Q3"}
DESC_COLS = ["season", "round", "raceName", "circuitId", "date"]


def _ergast() -> Ergast:
    fastf1.Cache.enable_cache(str(config.CACHE_DIR))
    # Jolpica caps page size at 100 rows, so we request 100 and paginate.
    return Ergast(result_type="pandas", auto_cast=True, limit=100)


def _call_with_retry(fn, *args, retries=6, base_delay=2.0, **kwargs):
    """Call an Ergast function, backing off and retrying on rate-limit (429).

    Non-rate-limit errors (and ValueError = 'no more pages') propagate normally.
    """
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except f1exc.ErgastError as e:
            is_rate = (isinstance(e, f1exc.RateLimitExceededError)
                       or "Too Many Requests" in str(e))
            if is_rate and attempt < retries - 1:
                wait = base_delay * (2 ** attempt)
                print(f"    rate limited - waiting {wait:.0f}s then retrying "
                      f"(attempt {attempt + 1}/{retries}) ...")
                time.sleep(wait)
                continue
            raise


def _iter_pages(resp, page_delay=0.5):
    """Yield (meta_row, content_df) across ALL pages of a paginated response."""
    while True:
        yield from zip(resp.description.itertuples(index=False), resp.content)
        if getattr(resp, "is_complete", True):
            break
        try:
            nxt = _call_with_retry(resp.get_next_result_page)
        except ValueError:
            break
        if nxt is None or len(nxt.content) == 0:
            break
        time.sleep(page_delay)   # be gentle between pages
        resp = nxt


def _safe_select(df: pd.DataFrame, wanted: list[str], label: str,
                 verbose: bool, optional: set[str] = frozenset()) -> pd.DataFrame:
    """Keep the wanted columns that exist; warn about unexpectedly-missing ones."""
    present = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns and c not in optional]
    if verbose:
        print(f"    [{label}] available columns: {list(df.columns)}")
    if missing:
        print(f"    [{label}] WARNING missing expected columns: {missing} "
              f"(FastF1 may have renamed them - adjust *_COLS in data_loader.py)")
    return df[present].copy()


def _load_season_results(erg: Ergast, season: int, verbose: bool) -> pd.DataFrame:
    """All race results for one season -> long frame with race metadata attached."""
    resp = _call_with_retry(erg.get_race_results, season=season)
    frames = []
    for meta_row, race_df in _iter_pages(resp):
        meta = {c: getattr(meta_row, c, pd.NA) for c in DESC_COLS}
        block = _safe_select(race_df, RESULT_COLS, "results",
                             verbose and not frames and season == config.SEASONS[0])
        for c in DESC_COLS:
            block[c] = meta[c]
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def _load_season_quali(erg: Ergast, season: int, verbose: bool) -> pd.DataFrame:
    """All qualifying results for one season -> long frame keyed by round+driver."""
    resp = _call_with_retry(erg.get_qualifying_results, season=season)
    frames = []
    for meta_row, q_df in _iter_pages(resp):
        rnd = getattr(meta_row, "round", pd.NA)
        block = _safe_select(q_df, QUALI_COLS, "qualifying",
                             verbose and not frames and season == config.SEASONS[0],
                             optional=QUALI_OPTIONAL)
        block = block.rename(columns={"position": "quali_position"})
        block["round"] = rnd
        block["season"] = season
        frames.append(block)
    if not frames:
        return pd.DataFrame(columns=["season", "round", "driverId"])
    return pd.concat(frames, ignore_index=True)


def build_driver_race_table(seasons: list[int] | None = None,
                            verbose: bool = True,
                            polite_delay: float = 1.0) -> pd.DataFrame:
    """Assemble the full driver-race table across all seasons and cache to parquet."""
    seasons = seasons or config.SEASONS
    erg = _ergast()

    all_results, all_quali = [], []
    for season in seasons:
        print(f"[season {season}] fetching results ...")
        res = _load_season_results(erg, season, verbose)
        all_results.append(res)
        time.sleep(polite_delay)

        print(f"[season {season}] fetching qualifying ...")
        qua = _load_season_quali(erg, season, verbose)
        all_quali.append(qua)
        time.sleep(polite_delay)

        print(f"[season {season}] results rows={len(res)} quali rows={len(qua)}")

    results = pd.concat(all_results, ignore_index=True)
    quali = pd.concat(all_quali, ignore_index=True)

    for df in (results, quali):
        for k in ("season", "round"):
            if k in df.columns:
                df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")

    # Guarantee the quali columns we merge on exist even if a whole season lacked
    # one (e.g. Q3), so the merge below never raises a KeyError.
    for col in ("quali_position", "Q1", "Q2", "Q3"):
        if col not in quali.columns:
            quali[col] = pd.NA

    merged = results.merge(
        quali[["season", "round", "driverId", "quali_position", "Q1", "Q2", "Q3"]],
        on=["season", "round", "driverId"],
        how="left",
        validate="one_to_one",
    )

    merged = _clean(merged)

    config.DRIVER_RACE_PARQUET.parent.mkdir(exist_ok=True)
    merged.to_parquet(config.DRIVER_RACE_PARQUET, index=False)
    print(f"\nSaved {len(merged)} driver-race rows -> {config.DRIVER_RACE_PARQUET}")
    _data_quality_report(merged)
    return merged


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light, honest cleaning. Every step is documented for the write-up."""
    df = df.copy()

    for col in ("position", "grid", "points", "laps", "quali_position"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "grid" in df.columns:
        df["pit_lane_start"] = (df["grid"] == 0).astype(int)

    if "status" in df.columns:
        s = df["status"].astype("string").fillna("")
        df["finished"] = (
            s.str.startswith("Finished") | s.str.contains(r"\+\d+ Lap")
        ).astype(int)

    df = df.sort_values(["season", "round", "position"], na_position="last")
    return df.reset_index(drop=True)


def _data_quality_report(df: pd.DataFrame) -> None:
    """Print the kind of note that belongs in the README's data-quality section."""
    print("\n--- data quality note ------------------------------------------")
    print(f"seasons           : {sorted(df['season'].dropna().unique().tolist())}")
    print(f"races             : {df.groupby('season')['round'].nunique().to_dict()}")
    print(f"rows (driver-race): {len(df)}")
    if "quali_position" in df:
        no_quali = df["quali_position"].isna().sum()
        print(f"rows w/o quali    : {no_quali} "
              f"(drivers who raced but have no qualifying row - penalties, "
              f"withdrawals; grid position is still available)")
    if "finished" in df:
        print(f"DNF rate          : {(1 - df['finished'].mean()):.1%}")
    print("----------------------------------------------------------------")


if __name__ == "__main__":
    build_driver_race_table()