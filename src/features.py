"""
Phase 3 - Feature engineering.

Turn the raw driver-race table into a model-ready feature table. THE governing
rule of this whole module: every feature for a given race must be computable
strictly from information available BEFORE that race starts. If a feature could
peek at the race it is attached to (or any later race), that is leakage, and it
would make the model look far better than it really is.

The mechanics that enforce this:
  * data is sorted chronologically (season, round) before any rolling/cumulative op
  * rolling driver/constructor "form" uses .shift(1) so the current race is never
    inside its own window
  * season-cumulative points use (cumsum - current) so the current round is excluded
  * championship-position "before the round" is computed on a per-round grid so a
    teammate's result in the SAME race can never leak in
  * historical track affinity uses an expanding mean that is shifted by one

Run this via  scripts/build_features.py.  It reads data/driver_race.parquet and
writes data/features.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


FORM_WINDOW = 5   # "recent form" = last 5 races


# --------------------------------------------------------------------------- #
#  Small building blocks
# --------------------------------------------------------------------------- #
def _chrono(df: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically. Everything downstream assumes this order."""
    return df.sort_values(["season", "round", "position"],
                          na_position="last").reset_index(drop=True)


def _best_quali_ms(df: pd.DataFrame) -> pd.Series:
    """Each driver's best qualifying lap (min of Q1/Q2/Q3) in milliseconds.

    Q1/Q2/Q3 arrive as timedeltas; some are missing (knocked out early / no
    session), so we take the min across whatever is present.
    """
    cols = [c for c in ("Q1", "Q2", "Q3") if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    times = df[cols].apply(pd.to_timedelta, errors="coerce")
    best = times.min(axis=1)                       # fastest of the three
    return best.dt.total_seconds() * 1000.0


def _standings_before(df: pd.DataFrame, entity: str, out_col: str) -> pd.Series:
    """Championship points an entity (driver or constructor) had BEFORE each round.

    Built on a deduplicated (entity, season, round) grid so that, for a
    constructor's two cars, neither car's CURRENT-round points leak in. Points
    are summed per round and cumulated within the season EXCLUDING the current
    round (cumsum - current).

    The result is returned aligned to df BY POSITION (not by index label): the
    caller's df may have been re-sorted, so we map back via an explicit row
    counter rather than relying on pandas index alignment, which would silently
    scatter values onto the wrong rows.
    """
    per_round = (df.groupby([entity, "season", "round"], as_index=False)["points"]
                   .sum()
                   .sort_values([entity, "season", "round"]))
    grp = per_round.groupby([entity, "season"])
    # cumulative within the season, minus this round => strictly-before total
    per_round[out_col] = grp["points"].cumsum() - per_round["points"]

    left = df[[entity, "season", "round"]].copy()
    left["_row"] = np.arange(len(df))
    merged = (left.merge(per_round[[entity, "season", "round", out_col]],
                         on=[entity, "season", "round"], how="left")
                  .sort_values("_row"))
    return pd.Series(merged[out_col].to_numpy(), index=df.index)


# --------------------------------------------------------------------------- #
#  Feature groups
# --------------------------------------------------------------------------- #
def _driver_form(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling driver form over the previous FORM_WINDOW races (leakage-safe)."""
    df = df.sort_values(["driverId", "season", "round"]).copy()
    g = df.groupby("driverId")

    # shift(1) => the current race is excluded from its own window
    df["driver_form_finish"] = g["position"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
    df["driver_points_last5"] = g["points"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).sum())
    dnf = 1 - df["finished"]
    df["driver_dnf_rate_last5"] = (
        dnf.groupby(df["driverId"])
           .transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()))
    return df


def _driver_track_affinity(df: pd.DataFrame) -> pd.DataFrame:
    """Driver's historical average finish at THIS circuit, before today."""
    df = df.sort_values(["driverId", "circuitId", "season", "round"]).copy()
    df["driver_track_avg_finish"] = (
        df.groupby(["driverId", "circuitId"])["position"]
          .transform(lambda s: s.shift(1).expanding().mean()))
    return df


def _quali_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Qualifying pace features, all knowable before the race."""
    df = df.copy()
    df["best_quali_ms"] = _best_quali_ms(df)

    # Gap to pole within each race (pole = fastest best_quali that race)
    pole = df.groupby(["season", "round"])["best_quali_ms"].transform("min")
    df["quali_gap_to_pole_ms"] = df["best_quali_ms"] - pole

    # Gap to teammate: driver's best minus the fastest OTHER car of same team.
    team_min = df.groupby(["season", "round", "constructorId"])["best_quali_ms"]
    # min of the team; if a driver IS the team's min, compare to the other car
    df["_team_best"] = team_min.transform("min")
    df["_team_second"] = team_min.transform(
        lambda s: s.nsmallest(2).max() if s.notna().sum() >= 2 else np.nan)
    is_team_fastest = df["best_quali_ms"] <= df["_team_best"]
    teammate_ref = np.where(is_team_fastest, df["_team_second"], df["_team_best"])
    df["quali_gap_to_teammate_ms"] = df["best_quali_ms"] - teammate_ref
    df = df.drop(columns=["_team_best", "_team_second"])
    return df


def _standings(df: pd.DataFrame) -> pd.DataFrame:
    """Championship points (driver & constructor) going into each round."""
    df = df.copy()
    df["driver_pts_before"] = _standings_before(df, "driverId", "driver_pts_before")
    df["constructor_pts_before"] = _standings_before(
        df, "constructorId", "constructor_pts_before")
    # Championship RANK before the round (1 = leader), computed within each round.
    df["driver_rank_before"] = (
        df.groupby(["season", "round"])["driver_pts_before"]
          .rank(ascending=False, method="min"))
    return df


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
FEATURE_COLS = [
    "grid", "pit_lane_start",
    "quali_gap_to_pole_ms", "quali_gap_to_teammate_ms",
    "driver_form_finish", "driver_points_last5", "driver_dnf_rate_last5",
    "driver_track_avg_finish",
    "driver_pts_before", "constructor_pts_before", "driver_rank_before",
    "round",
]


def build_features(in_path=config.DRIVER_RACE_PARQUET,
                   out_path=None) -> pd.DataFrame:
    df = pd.read_parquet(in_path)
    df = _chrono(df)

    df = _driver_form(df)
    df = _driver_track_affinity(df)
    df = _quali_gaps(df)
    df = _standings(df)

    df = _chrono(df)   # restore canonical order after the per-group sorts

    out_path = out_path or (config.DATA_DIR / "features.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved features -> {out_path}")
    _feature_report(df)
    return df


def _feature_report(df: pd.DataFrame) -> None:
    print("\n--- feature summary --------------------------------------------")
    print(f"rows                 : {len(df)}")
    print(f"feature columns      : {len(FEATURE_COLS)}")
    for c in FEATURE_COLS:
        if c in df.columns:
            miss = df[c].isna().mean()
            print(f"  {c:<26} missing={miss:5.1%}")
    print("----------------------------------------------------------------")
    print("Note: early-career / first-of-season rows have missing rolling")
    print("features by design (no prior races to look back on).")


if __name__ == "__main__":
    build_features()
    