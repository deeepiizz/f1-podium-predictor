"""
Leakage test for the Phase 3 feature pipeline.

The idea: build features on synthetic race data, then CORRUPT the outcome
(finishing position / points / finished flag) of one specific race and rebuild.
Every pre-race feature for that race must be byte-for-byte identical, because a
feature computed correctly cannot depend on the race it is trying to predict.
If any pre-race feature changes, a future value leaked into the past.

Run from the project root with:   python -m tests.test_no_leakage
"""

import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import config
from src import features as F

PRE_RACE_COLS = [
    "driver_form_finish", "driver_points_last5", "driver_dnf_rate_last5",
    "driver_track_avg_finish",
    "driver_pts_before", "constructor_pts_before", "driver_rank_before",
]


def _synthetic_driver_race(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = {f"d{i}": f"t{(i - 1) // 2}" for i in range(1, 21)}
    circuits = [f"c{i}" for i in range(1, 6)]
    rows = []
    for season in (2022, 2023, 2024):
        for rnd in range(1, 11):
            circuit = circuits[(rnd - 1) % len(circuits)]
            grid = rng.permutation(np.arange(1, 21))
            for gi, drv in enumerate([f"d{i}" for i in range(1, 21)]):
                g = int(grid[gi])
                dnf = rng.random() < 0.12
                pos = np.nan if dnf else int(np.clip(round(g + rng.normal(0, 3)), 1, 20))
                pts = 0.0 if (dnf or pos > 10) else float(max(0, 26 - 2 * pos))
                base = 90000 + g * 120 + rng.normal(0, 50)
                rows.append(dict(
                    season=season, round=rnd, circuitId=circuit,
                    driverId=drv, constructorId=teams[drv],
                    grid=g, position=pos, points=pts,
                    status="Finished" if not dnf else "Accident",
                    finished=0 if dnf else 1, pit_lane_start=0,
                    Q1=pd.to_timedelta(base + 300, unit="ms"),
                    Q2=pd.to_timedelta(base + 100, unit="ms"),
                    Q3=pd.to_timedelta(base, unit="ms"),
                ))
    return pd.DataFrame(rows)


def test_no_leakage(target=(2023, 6)) -> bool:
    df = _synthetic_driver_race()
    p1 = config.DATA_DIR / "_leak_a.parquet"
    p2 = config.DATA_DIR / "_leak_b.parquet"
    df.to_parquet(p1, index=False)
    feat = F.build_features(in_path=p1, out_path=config.DATA_DIR / "_leak_fa.parquet")

    rng = np.random.default_rng(99)
    df2 = df.copy()
    m = (df2.season == target[0]) & (df2["round"] == target[1])
    for col in ("position", "points", "finished"):
        df2.loc[m, col] = rng.permutation(df2.loc[m, col].values)
    df2.to_parquet(p2, index=False)
    feat2 = F.build_features(in_path=p2, out_path=config.DATA_DIR / "_leak_fb.parquet")

    def race(f):
        return (f[(f.season == target[0]) & (f["round"] == target[1])]
                .set_index("driverId")[PRE_RACE_COLS].sort_index())

    before, after = race(feat), race(feat2)

    for p in ("_leak_a", "_leak_b", "_leak_fa", "_leak_fb"):
        (config.DATA_DIR / f"{p}.parquet").unlink(missing_ok=True)

    ok = before.equals(after)
    if ok:
        print("PASS - no pre-race feature changed when the race outcome was corrupted.")
    else:
        diff = (before.fillna(-999) != after.fillna(-999))
        leaked = [c for c in PRE_RACE_COLS if diff[c].any()]
        print(f"FAIL - these features leaked: {leaked}")
    return ok


if __name__ == "__main__":
    ok = test_no_leakage()
    sys.exit(0 if ok else 1)