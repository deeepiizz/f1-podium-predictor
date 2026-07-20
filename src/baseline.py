"""
Phase 2 - Labels and the grid-only baseline.

Two jobs:
  1. Create the podium label (finishing position <= PODIUM_CUTOFF).
  2. Build the baseline every ML model must beat: "predict podium if the driver
     started in the top 3 on the grid." Report precision / recall / F1 / PR-AUC.

Why this baseline matters: grid position alone predicts the podium remarkably
well, so the real research question of the whole project is "how much signal
can I add BEYOND where they started?" These numbers are the bar.

This module is deliberately free of any network / FastF1 dependency: it works
on the parquet table produced in Phase 1, so it can be tested on synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def add_podium_label(df: pd.DataFrame,
                     cutoff: int = config.PODIUM_CUTOFF) -> pd.DataFrame:
    """
    podium = 1 if finishing position <= cutoff, else 0.

    Retirements (NaN finishing position) become 0: they did not podium. This is
    the DNF policy we committed to in config.py.
    """
    df = df.copy()
    df["podium"] = (df["position"] <= cutoff).fillna(False).astype(int)
    return df


def grid_only_prediction(df: pd.DataFrame,
                         cutoff: int = config.PODIUM_CUTOFF) -> pd.Series:
    """Baseline rule: start in the top `cutoff` on the grid -> predict podium.

    Grid 0 (pit-lane start) is treated as NOT a top-grid slot.
    """
    grid = df["grid"]
    return ((grid >= 1) & (grid <= cutoff)).astype(int)


def evaluate(y_true: pd.Series, y_pred: pd.Series,
             y_score: pd.Series | None = None,
             label: str = "model") -> dict:
    """Compute the metrics that actually matter for a ~15% positive class.

    Accuracy is intentionally omitted: predicting "never podium" already scores
    ~85%, so accuracy is misleading here. We report precision / recall / F1 and,
    when a score is available, PR-AUC (average precision).
    """
    metrics = {
        "label": label,
        "base_rate": float(y_true.mean()),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))
    return metrics


def print_metrics(m: dict) -> None:
    print(f"\n=== {m['label']} ===")
    print(f"positive base rate : {m['base_rate']:.1%}  (podiums / all starters)")
    print(f"precision          : {m['precision']:.3f}  "
          f"(of predicted podiums, how many were right)")
    print(f"recall             : {m['recall']:.3f}  "
          f"(of real podiums, how many we caught)")
    print(f"F1                 : {m['f1']:.3f}")
    if "pr_auc" in m:
        print(f"PR-AUC             : {m['pr_auc']:.3f}")
    print(f"confusion          : TP={m['tp']} FP={m['fp']} "
          f"FN={m['fn']} TN={m['tn']}")


def describe_podium_provenance(df: pd.DataFrame,
                               cutoff: int = config.PODIUM_CUTOFF) -> None:
    """A descriptive stat that frames the baseline: what share of real podiums
    were achieved from a top-3 grid slot? (Shows why grid is so strong.)"""
    pod = df[df["podium"] == 1]
    if len(pod) == 0:
        return
    from_top = ((pod["grid"] >= 1) & (pod["grid"] <= cutoff)).mean()
    print(f"\n{from_top:.1%} of actual podiums came from a top-{cutoff} "
          f"grid slot (n={len(pod)} podium finishes).")


def run(parquet_path=config.DRIVER_RACE_PARQUET) -> dict:
    """End-to-end Phase 2 on the assembled table."""
    df = pd.read_parquet(parquet_path)
    df = add_podium_label(df)
    describe_podium_provenance(df)

    y_true = df["podium"]
    y_pred = grid_only_prediction(df)
    # For the baseline, the "score" is just -grid so a lower start ranks higher.
    y_score = -df["grid"].fillna(df["grid"].max() + 1)

    m = evaluate(y_true, y_pred, y_score, label="grid-only baseline")
    print_metrics(m)
    return m


if __name__ == "__main__":
    run()