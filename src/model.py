"""
Phase 4 - Modelling under temporal validation.

We answer the project's core question: how much signal do the engineered
features add BEYOND grid position? Everything here is built to make that answer
honest.

Key design decisions (each is a talking point):

  * TEMPORAL validation, never a random split. We train on past seasons and
    test on a future one (expanding window: train <=2023 -> test 2024, then
    train <=2024 -> test 2025). A random split would let the model peek at the
    future and inflate every score.

  * TOP-3-PER-RACE prediction. Each race has exactly three podium slots, so for
    precision/recall/F1 we mark the three highest-probability drivers in each
    race as the predicted podium. This mirrors reality AND makes the comparison
    to the grid-only baseline (top-3 on the grid) perfectly apples-to-apples.

  * PR-AUC too. F1 depends on a threshold; PR-AUC (average precision) is
    threshold-free and uses the raw probabilities, so it captures ranking
    quality directly. Accuracy is never used (~85% by always saying "no podium").

  * Class imbalance handled explicitly (podium ~15%): balanced class weights for
    logistic regression / random forest, scale_pos_weight for XGBoost.

  * Models built up in sophistication: logistic regression -> random forest ->
    gradient boosting (XGBoost). We report all three against the baseline.

Run this via  scripts/run_models.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.features import FEATURE_COLS  # noqa: E402
from src.baseline import add_podium_label, evaluate, print_metrics  # noqa: E402

TEST_SEASONS = [2024, 2025]   # each is predicted using only earlier seasons


# --------------------------------------------------------------------------- #
#  Prediction helper
# --------------------------------------------------------------------------- #
def _top3_per_race(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Mark the 3 highest-scoring drivers in each race as predicted podium.

    Mirrors the fact that every race has exactly three podium places, and makes
    F1 directly comparable to the grid-only baseline.
    """
    tmp = df[["season", "round"]].copy()
    tmp["score"] = score
    tmp["rank"] = (tmp.groupby(["season", "round"])["score"]
                      .rank(ascending=False, method="first"))
    return (tmp["rank"] <= config.PODIUM_CUTOFF).astype(int).to_numpy()


# --------------------------------------------------------------------------- #
#  Model specs
# --------------------------------------------------------------------------- #
def _build_models(pos_weight: float) -> dict:
    """Each model as a ready-to-fit estimator.

    LogReg and RF get median imputation (they can't take NaN); LogReg also gets
    scaling. XGBoost handles NaN natively and uses scale_pos_weight for the
    imbalance.
    """
    models = {
        "logistic regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "random forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=5,
                class_weight="balanced", n_jobs=-1, random_state=0)),
        ]),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=0)
    except ImportError:
        print("  (xgboost not installed - skipping it)")
    return models


# --------------------------------------------------------------------------- #
#  Temporal evaluation
# --------------------------------------------------------------------------- #
def _predict_out_of_sample(model, df, X, y):
    """Expanding-window temporal CV: predict each TEST_SEASON using only earlier
    seasons. Returns pooled out-of-sample scores aligned to df rows."""
    scores = np.full(len(df), np.nan)
    for test_season in TEST_SEASONS:
        tr = df["season"] < test_season
        te = df["season"] == test_season
        if tr.sum() == 0 or te.sum() == 0:
            continue
        model.fit(X[tr], y[tr])
        if hasattr(model, "predict_proba"):
            scores[te.to_numpy()] = model.predict_proba(X[te])[:, 1]
        else:  # pragma: no cover
            scores[te.to_numpy()] = model.decision_function(X[te])
    return scores


def run(features_path=None) -> pd.DataFrame:
    features_path = features_path or (config.DATA_DIR / "features.parquet")
    df = pd.read_parquet(features_path)
    df = add_podium_label(df)

    # Only evaluate on the rows we can actually test (the TEST_SEASONS).
    test_mask = df["season"].isin(TEST_SEASONS)

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["podium"].to_numpy()

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    models = _build_models(pos_weight)

    results = []
    model_scores = {}   # keep per-model out-of-sample scores for surprise analysis

    # --- Baseline on the SAME test rows, for a fair comparison --------------
    grid_score = -df["grid"].fillna(df["grid"].max() + 1).to_numpy()
    base_pred = _top3_per_race(df, grid_score)
    m = evaluate(df.loc[test_mask, "podium"],
                 pd.Series(base_pred[test_mask.to_numpy()]),
                 pd.Series(grid_score[test_mask.to_numpy()]),
                 label="grid-only baseline")
    print_metrics(m)
    results.append(m)

    # --- Each model ---------------------------------------------------------
    for name, model in models.items():
        scores = _predict_out_of_sample(model, df, X, y)
        model_scores[name] = scores
        preds = _top3_per_race(df, np.nan_to_num(scores, nan=-1e9))
        tm = test_mask.to_numpy()
        m = evaluate(df.loc[test_mask, "podium"],
                     pd.Series(preds[tm]),
                     pd.Series(scores[tm]),
                     label=name)
        print_metrics(m)
        results.append(m)

    _comparison_table(results)
    _surprise_analysis(df, model_scores, test_mask)
    return pd.DataFrame(results)


def _surprise_analysis(df, model_scores, test_mask) -> None:
    """The question the project set out to answer: among TEST-season podiums that
    started OUTSIDE the top-3 grid, how many does each model recover? The
    grid-only baseline recovers ZERO by construction, so any recovery here is
    signal grid literally cannot express.
    """
    tm = test_mask.to_numpy()
    surprise = ((df["podium"] == 1) & (df["grid"] > config.PODIUM_CUTOFF)).to_numpy() & tm
    n = int(surprise.sum())

    print(f"\n--- surprise-podium analysis (test seasons {TEST_SEASONS}) ------")
    print(f"'surprise' podiums = finished top-{config.PODIUM_CUTOFF} from grid "
          f"> {config.PODIUM_CUTOFF}:  {n}")
    print(f"{'model':<22}{'recovered':>11}{'rate':>9}")
    print(f"{'grid-only baseline':<22}{0:>11}{'0.0%':>9}   (0 by construction)")
    for name, scores in model_scores.items():
        preds = _top3_per_race(df, np.nan_to_num(scores, nan=-1e9)).astype(bool)
        rec = int((preds & surprise).sum())
        rate = rec / n if n else 0.0
        print(f"{name:<22}{rec:>11}{rate:>8.1%}")
    print("----------------------------------------------------------------")
    print("Recovering any of these is genuine value beyond grid. The trade-off:")
    print("promoting a surprise driver means demoting a top-3 starter, which can")
    print("cost an 'obvious' podium - which is why aggregate F1 can stay flat.")


def _comparison_table(results: list[dict]) -> None:
    print("\n================ COMPARISON (test seasons "
          f"{TEST_SEASONS}) ================")
    print(f"{'model':<22}{'precision':>10}{'recall':>9}{'F1':>7}{'PR-AUC':>9}")
    base_f1 = results[0]["f1"]
    for m in results:
        delta = m["f1"] - base_f1
        tag = "" if m["label"] == "grid-only baseline" else f"  ({delta:+.3f} vs base)"
        print(f"{m['label']:<22}{m['precision']:>10.3f}{m['recall']:>9.3f}"
              f"{m['f1']:>7.3f}{m.get('pr_auc', float('nan')):>9.3f}{tag}")
    print("=" * 63)
    print("F1 uses top-3-per-race; PR-AUC is threshold-free. Positive delta means")
    print("the model adds signal beyond grid position under honest temporal CV.")


if __name__ == "__main__":
    run()