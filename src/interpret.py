"""
Phase 5 - Interpretation with SHAP.

Phase 4 told us WHAT happened (grid dominates; the models recover some
come-from-behind podiums). This phase explains WHY, by attributing each
prediction to the features that drove it.

SHAP (SHapley Additive exPlanations) assigns every feature a signed contribution
to each prediction: positive = pushed the podium probability up, negative =
pushed it down. Averaging the magnitudes over all races gives a principled
feature-importance ranking; the beeswarm summary plot shows both importance and
direction at once.

We interpret all three models. The right SHAP explainer depends on the model:
  * tree models (random forest, xgboost) -> fast exact TreeExplainer
  * logistic regression                  -> LinearExplainer
For the pipeline models we explain the final estimator on the pre-processed
features (imputed / scaled), keeping the original feature names and order.

Outputs go to  reports/  (committed, so the README can show them):
  * one SHAP beeswarm summary plot per model
  * a printed importance ranking (mean |SHAP|) per model

Run this via  scripts/run_interpret.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # headless: save figures, don't try to display
import matplotlib.pyplot as plt
import shap

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.features import FEATURE_COLS  # noqa: E402
from src.baseline import add_podium_label  # noqa: E402
from src.model import TEST_SEASONS, _build_models  # noqa: E402

REPORTS_DIR = config.ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def _to_2d_positive(sv) -> np.ndarray:
    """Normalise SHAP output to a 2D (rows x features) array for the podium class.

    Different SHAP/estimator combinations return a list [class0, class1], a 3D
    array (rows, features, classes), or already-2D values. Handle all three.
    """
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:            # (rows, features, classes)
        sv = sv[:, :, -1]       # last class = podium (positive)
    return sv


def _shap_values(name, model, X_raw):
    """Return (shap_matrix, data_used) for a fitted model on the given rows."""
    if name == "xgboost":
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_raw)          # NaN handled natively
        return _to_2d_positive(sv), X_raw

    pre, clf = model[:-1], model[-1]               # pipeline: preprocessing + clf
    Xt = pre.transform(X_raw)
    if name == "random forest":
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
    else:                                          # logistic regression
        explainer = shap.LinearExplainer(clf, Xt)
        sv = explainer.shap_values(Xt)
    return _to_2d_positive(sv), Xt


def _importance_table(name, sv) -> pd.DataFrame:
    imp = (pd.DataFrame({"feature": FEATURE_COLS,
                         "mean_abs_shap": np.abs(sv).mean(axis=0)})
             .sort_values("mean_abs_shap", ascending=False)
             .reset_index(drop=True))
    print(f"\n=== {name}: feature importance (mean |SHAP|) ===")
    for _, r in imp.iterrows():
        bar = "#" * int(round(40 * r.mean_abs_shap / imp.mean_abs_shap.max()))
        print(f"  {r.feature:<26}{r.mean_abs_shap:8.4f}  {bar}")
    return imp


def _summary_plot(name, sv, data) -> Path:
    plt.figure()
    shap.summary_plot(sv, data, feature_names=FEATURE_COLS, show=False,
                      plot_size=(9, 6))
    plt.title(f"SHAP summary - {name}")
    plt.tight_layout()
    out = REPORTS_DIR / f"shap_summary_{name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return out


def run(features_path=None) -> None:
    features_path = features_path or (config.DATA_DIR / "features.parquet")
    df = pd.read_parquet(features_path)
    df = add_podium_label(df)

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["podium"].to_numpy()
    tr = (df["season"] < min(TEST_SEASONS)).to_numpy()
    te = df["season"].isin(TEST_SEASONS).to_numpy()

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    models = _build_models(pos_weight)

    for name, model in models.items():
        model.fit(X[tr], y[tr])                     # train on pre-test seasons
        sv, data = _shap_values(name, model, X[te]) # explain the test races
        _importance_table(name, sv)
        out = _summary_plot(name, sv, data)
        print(f"  saved plot -> {out}")

    # Bonus: logistic-regression coefficients (on standardized features) as an
    # independent cross-check of the SHAP story for the 'brave' model.
    if "logistic regression" in models:
        clf = models["logistic regression"][-1]
        coef = (pd.DataFrame({"feature": FEATURE_COLS, "coef": clf.coef_[0]})
                  .sort_values("coef", key=np.abs, ascending=False))
        print("\n=== logistic regression: standardized coefficients ===")
        print("(positive => raises podium odds; negative => lowers them)")
        for _, r in coef.iterrows():
            print(f"  {r.feature:<26}{r.coef:+8.3f}")

    print(f"\nAll SHAP plots saved in: {REPORTS_DIR}")


if __name__ == "__main__":
    run()