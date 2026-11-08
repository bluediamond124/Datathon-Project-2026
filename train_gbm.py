"""
Gradient boosting (LightGBM) yield model on a snapshot table.

Usage:
    python3 train_gbm.py [snapshot_csv] [late_csv] [out_dir] [--drop-test-optional]
    python3 train_gbm.py snapshot_0201.csv late_applicants_0201.csv gbm_out/

Defaults: snapshot_0201.csv, late_applicants_0201.csv and gbm_out/, all next to this script.

Design:
    train 2022-24 -> early-stop + isotonic-calibrate on 2025 -> predict 2026 once.
    No refit on 2022-25: a refit model has no held-out year left to calibrate on,
    and calibrating on 2025 also pulls predictions toward the most recent yield level.

Outputs (out_dir):
    predictions_gbm.csv     row_id, entry_year, p_enroll   (2025 + 2026; same format as logistic)
    metrics_gbm.txt         AUC / Brier / class-size forecast vs actual vs baselines
    shap_importance.csv     mean |SHAP| per feature (2026 rows)
    shap_importance.png     top-15 bar chart
    calibration_gbm.png     predicted vs actual by decile, 2026
"""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore", category=lgb.basic.LGBMDeprecationWarning)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRAIN, VALID, TEST = [2022, 2023, 2024], 2025, 2026
KEYS = ["row_id", "entry_year", "snapshot_date", "enrolled"]
DROP = ["veteran", "first_generation", "visit_count_uncertain"]  # inconsistent across years
CATS = ["residency", "college", "app_fee", "sex", "distance_band"]

PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    n_estimators=3000, random_state=42, verbose=-1,
)


def load(snapshot_csv, drop_test_optional=False):
    df = pd.read_csv(snapshot_csv)
    drop = DROP + (["test_optional"] if drop_test_optional else [])
    feats = [c for c in df.columns if c not in KEYS + drop]
    if drop_test_optional:
        # blanks in score columns also reveal test-optional status, so fill them with train medians
        train = df.entry_year.isin(TRAIN)
        for c in ["test_score", "act_engl", "act_math"]:
            df[c] = df[c].fillna(df.loc[train, c].median())
    for c in CATS:
        if c in df:
            df[c] = pd.Categorical(df[c], categories=sorted(df[c].dropna().unique()))
    return df, feats


def late_addon(late_csv, df):
    """Late enrollees expected for TEST = last year's (late_enrolled / pool_admits) x this year's pool."""
    late = pd.read_csv(late_csv)
    late = late[late.residency != "ALL"]
    pool = df.groupby(["entry_year", "residency"]).size().rename("pool_admits").reset_index()
    late = late.merge(pool, on=["entry_year", "residency"])
    rate = late[late.entry_year == VALID].set_index("residency")
    pool26 = pool[pool.entry_year == TEST].set_index("residency").pool_admits
    est = (rate.late_enrolled / rate.pool_admits * pool26).round(1)
    actual = late[late.entry_year == TEST].set_index("residency").late_enrolled
    return est, actual


def baselines(df, late_csv):
    """Class-size baselines for TEST (pool + late), using only pre-TEST info."""
    late = pd.read_csv(late_csv)
    late_all = late[late.residency == "ALL"].set_index("entry_year")
    admits = df.groupby("entry_year").size() + late_all.late_admits
    enrolled = df.groupby("entry_year").enrolled.sum() + late_all.late_enrolled
    last_yield = enrolled[VALID] / admits[VALID]
    out = {"last year's yield x admits": last_yield * admits[TEST]}

    # deposited-by-cutoff split: each group's last-year yield x this year's group size
    v, t = df[df.entry_year == VALID], df[df.entry_year == TEST]
    r = v.groupby("deposited_by_cut").enrolled.mean()
    pool_pred = (t.deposited_by_cut.map(r)).sum()
    late_est, _ = late_addon(late_csv, df)
    out["deposit split (last year) + late"] = pool_pred + late_est.sum()
    return out, enrolled[TEST]


def main(snapshot_csv, late_csv, out_dir, drop_test_optional):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, feats = load(snapshot_csv, drop_test_optional)
    tr, va, te = (df[df.entry_year.isin(TRAIN)], df[df.entry_year == VALID], df[df.entry_year == TEST])

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(tr[feats], tr.enrolled, eval_set=[(va[feats], va.enrolled)], eval_metric="binary_logloss",
              categorical_feature=[c for c in CATS if c in feats],
              callbacks=[lgb.early_stopping(100, verbose=False)])

    raw_va = model.predict_proba(va[feats])[:, 1]
    raw_te = model.predict_proba(te[feats])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_va, va.enrolled)
    p_va, p_te = iso.predict(raw_va), iso.predict(raw_te)

    preds = pd.concat([
        pd.DataFrame({"row_id": va.row_id, "entry_year": VALID, "p_enroll": p_va}),
        pd.DataFrame({"row_id": te.row_id, "entry_year": TEST, "p_enroll": p_te}),
    ])
    preds.to_csv(out / "predictions_gbm.csv", index=False)

    # ---- metrics ----
    late_est, late_act = late_addon(late_csv, df)
    te = te.assign(p=p_te)
    by_res = te.groupby("residency", observed=True).agg(pool_pred=("p", "sum"), pool_actual=("enrolled", "sum"))
    by_res["late_pred"], by_res["late_actual"] = late_est, late_act
    by_res["pred"] = by_res.pool_pred + by_res.late_pred
    by_res["actual"] = by_res.pool_actual + by_res.late_actual
    by_res.loc["ALL"] = by_res.sum()
    by_res["error"] = by_res.pred - by_res.actual
    by_res["error_pct"] = (100 * by_res.error / by_res.actual).round(1)

    by_col = te.groupby("college", observed=True).agg(pool_pred=("p", "sum"), pool_actual=("enrolled", "sum"))
    by_col["error"] = by_col.pool_pred - by_col.pool_actual

    base, actual_total = baselines(df, late_csv)
    lines = [
        f"features: {len(feats)}   trees: {model.best_iteration_}   test_optional dropped: {drop_test_optional}",
        "",
        f"2026 student-level:  AUC {roc_auc_score(te.enrolled, p_te):.4f}   Brier {brier_score_loss(te.enrolled, p_te):.4f}"
        f"   (uncalibrated Brier {brier_score_loss(te.enrolled, raw_te):.4f})",
        f"2025 (calibration year, optimistic): AUC {roc_auc_score(va.enrolled, p_va):.4f}",
        "",
        "2026 class size by residency (pool = model, late = add-on):",
        by_res.round(1).to_string(),
        "",
        "2026 pool by college (model only, excludes late add-on):",
        by_col.round(1).to_string(),
        "",
        f"Baselines vs actual 2026 total ({actual_total}):",
        *[f"  {k:38s} {v:8.1f}   error {v - actual_total:+.1f}" for k, v in base.items()],
        f"  {'GBM pool + late':38s} {by_res.loc['ALL', 'pred']:8.1f}   error {by_res.loc['ALL', 'error']:+.1f}",
    ]
    report = "\n".join(lines)
    (out / "metrics_gbm.txt").write_text(report)
    print(report)

    # ---- SHAP (LightGBM native, no shap package needed) ----
    contrib = model.booster_.predict(te[feats], pred_contrib=True)[:, :-1]
    imp = (pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(contrib).mean(axis=0)})
             .sort_values("mean_abs_shap", ascending=False))
    imp.to_csv(out / "shap_importance.csv", index=False)
    top = imp.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top.feature, top.mean_abs_shap, color="#3f74d0")
    plt.xlabel("mean |SHAP| (log-odds)")
    plt.title("What drives the GBM's predictions (2026)")
    plt.tight_layout()
    plt.savefig(out / "shap_importance.png", dpi=150)
    plt.close()

    # ---- calibration plot ----
    dec = pd.qcut(te.p, 10, duplicates="drop")
    cal = te.groupby(dec, observed=True).agg(pred=("p", "mean"), actual=("enrolled", "mean"))
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], color="#b8b7b2", lw=1)
    plt.plot(cal.pred, cal.actual, "o-", color="#3f74d0")
    plt.xlabel("predicted probability")
    plt.ylabel("actual enrollment rate")
    plt.title("Calibration, 2026 (deciles)")
    plt.tight_layout()
    plt.savefig(out / "calibration_gbm.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    here = Path(__file__).resolve().parent  # defaults work from the editor's Run button
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if len(args) > 0 else here / "snapshot_0201.csv",
        args[1] if len(args) > 1 else here / "late_applicants_0201.csv",
        args[2] if len(args) > 2 else here / "gbm_out",
        "--drop-test-optional" in sys.argv,
    )