"""
Logistic regression yield model on a snapshot table, scored exactly like train_gbm.py.

Usage:
    python3 train_logit.py [snapshot_csv] [late_csv] [out_dir]
    python3 train_logit.py snapshot_0201.csv late_applicants_0201.csv logit_feb01/

Defaults: snapshot_0201.csv, late_applicants_0201.csv and logit_feb01/, all next to this script.

Needs train_gbm.py in the same folder: it reuses its loading, late-applicant add-on and
baselines so both models are compared on identical inputs. Test-optional is always removed
(same as running train_gbm.py with --drop-test-optional).

Design (mirrors the GBM):
    train 2022-24 -> pick regularization C + recalibrate on 2025 -> predict 2026 once.

Outputs (out_dir):
    predictions.csv     row_id, entry_year, p_enroll   (2025 + 2026)
    metrics.txt         AUC / Brier / class-size forecast vs actual vs baselines
    odds_ratios.csv     odds ratios from a lean, non-overlapping version of the model (for slides)
    odds_ratios.png     top-15 effects, both directions

Two models on purpose: the full 45-feature model makes the forecast (better by residency),
and a lean ~27-term model gives the odds ratios (each one means a single thing).
    calibration.png     predicted vs actual by decile, 2026
"""
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import train_gbm as G  # shared loading / late add-on / baselines

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

# plain-English labels for the odds-ratio slide (lean model)
LABELS = {
    "deposited_by_cut": "Deposited by cutoff", "residency_Resident": "Resident (vs bordering state)",
    "residency_Non-Contiguous": "Non-contiguous state (vs bordering state)",
    "visited_by_cut": "Visited campus by cutoff", "legacy": "Legacy",
    "hs_rank": "HS class rank (+1 SD)", "test_score": "Test score (+1 SD)",
    "days_app_to_start": "Applied earlier (+1 SD)", "days_since_deposit": "Deposited earlier (+1 SD)",
    "days_first_contact_to_start": "First contact earlier (+1 SD)", "log_miles": "Farther from campus (+1 SD, log)",
    "support_needs": "Academic support needs (+1 SD)", "app_fee_Paid": "Paid app fee (vs deferred)",
    "app_fee_Waived": "App fee waived (vs deferred)", "stealth": "Stealth applicant", "grad_expct": "Expects grad school",
    "stem_major": "STEM major", "undecided": "Undecided major", "coll_math": "College math credits (+1 SD)",
    "male": "Male", "urm": "URM", "age": "Age (+1 SD)",
}


def lean_design(df, train_mask):
    """Small, non-overlapping feature set so each odds ratio means one thing (for slides, not the forecast)."""
    d = pd.DataFrame(index=df.index)
    for c in ["days_since_deposit", "test_score", "hs_rank", "days_app_to_start",
              "days_first_contact_to_start", "log_miles", "coll_math", "age"]:
        d[c] = df[c].astype(float).fillna(df.loc[train_mask, c].median())
    d["support_needs"] = df[["ed_plan_asst", "wrtng_asst", "read_asst", "study_skll_asst", "math_asst"]].sum(axis=1)
    for c in ["deposited_by_cut", "visited_by_cut", "legacy", "grad_expct", "stem_major", "undecided", "stealth", "urm"]:
        d[c] = df[c]
    d["male"] = (df.sex == "M").astype(int)
    d = pd.concat([d, pd.get_dummies(df[["residency", "college", "app_fee"]].astype(str),
                                     drop_first=True, dtype=int)], axis=1)
    scale = [c for c in d if not set(d[c].unique()) <= {0, 1}]
    mu, sd = d.loc[train_mask, scale].mean(), d.loc[train_mask, scale].std().replace(0, 1)
    d[scale] = (d[scale] - mu) / sd
    return d
MISSING_FLAGS = ["hs_math", "hs_sci", "hs_ss", "hs_flang"]  # gaps here may mean something; scores excluded on purpose


def design(df, feats, train_mask):
    """Impute (train medians), flag missing coursework, one-hot categoricals, standardize on train."""
    num = [f for f in feats if f not in G.CATS]
    X = df[num].astype(float).copy()
    for c in MISSING_FLAGS:
        X[f"{c}_missing"] = X[c].isna().astype(int)
    X = X.fillna(X[train_mask].median())

    mu, sd = X[train_mask].mean(), X[train_mask].std().replace(0, 1)
    binary = [c for c in X if set(X[c].unique()) <= {0, 1}]
    scale = [c for c in X if c not in binary]
    X[scale] = (X[scale] - mu[scale]) / sd[scale]

    dummies = pd.get_dummies(df[[c for c in G.CATS if c in feats]].astype(str), drop_first=True, dtype=int)
    refs = {c: sorted(df[c].dropna().astype(str).unique())[0] for c in G.CATS if c in feats}
    return pd.concat([X, dummies], axis=1), scale, refs


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main(snapshot_csv, late_csv, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, feats = G.load(snapshot_csv, drop_test_optional=True)
    train = df.entry_year.isin(G.TRAIN)
    X, scaled, refs = design(df, feats, train)
    va_m, te_m = df.entry_year == G.VALID, df.entry_year == G.TEST
    y = df.enrolled

    # pick C on 2025 log loss
    scores = {}
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=5000).fit(X[train], y[train])
        scores[C] = log_loss(y[va_m], m.predict_proba(X[va_m])[:, 1])
    best_C = min(scores, key=scores.get)
    model = LogisticRegression(C=best_C, max_iter=5000).fit(X[train], y[train])

    # recalibrate on 2025 (Platt: refit intercept + slope on the model's log-odds)
    raw_va, raw_te = model.predict_proba(X[va_m])[:, 1], model.predict_proba(X[te_m])[:, 1]
    platt = LogisticRegression(C=1e6).fit(logit(raw_va).reshape(-1, 1), y[va_m])
    p_va = platt.predict_proba(logit(raw_va).reshape(-1, 1))[:, 1]
    p_te = platt.predict_proba(logit(raw_te).reshape(-1, 1))[:, 1]

    va, te = df[va_m], df[te_m].assign(p=p_te)
    pd.concat([
        pd.DataFrame({"row_id": va.row_id, "entry_year": G.VALID, "p_enroll": p_va}),
        pd.DataFrame({"row_id": te.row_id, "entry_year": G.TEST, "p_enroll": p_te}),
    ]).to_csv(out / "predictions.csv", index=False)

    # ---- metrics (same layout as train_gbm.py) ----
    late_est, late_act = G.late_addon(late_csv, df)
    by_res = te.groupby("residency", observed=True).agg(pool_pred=("p", "sum"), pool_actual=("enrolled", "sum"))
    by_res["late_pred"], by_res["late_actual"] = late_est, late_act
    by_res["pred"] = by_res.pool_pred + by_res.late_pred
    by_res["actual"] = by_res.pool_actual + by_res.late_actual
    by_res.loc["ALL"] = by_res.sum()
    by_res["error"] = by_res.pred - by_res.actual
    by_res["error_pct"] = (100 * by_res.error / by_res.actual).round(1)

    by_col = te.groupby("college", observed=True).agg(pool_pred=("p", "sum"), pool_actual=("enrolled", "sum"))
    by_col["error"] = by_col.pool_pred - by_col.pool_actual

    base, actual_total = G.baselines(df, late_csv)
    raw_total = raw_te.sum() + late_est.sum()
    lines = [
        f"features: {len(feats)} (+{len(MISSING_FLAGS)} missing flags, one-hot -> {X.shape[1]} columns)"
        f"   C: {best_C}   test_optional dropped: True",
        "",
        f"2026 student-level:  AUC {roc_auc_score(te.enrolled, p_te):.4f}   Brier {brier_score_loss(te.enrolled, p_te):.4f}"
        f"   (before recalibration: Brier {brier_score_loss(te.enrolled, raw_te):.4f}, total {raw_total:.1f})",
        f"2025 (tuning/calibration year, optimistic): AUC {roc_auc_score(va.enrolled, p_va):.4f}",
        "",
        "2026 class size by residency (pool = model, late = add-on):",
        by_res.round(1).to_string(),
        "",
        "2026 pool by college (model only, excludes late add-on):",
        by_col.round(1).to_string(),
        "",
        f"Baselines vs actual 2026 total ({actual_total}):",
        *[f"  {k:38s} {v:8.1f}   error {v - actual_total:+.1f}" for k, v in base.items()],
        f"  {'Logistic pool + late':38s} {by_res.loc['ALL', 'pred']:8.1f}   error {by_res.loc['ALL', 'error']:+.1f}",
    ]
    report = "\n".join(lines)
    (out / "metrics.txt").write_text(report)
    print(report)

    # ---- odds ratios from the lean model (the full model's overlapping columns split credit) ----
    Xl = lean_design(df, train)
    lean = LogisticRegression(C=1.0, max_iter=5000).fit(Xl[train], y[train])
    coef = pd.Series(lean.coef_[0], index=Xl.columns)
    ors = pd.DataFrame({
        "feature": coef.index,
        "label": [LABELS.get(c, c.replace("college_", "").replace(" and ", " & ") + " (vs Applied Sci & Eng)")
                  for c in coef.index],
        "odds_ratio": np.exp(coef.values).round(2),
    })
    ors["abs_log_or"] = np.abs(coef.values)
    ors = ors.sort_values("abs_log_or", ascending=False).drop(columns="abs_log_or")
    ors.to_csv(out / "odds_ratios.csv", index=False)

    top = ors.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    colors = ["#3f74d0" if v >= 1 else "#c8553d" for v in top.odds_ratio]
    plt.barh(top.label, top.odds_ratio - 1, left=1, color=colors)  # bars grow out from 1
    plt.axvline(1, color="#888", lw=1)
    plt.xscale("log")
    plt.xlabel("odds ratio (log scale; >1 raises odds of enrolling, <1 lowers)")
    plt.title("What changes the odds of enrolling\n(each factor with the others held fixed)")
    plt.tight_layout()
    plt.savefig(out / "odds_ratios.png", dpi=150)
    plt.close()

    # ---- calibration plot ----
    dec = pd.qcut(te.p, 10, duplicates="drop")
    cal = te.groupby(dec, observed=True).agg(pred=("p", "mean"), actual=("enrolled", "mean"))
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], color="#b8b7b2", lw=1)
    plt.plot(cal.pred, cal.actual, "o-", color="#3f74d0")
    plt.xlabel("predicted probability")
    plt.ylabel("actual enrollment rate")
    plt.title("Calibration, 2026 (deciles) - logistic")
    plt.tight_layout()
    plt.savefig(out / "calibration.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    here = Path(__file__).resolve().parent  # defaults work from the editor's Run button
    args = sys.argv[1:]
    main(
        args[0] if len(args) > 0 else here / "snapshot_0201.csv",
        args[1] if len(args) > 1 else here / "late_applicants_0201.csv",
        args[2] if len(args) > 2 else here / "logit_feb01",
    )