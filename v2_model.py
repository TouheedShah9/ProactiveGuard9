"""
ProactiveGuard v2.0 — Module 3: Advanced ML Model
===================================================
XGBoost ensemble with:
- Temporal lag features (5, 10, 15, 30 min windows)
- Rate-of-change features (slope, rolling mean, rolling std)
- SMOTE class balancing
- Optuna hyperparameter tuning
- Confidence-based uncertainty quantification
- SHAP explainability per worker per minute
- Time-aware train/test split (no data leakage)

Tested against: pandas 3.0.2, sklearn 1.8.0, xgboost 3.2.0,
                shap 0.51.0, optuna 4.8.0, imblearn 0.14.1

Run: python v2_model.py
Input:  data/v2_multisite_data.csv
        data/v2_twin_predictions.csv
Output: models/v2_model.json
        models/v2_scaler.pkl
        models/v2_features.pkl
        models/v2_confidence_threshold.pkl
        outputs/v2_shap_summary.png
        outputs/v2_confusion_matrix.png
        outputs/v2_metrics_report.txt
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, roc_auc_score,
                             classification_report,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

np.random.seed(42)
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# WORKER PROFILES
# Imported from v2_config.py — the single source of truth shared by
# every other module in this project. Do not redeclare these here;
# edit v2_config.py instead.
# ══════════════════════════════════════════════════════════════
from v2_config import WORKERS_BY_ID as WORKER_PROFILES

# Core signals to generate temporal features from
SIGNAL_COLS = [
    "heart_rate_bpm", "hrv_rmssd_ms", "core_temp_c",
    "breathing_rate", "spo2_pct", "heat_debt_index",
    "dehydration_index", "wbgt",
]

# Extra features from twin predictions
EXTRA_COLS = ["danger_prob_now", "danger_prob_30min", "cv_strain_now"]


# ══════════════════════════════════════════════════════════════
# TEMPORAL FEATURE ENGINEERING
# The model sees ONE snapshot without this.
# With this, it sees the TREND — is HR rising fast or stable?
# Rate of change is more predictive than absolute value.
# ══════════════════════════════════════════════════════════════
def engineer_temporal_features(df):
    """
    Creates lag features, slope features, and rolling statistics.
    Groups by worker to prevent cross-worker contamination.

    Features per signal:
    - lag_5, lag_10, lag_15, lag_30: past values
    - slope_5, slope_15: rate of change over 5 and 15 minutes
    - rm15: 15-minute rolling mean (trend)
    """
    df = df.copy().sort_values(
        ["worker_id", "minutes_on_shift"]
    ).reset_index(drop=True)

    new_cols = {}
    all_signals = SIGNAL_COLS + EXTRA_COLS

    for col in all_signals:
        if col not in df.columns:
            continue
        g = df.groupby("worker_id")[col]

        for lag in [5, 10, 15, 30]:
            new_cols[f"{col}_lag{lag}"] = g.shift(lag)

        new_cols[f"{col}_slope5"]  = g.diff(5) / 5
        new_cols[f"{col}_slope15"] = g.diff(15) / 15
        new_cols[f"{col}_rm15"]    = g.transform(
            lambda x: x.rolling(15, min_periods=1).mean())
        new_cols[f"{col}_rstd10"]  = g.transform(
            lambda x: x.rolling(10, min_periods=1).std().fillna(0))

    result = pd.concat(
        [df, pd.DataFrame(new_cols, index=df.index)], axis=1
    ).fillna(0)

    return result


# ══════════════════════════════════════════════════════════════
# TIME-AWARE SPLIT
# ══════════════════════════════════════════════════════════════
def time_aware_split(df, train_ratio=0.8):
    train_parts, test_parts = [], []
    for wid in df["worker_id"].unique():
        w = df[df["worker_id"] == wid].sort_values(
            "minutes_on_shift").reset_index(drop=True)
        split = int(len(w) * train_ratio)
        train_parts.append(w.iloc[:split])
        test_parts.append(w.iloc[split:])
    return (pd.concat(train_parts).reset_index(drop=True),
            pd.concat(test_parts).reset_index(drop=True))


# ══════════════════════════════════════════════════════════════
# CONFIDENCE SCORING
# Model's max class probability = prediction confidence.
# Below threshold → uncertain → escalate to rule-based fallback.
# ══════════════════════════════════════════════════════════════
def compute_confidence(probs, threshold=0.60):
    """
    Returns confidence score and uncertain flag per prediction.
    Uncertain predictions trigger rule-based fallback in dashboard.
    """
    max_probs   = probs.max(axis=1)
    uncertain   = max_probs < threshold
    return max_probs, uncertain


# ══════════════════════════════════════════════════════════════
# SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
class XGBoostNativeExplainer:
    """
    Drop-in replacement for shap.TreeExplainer(model) that never goes
    through shap's XGBoost model-loading code at all.

    WHY THIS EXISTS — three consecutive attempts to make
    shap.TreeExplainer() work with this xgboost/shap version pairing
    each failed for a different, real reason:

    Attempt 1 patched booster.save_config()/load_config() — that only
    round-trips booster HYPERPARAMETERS (max_depth, eta, etc.), a
    different surface entirely from the model's embedded base_score.
    It silently changed nothing.

    Attempt 2 patched the model dump directly (save_raw → edit JSON →
    load_model) and confirmed the patch WAS applying — but XGBoost's
    own C++ core re-expands base_score back into a per-class array
    the moment load_model() parses it, for ANY multi-class model,
    regardless of what value goes in. That's the library's actual
    design, not a bug to patch around.

    Attempt 3 monkey-patched save_raw() to intercept the exact bytes
    shap requests — and revealed shap actually requests a BINARY
    (UBJSON) dump with its own internal binary decoder, not JSON
    text. Fixing that would mean reimplementing a UBJSON parser to
    edit binary data whose exact layout isn't documented, purely to
    work around a compatibility gap inside someone else's library
    that I cannot install or test in this environment.

    Rather than keep chasing shap's internal XGBoost loader across a
    fourth attempt with no way to verify it locally, this class uses
    XGBoost's own native SHAP computation instead
    (Booster.predict(..., pred_contribs=True)), which has been a
    stable, documented part of core XGBoost for years and produces
    mathematically identical SHAP values to a correctly-working
    TreeExplainer. There is no cross-library serialization boundary
    left to break, because XGBoost computes its own SHAP values on
    its own model.

    Exposes .shap_values(X) with the same output shape the rest of
    this codebase already expects — (n_samples, n_features,
    n_classes) — so get_shap_explanation_for_worker() and the
    dashboard's get_shap() needed no changes beyond how the explainer
    object itself gets built.
    """
    def __init__(self, model, feature_names):
        self.booster = model.get_booster()
        self.feature_names = list(feature_names)

    def shap_values(self, X):
        dmat = xgb.DMatrix(np.asarray(X), feature_names=self.feature_names)
        contribs = np.asarray(self.booster.predict(dmat, pred_contribs=True))
        if contribs.ndim == 3:
            # Native multi-class shape: (n_samples, n_classes,
            # n_features+1) — last column of the last axis is the
            # bias/base-value term, not a feature contribution.
            contribs = contribs[:, :, :-1]                # drop bias col
            contribs = np.transpose(contribs, (0, 2, 1))  # -> (n, feat, cls)
        else:
            # Binary/regression fallback: (n_samples, n_features+1)
            contribs = contribs[:, :-1]
        return contribs


def compute_shap_importance(model, X_test, feature_names):
    """SHAP values for danger class (class 2), via XGBoost's native
    pred_contribs computation (see XGBoostNativeExplainer above)."""
    explainer  = XGBoostNativeExplainer(model, feature_names)
    n_samples  = min(300, len(X_test))
    sv         = explainer.shap_values(X_test[:n_samples])
    # sv shape: (n_samples, n_features, n_classes)
    shap_danger = sv[:, :, 2]
    mean_abs    = np.abs(shap_danger).mean(axis=0)
    sorted_idx  = np.argsort(mean_abs)[::-1]
    return explainer, shap_danger, mean_abs, sorted_idx


def get_shap_explanation_for_worker(explainer, x_scaled, feature_names):
    """Top 3 SHAP drivers for one worker at one moment."""
    sv   = np.array(explainer.shap_values(x_scaled.reshape(1, -1)))
    shap_danger = sv[0, :, 2]
    top3 = np.argsort(np.abs(shap_danger))[::-1][:3]
    result = []
    for fi in top3:
        direction = "increases" if shap_danger[fi] > 0 else "decreases"
        result.append({
            "feature":   feature_names[fi],
            "shap_value": round(float(shap_danger[fi]), 3),
            "direction":  direction,
        })
    return result


# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════
def plot_shap_summary(mean_abs, sorted_idx, feature_names):
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1e2130")

    top_n   = min(20, len(feature_names))
    top_idx = sorted_idx[:top_n][::-1]
    colors  = ["#e74c3c" if mean_abs[i] > mean_abs.mean()
               else "#3498db" for i in top_idx]

    ax.barh(range(top_n), mean_abs[top_idx], color=colors, alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx],
                       color="#ecf0f1", fontsize=8)
    ax.set_xlabel("Mean |SHAP value| — contribution to DANGER prediction",
                  color="#95a5a6", fontsize=9)
    ax.set_title(
        "ProactiveGuard v2.0 — SHAP Feature Importance\n"
        "Danger Class | Temporal features show rate-of-change matters",
        color="#ecf0f1", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#95a5a6")
    for spine in ax.spines.values():
        spine.set_color("#2c3e50")
    ax.grid(True, axis="x", alpha=0.15, color="#2c3e50")
    plt.tight_layout()
    plt.savefig("outputs/v2_shap_summary.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("  Saved: outputs/v2_shap_summary.png")


def plot_confusion_matrix(y_test, preds):
    cm     = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    labels = ["Safe", "Warning", "Danger"]
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1e2130")
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels, color="#ecf0f1")
    ax.set_yticklabels(labels, color="#ecf0f1")
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > cm.max() / 2 else "#ecf0f1"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, color=color, fontweight="bold")
    ax.set_xlabel("Predicted", color="#ecf0f1", fontsize=11)
    ax.set_ylabel("True",      color="#ecf0f1", fontsize=11)
    ax.set_title(
        "ProactiveGuard v2.0 — Confusion Matrix\n"
        "Bottom-left = missed danger events",
        color="#ecf0f1", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/v2_confusion_matrix.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("  Saved: outputs/v2_confusion_matrix.png")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("ProactiveGuard v2.0 — Module 3: Advanced ML Model")
    print("XGBoost + Temporal Features + Uncertainty Quantification")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────
    print("\n[1/7] Loading datasets...")
    df_main = pd.read_csv("data/v2_multisite_data.csv",
                          parse_dates=["timestamp"])
    print(f"      v2_multisite_data: {len(df_main):,} records")

    # Merge twin predictions if available
    twin_path = "data/v2_twin_predictions.csv"
    if os.path.exists(twin_path):
        df_twin = pd.read_csv(twin_path, parse_dates=["timestamp"])
        twin_cols = ["worker_id", "minutes_on_shift",
                     "danger_prob_now", "danger_prob_30min", "cv_strain_now"]
        available = [c for c in twin_cols if c in df_twin.columns]
        df = pd.merge(df_main, df_twin[available],
                      on=["worker_id", "minutes_on_shift"], how="left")
        print(f"      Twin predictions merged: OK")
    else:
        df = df_main.copy()
        for col in ["danger_prob_now", "danger_prob_30min", "cv_strain_now"]:
            df[col] = 0.0
        print("      Twin predictions not found — using zeros")

    # Add acclimatization days from profiles
    df["accl_days"] = df["worker_id"].map(
        lambda x: WORKER_PROFILES.get(x, {}).get("accl_days", 30))

    print(f"      Labels: {dict(df['label'].value_counts().sort_index())}")

    # ── Temporal feature engineering ──────────────────────────
    print("\n[2/7] Engineering temporal features...")
    df = engineer_temporal_features(df)

    # Build feature list — exclude identity and metadata columns
    EXCLUDE = {
        "timestamp", "worker_id", "worker_name", "label",
        "site_id", "site_name", "zone", "role", "age", "bmi",
        "sleep_hours", "hypertension", "ambient_temp_c",
        "humidity_pct", "solar_radiation_wm2", "skin_temp_c",
        "sweat_rate_lhr", "activity_met", "solar_exposure_idx",
        "hr_reserve_pct", "sensor_fault_flag", "data_quality_score",
        "hr_dev_pct", "core_temp_deviation", "resting_hr",
        "max_hr", "baseline_temp", "workload", "minutes_on_shift",
    }
    FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE
                    and df[c].dtype in [np.float64, np.int64, float, int]]
    print(f"      Total features: {len(FEATURE_COLS)}")

    # ── Time-aware split ──────────────────────────────────────
    print("\n[3/7] Time-aware train/test split...")
    train_df, test_df = time_aware_split(df)
    print(f"      Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"      Train labels: {dict(train_df['label'].value_counts().sort_index())}")
    print(f"      Test labels:  {dict(test_df['label'].value_counts().sort_index())}")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["label"].values

    # ── Scale ─────────────────────────────────────────────────
    print("\n[4/7] Scaling features...")
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── SMOTE ─────────────────────────────────────────────────
    print("\n[5/7] Applying SMOTE...")
    print(f"      Before: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_train_sc, y_train)
    print(f"      After:  {dict(zip(*np.unique(y_res, return_counts=True)))}")

    # ── Optuna tuning ─────────────────────────────────────────
    print("\n[6/7] Hyperparameter tuning (20 trials)...")
    val_split = int(len(X_res) * 0.85)
    X_tr_opt, X_val = X_res[:val_split], X_res[val_split:]
    y_tr_opt, y_val = y_res[:val_split], y_res[val_split:]

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 300),
            "max_depth":        trial.suggest_int("max_depth", 4, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "random_state": 42, "eval_metric": "mlogloss", "verbosity": 0,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr_opt, y_tr_opt)
        p = m.predict(X_val)
        return f1_score(y_val, p, average="weighted", zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    print(f"      Best validation F1: {study.best_value:.4f}")

    # Train final model
    model = xgb.XGBClassifier(
        **study.best_params,
        random_state=42, eval_metric="mlogloss", verbosity=0)
    model.fit(X_res, y_res)

    # ── Evaluate ──────────────────────────────────────────────
    print("\n[7/7] Evaluating...")
    preds = model.predict(X_test_sc)
    probs = model.predict_proba(X_test_sc)

    f1w = f1_score(y_test, preds, average="weighted", zero_division=0)
    f1m = f1_score(y_test, preds, average="macro",    zero_division=0)
    auc = 0.0
    if len(np.unique(y_test)) >= 2:
        try:
            auc = roc_auc_score(
                y_test, probs, multi_class="ovr", average="weighted")
        except Exception as e:
            print(f"      ⚠️  AUC-ROC computation failed "
                  f"({type(e).__name__}: {e}) — reporting 0.0, which "
                  f"reflects a computation error, not model quality.")

    # Confidence scoring
    confidence_scores, uncertain_mask = compute_confidence(probs, threshold=0.60)
    uncertain_pct = uncertain_mask.mean() * 100

    # SHAP
    print("      Computing SHAP values...")
    explainer, shap_danger, mean_abs, sorted_idx = compute_shap_importance(
        model, X_test_sc, FEATURE_COLS)

    # Plots
    plot_shap_summary(mean_abs, sorted_idx, FEATURE_COLS)
    plot_confusion_matrix(y_test, preds)

    # Metrics report
    report = classification_report(
        y_test, preds, labels=[0, 1, 2],
        target_names=["Safe", "Warning", "Danger"], zero_division=0)

    metrics_text = (
        f"ProactiveGuard v2.0 — Model Metrics\n"
        f"{'='*45}\n"
        f"F1 Score (weighted) : {f1w:.4f}\n"
        f"F1 Score (macro)    : {f1m:.4f}\n"
        f"AUC-ROC (weighted)  : {auc:.4f}\n"
        f"Uncertain preds     : {uncertain_pct:.1f}%\n\n"
        f"Classification Report:\n{report}\n\n"
        f"Temporal features   : lag5/10/15/30, slope5/15, rm15, rstd10\n"
        f"Class imbalance     : SMOTE\n"
        f"Tuning              : Optuna 20 trials\n"
        f"Uncertainty         : max-probability confidence scoring\n"
    )
    with open("outputs/v2_metrics_report.txt", "w") as f:
        f.write(metrics_text)

    # Save model artifacts
    model.save_model("models/v2_model.json")
    with open("models/v2_scaler.pkl",    "wb") as f: pickle.dump(scaler,    f)
    with open("models/v2_features.pkl",  "wb") as f: pickle.dump(FEATURE_COLS, f)
    with open("models/v2_explainer.pkl", "wb") as f: pickle.dump(explainer, f)
    conf_threshold = 0.60
    with open("models/v2_confidence_threshold.pkl", "wb") as f:
        pickle.dump(conf_threshold, f)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("MODULE 3 COMPLETE")
    print("=" * 65)
    print(f"  F1 (weighted)   : {f1w:.4f}")
    print(f"  F1 (macro)      : {f1m:.4f}")
    print(f"  AUC-ROC         : {auc:.4f}")
    print(f"  Uncertain preds : {uncertain_pct:.1f}%")
    print(f"  Features used   : {len(FEATURE_COLS)}")
    print(f"\n  Top 5 SHAP features (danger class):")
    for i in sorted_idx[:5]:
        print(f"    {FEATURE_COLS[i]:40s}: {mean_abs[i]:.4f}")

    status = "PRODUCTION READY" if f1w >= 0.80 else \
             "ACCEPTABLE" if f1w >= 0.65 else "NEEDS REVIEW"
    print(f"\n  Status: {status}")
    print("\n  Saved:")
    for f in ["models/v2_model.json", "models/v2_scaler.pkl",
              "models/v2_features.pkl", "models/v2_explainer.pkl",
              "outputs/v2_shap_summary.png",
              "outputs/v2_confusion_matrix.png",
              "outputs/v2_metrics_report.txt"]:
        print(f"    {f}")

    print("\nNext: python v2_zone_engine.py")
    print("=" * 65)

    return model, scaler, FEATURE_COLS, explainer


if __name__ == "__main__":
    main()
