import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')


# ================================================================
# Data Paths
# ================================================================
TEST_PATH = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\01. TEST_KJC.csv"
VAL_PATH  = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\02. VAL_KJC.csv"


# ================================================================
# Model Configuration
# ================================================================
MODEL_CONFIG = {
    "l1_ratio": 0.9,
    "C": 0.5,
    "class_weight": {0: 1, 1: 15},
    "max_iter": 2000,
    "threshold": 0.2  # Recall 최적화를 위해 0.5 -> 0.2로 낮춤
}


# ================================================================
# Helper Functions
# ================================================================
def load_nextclade(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", low_memory=False)

    if "qc.overallStatus" in df.columns:
        df = df[df["qc.overallStatus"].astype(str).str.lower() == "good"].copy()

    s = df["seqName"].astype(str)

    # 1차: isolate name(A/.../YYYY) 에서 /YYYY| 패턴으로 연도 추출 (가장 신뢰도 높음)
    year_isolate = s.str.extract(r'/(\d{4})\|')[0].pipe(pd.to_numeric, errors='coerce')

    # 2차: 마지막 | 필드에서 연도 추출 (fallback)
    year_last = s.str.split('|').str[-1].pipe(pd.to_numeric, errors='coerce')

    # 1차가 유효하면(2005~2025) 사용, 아니면 2차 사용
    df["year"] = year_isolate.where(year_isolate.between(2005, 2025),
                                     year_last.where(year_last.between(2005, 2025)))

    df = df[(df["year"] >= 2005) & (df["year"] <= 2025)].copy()

    df["nonsyn"] = pd.to_numeric(df.get("totalAminoacidSubstitutions"), errors="coerce")
    df["total_subs"] = pd.to_numeric(df.get("totalSubstitutions"), errors="coerce")
    df["syn_proxy"] = df["total_subs"] - df["nonsyn"]
    df["novelty"] = df["nonsyn"].fillna(0) + 0.2 * df["syn_proxy"].fillna(0)

    # 아미노산 역돌연변이
    df["pam_reversion"] = pd.to_numeric(
        df.get("privateAaMutations.totalReversionSubstitutions"), errors="coerce"
    )

    keep = ["seqName", "clade", "year", "nonsyn", "syn_proxy", "novelty", "pam_reversion"]
    df = df[keep].dropna(subset=["clade", "year"]).copy()
    df["year"] = df["year"].astype(int)
    return df


def make_clade_year_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["year", "clade", "n", "nonsyn_med", "syn_med",
                                      "novelty_med", "pam_reversion_med",
                                      "year_total", "freq", "freq_prev", "freq_delta"])

    g = (df.groupby(["year", "clade"])
           .agg(n=("seqName", "count"),
                nonsyn_med=("nonsyn", "median"),
                syn_med=("syn_proxy", "median"),
                novelty_med=("novelty", "median"),
                pam_reversion_med=("pam_reversion", "median"))
           .reset_index())

    totals = g.groupby("year")["n"].sum().reset_index(name="year_total")
    g = g.merge(totals, on="year", how="left")
    g["freq"] = g["n"] / g["year_total"]

    g = g.sort_values(["clade", "year"])
    g["freq_prev"] = g.groupby("clade")["freq"].shift(1).fillna(0)
    g["freq_delta"] = (g["freq"] - g["freq_prev"]).fillna(0)

    return g


def compute_CR_by_year(clade_year: pd.DataFrame) -> pd.Series:
    idx = clade_year.groupby("year")["n"].idxmax()
    cr = clade_year.loc[idx, ["year", "clade"]].set_index("year")["clade"]
    return cr


def build_supervised_dataset(clade_year: pd.DataFrame) -> pd.DataFrame:
    cr = compute_CR_by_year(clade_year)
    clade_year = clade_year.copy()
    clade_year["CR_next"] = clade_year["year"].map(lambda y: cr.get(y+1, None))
    clade_year = clade_year.dropna(subset=["CR_next"]).copy()
    clade_year["y"] = (clade_year["clade"] == clade_year["CR_next"]).astype(int)
    return clade_year


def topk_hit(df_year: pd.DataFrame, proba_col: str, k: int = 3) -> int:
    true_clade = df_year["CR_next"].iloc[0]
    topk = df_year.sort_values(proba_col, ascending=False).head(k)["clade"].tolist()
    return int(true_clade in topk)


def make_optimized_model():
    """최적화된 ElasticNet 모델 생성"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=MODEL_CONFIG["l1_ratio"],
            C=MODEL_CONFIG["C"],
            class_weight=MODEL_CONFIG["class_weight"],
            max_iter=MODEL_CONFIG["max_iter"]
        ))
    ])


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Influenza A H3N2 Clade Prediction")
    print("Recall Optimized Model (threshold=0.2)")
    print("=" * 60)

    # -----------------------
    # Load Data
    # -----------------------
    test_raw = load_nextclade(TEST_PATH)
    val_raw  = load_nextclade(VAL_PATH)

    print(f"\nTEST data: {len(test_raw)} samples")
    print(f"VAL data: {len(val_raw)} samples")

    test_cy = make_clade_year_table(test_raw)
    val_cy  = make_clade_year_table(val_raw)

    train_df = build_supervised_dataset(test_cy)

    # 8개 Features
    features = ["n", "freq", "freq_prev", "freq_delta", "nonsyn_med", "syn_med", "novelty_med", "pam_reversion_med"]
    print(f"\nTraining data: {len(train_df)} clade-year rows")
    print(f"y=1 (minority): {train_df['y'].sum()}, y=0 (majority): {(train_df['y']==0).sum()}")

    # -----------------------
    # Model Configuration
    # -----------------------
    print("\n" + "=" * 60)
    print("Model Configuration")
    print("=" * 60)
    print(f"  Algorithm: ElasticNet Logistic Regression")
    print(f"  l1_ratio: {MODEL_CONFIG['l1_ratio']}")
    print(f"  C: {MODEL_CONFIG['C']}")
    print(f"  class_weight: {MODEL_CONFIG['class_weight']}")
    print(f"  threshold: {MODEL_CONFIG['threshold']} (Recall 최적화)")
    print(f"  Features ({len(features)}): {features}")

    # ================================================================
    # Time-Series Cross Validation
    # ================================================================
    print("\n" + "=" * 60)
    print("Time-Series Cross Validation (Expanding Window)")
    print("=" * 60)

    THRESHOLD = MODEL_CONFIG["threshold"]

    years = sorted(train_df["year"].unique())
    MIN_TRAIN_YEARS = 3
    cv_results = defaultdict(list)

    for i in range(MIN_TRAIN_YEARS, len(years)):
        val_year = years[i]
        train_years = years[:i]

        train_fold = train_df[train_df["year"].isin(train_years)].copy()
        val_fold = train_df[train_df["year"] == val_year].copy()

        if len(val_fold) == 0:
            continue

        X_train = train_fold[features].fillna(0)
        y_train = train_fold["y"]

        if y_train.nunique() < 2:
            continue
        X_val = val_fold[features].fillna(0)
        y_val = val_fold["y"]

        n_classes = y_val.nunique()

        # Train model
        model = make_optimized_model()
        model.fit(X_train, y_train)
        val_fold["p_pred"] = model.predict_proba(X_val)[:, 1]

        # Evaluate with custom threshold
        if n_classes > 1:
            auc = roc_auc_score(y_val, val_fold["p_pred"])
            y_pred = (val_fold["p_pred"] >= THRESHOLD).astype(int)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
        else:
            auc = precision = recall = f1 = np.nan

        hit1 = topk_hit(val_fold, "p_pred", k=1)
        hit3 = topk_hit(val_fold, "p_pred", k=3)

        cv_results["val_year"].append(val_year)
        cv_results["auc"].append(auc)
        cv_results["precision"].append(precision)
        cv_results["recall"].append(recall)
        cv_results["f1"].append(f1)
        cv_results["hit1"].append(hit1)
        cv_results["hit3"].append(hit3)
        cv_results["train_size"].append(len(train_fold))
        cv_results["val_size"].append(len(val_fold))

    # CV Results Table
    cv_df = pd.DataFrame(cv_results)
    print("\n[CV Results by Year]")
    print(cv_df.to_string(index=False))

    # ================================================================
    # Performance Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)

    def safe_mean(lst):
        valid = [x for x in lst if not np.isnan(x)]
        return np.mean(valid) if valid else np.nan

    avg_auc = safe_mean(cv_results["auc"])
    avg_precision = safe_mean(cv_results["precision"])
    avg_recall = safe_mean(cv_results["recall"])
    avg_f1 = safe_mean(cv_results["f1"])
    avg_hit1 = np.mean(cv_results["hit1"])
    avg_hit3 = np.mean(cv_results["hit3"])

    print(f"\n[Current Model - Recall Optimized (th={THRESHOLD})]")
    print(f"  AUC:       {avg_auc:.3f}")
    print(f"  Precision: {avg_precision:.3f}")
    print(f"  Recall:    {avg_recall:.3f}")
    print(f"  F1:        {avg_f1:.3f}")
    print(f"  Top-1 Hit: {avg_hit1:.2f}")
    print(f"  Top-3 Hit: {avg_hit3:.2f}")

    print(f"\n[Comparison: Baseline (th=0.5) vs Recall Optimized (th={THRESHOLD})]")
    print(f"  {'Metric':<12} {'Baseline':>12} {'Optimized':>12} {'Change':>12}")
    print(f"  {'-'*48}")
    print(f"  {'AUC':<12} {'0.807':>12} {avg_auc:>12.3f} {avg_auc - 0.807:>+12.3f}")
    print(f"  {'Precision':<12} {'0.517':>12} {avg_precision:>12.3f} {avg_precision - 0.517:>+12.3f}")
    print(f"  {'Recall':<12} {'0.778':>12} {avg_recall:>12.3f} {avg_recall - 0.778:>+12.3f}")
    print(f"  {'F1':<12} {'0.563':>12} {avg_f1:>12.3f} {avg_f1 - 0.563:>+12.3f}")
    print(f"  {'Top-3 Hit':<12} {'0.50':>12} {avg_hit3:>12.2f} {avg_hit3 - 0.50:>+12.2f}")

    # ================================================================
    # Final Model Training
    # ================================================================
    print("\n" + "=" * 60)
    print("Final Model: Train on ALL data")
    print("=" * 60)

    X_all, y_all = train_df[features].fillna(0), train_df["y"]

    final_model = make_optimized_model()
    final_model.fit(X_all, y_all)

    print(f"Model trained on {len(X_all)} samples with {len(features)} features")

    # Feature Coefficients
    print("\n[Feature Coefficients]")
    coefs = final_model.named_steps['clf'].coef_[0]
    feat_coef = pd.DataFrame({
        "Feature": features,
        "Coefficient": coefs,
        "Abs_Coef": np.abs(coefs)
    }).sort_values("Abs_Coef", ascending=False)
    print(feat_coef[["Feature", "Coefficient"]].to_string(index=False))

    # ================================================================
    # 2025 Prediction
    # ================================================================
    print("\n" + "=" * 60)
    print(f"2025 Prediction (2026 CR Candidates) - threshold={THRESHOLD}")
    print("=" * 60)

    if not val_cy.empty:
        val_pred = val_cy.copy()
        val_pred["probability"] = final_model.predict_proba(val_pred[features].fillna(0))[:, 1]
        val_pred["predicted_CR"] = (val_pred["probability"] >= THRESHOLD).astype(int)

        # dN/dS ratio
        val_pred["dN/dS"] = (val_pred["nonsyn_med"] / val_pred["syn_med"]).round(3)

        print("\n[2025 Clade Analysis - Top 10]")
        result = val_pred.sort_values("probability", ascending=False)[
            ["clade", "n", "freq", "nonsyn_med", "syn_med", "dN/dS", "pam_reversion_med", "probability", "predicted_CR"]
        ].head(10)
        print(result.to_string(index=False))

        # Prediction Summary
        n_predicted = val_pred["predicted_CR"].sum()
        print(f"\n[Prediction Summary]")
        print(f"  Total clades in 2025: {len(val_pred)}")
        print(f"  Predicted as 2026 CR: {n_predicted} clades (threshold={THRESHOLD})")

        # Top 3 Candidates
        print("\n[2026 CR Prediction - Top 3 Candidates]")
        print("-" * 50)
        top3 = val_pred.sort_values("probability", ascending=False).head(3)
        for i, (_, row) in enumerate(top3.iterrows(), 1):
            status = "*** PREDICTED CR ***" if row["predicted_CR"] == 1 else ""
            print(f"  #{i}: {row['clade']}")
            print(f"      Probability: {row['probability']:.3f}")
            print(f"      Frequency:   {row['freq']:.1%} ({row['n']} samples)")
            print(f"      dN/dS:       {row['dN/dS']}")
            print(f"      AA Reversion: {row['pam_reversion_med']}")
            print(f"      {status}")
            print()

    else:
        print("\n[WARNING] No validation data available.")

    # ================================================================
    # Historical CR Trend
    # ================================================================
    print("=" * 60)
    print("Historical CR Trend (Recent 5 Years)")
    print("=" * 60)

    cr_history = compute_CR_by_year(test_cy)
    for year in sorted(cr_history.index)[-5:]:
        clade = cr_history[year]
        year_data = test_raw[test_raw["year"] == year]
        total = len(year_data)
        clade_count = len(year_data[year_data["clade"] == clade])
        pct = clade_count / total * 100 if total > 0 else 0
        print(f"  {year}: {clade} ({clade_count}/{total}, {pct:.1f}%)")

    # ================================================================
    # Model Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("MODEL SUMMARY (Recall Optimized)")
    print("=" * 60)
    print(f"""
  Algorithm:     ElasticNet Logistic Regression
  Optimization:  Recall focused (threshold={THRESHOLD})
  Features:      {len(features)} features

  Key Change:
    - Threshold: 0.5 -> {THRESHOLD}
    - 더 낮은 확률도 CR 후보로 포함하여 놓치는 것 방지

  Performance:
    - AUC:       {avg_auc:.3f}
    - Precision: {avg_precision:.3f}
    - Recall:    {avg_recall:.3f} (+{(avg_recall - 0.778)*100:.0f}%p from baseline)
    - F1:        {avg_f1:.3f}
    - Top-3 Hit: {avg_hit3:.0%}

  Trade-off:
    - Recall 향상: 0.778 -> {avg_recall:.3f} (+{(avg_recall - 0.778)*100:.1f}%p)
    - Precision 감소: 0.517 -> {avg_precision:.3f} ({(avg_precision - 0.517)*100:.1f}%p)
""")

    print("=" * 60)
    print("Done!")
    print("=" * 60)
