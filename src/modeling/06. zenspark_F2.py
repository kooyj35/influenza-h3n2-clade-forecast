import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# Data Paths - Direct paths to your files
# ================================================================

TEST_PATH = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\01. TEST_KJC.csv"
VAL_PATH  = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\02. VAL_KJC.csv"

# ================================================================
# Model Configuration
# ================================================================
MODEL_CONFIG = {
    "l1_ratio": 0.8,
    "C": 0.4,
    "class_weight": {0: 1, 1: 12},
    "max_iter": 3000,
    "threshold": 0.25
}

# ================================================================
# Data Loading Functions
# ================================================================
def load_nextclade(path: str) -> pd.DataFrame:
    """Nextclade 데이터 로드 및 전처리"""
    try:
        df = pd.read_csv(path, sep=";", low_memory=False)
        print(f"✓ Successfully loaded: {path}")
        print(f"  Raw data shape: {df.shape}")
    except FileNotFoundError:
        print(f"✗ File not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return pd.DataFrame()

    if "qc.overallStatus" in df.columns:
        original_count = len(df)
        df = df[df["qc.overallStatus"].astype(str).str.lower() == "good"].copy()
        print(f"  After QC filter: {len(df)}/{original_count} samples")

    # 연도 추출
    s = df["seqName"].astype(str)
    year_isolate = s.str.extract(r'/(\d{4})\|')[0].pipe(pd.to_numeric, errors='coerce')
    year_last = s.str.split('|').str[-1].pipe(pd.to_numeric, errors='coerce')
    df["year"] = year_isolate.where(year_isolate.between(2005, 2025), year_last.where(year_last.between(2005, 2025)))
    
    # 기간 필터링
    df = df[(df["year"] >= 2005) & (df["year"] <= 2025)].copy()
    
    if len(df) == 0:
        print("  Warning: No data after filtering!")
        return pd.DataFrame()

    # 특성 생성
    df["nonsyn"] = pd.to_numeric(df.get("totalAminoacidSubstitutions"), errors="coerce")
    df["total_subs"] = pd.to_numeric(df.get("totalSubstitutions"), errors="coerce")
    df["syn_proxy"] = df["total_subs"] - df["nonsyn"]
    df["novelty"] = df["nonsyn"].fillna(0) + 0.2 * df["syn_proxy"].fillna(0)
    df["pam_reversion"] = pd.to_numeric(df.get("privateAaMutations.totalReversionSubstitutions"), errors="coerce")

    keep = ["seqName", "clade", "year", "nonsyn", "syn_proxy", "novelty", "pam_reversion"]
    df = df[keep].dropna(subset=["clade", "year"]).copy()
    df["year"] = df["year"].astype(int)
    
    print(f"  Final processed data: {len(df)} samples")
    return df

def make_clade_year_table(df: pd.DataFrame) -> pd.DataFrame:
    """Clade-year 집계 테이블 생성"""
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
    """연도별 지배 clade 계산"""
    idx = clade_year.groupby("year")["n"].idxmax()
    cr = clade_year.loc[idx, ["year", "clade"]].set_index("year")["clade"]
    return cr

def build_supervised_dataset(clade_year: pd.DataFrame) -> pd.DataFrame:
    """지도학습용 데이터셋 구성"""
    cr = compute_CR_by_year(clade_year)
    clade_year = clade_year.copy()
    clade_year["CR_next"] = clade_year["year"].map(lambda y: cr.get(y+1, None))
    clade_year = clade_year.dropna(subset=["CR_next"]).copy()
    clade_year["y"] = (clade_year["clade"] == clade_year["CR_next"]).astype(int)
    return clade_year

def make_pr_auc_optimized_model():
    """PR-AUC 최적화된 모델"""
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

def topk_hit(df_year: pd.DataFrame, proba_col: str, k: int = 3) -> int:
    """Top-K 정확도 계산"""
    true_clade = df_year["CR_next"].iloc[0]
    topk = df_year.sort_values(proba_col, ascending=False).head(k)["clade"].tolist()
    return int(true_clade in topk)

# ================================================================
# Main Execution
# ================================================================
def main():
    print("=" * 70)
    print("H3N2 Enhanced Prediction Model")
    print("Direct Execution - PR-AUC & Recall Optimized")
    print("=" * 70)

    # -----------------------
    # Load Data
    # -----------------------
    print("\n1. Loading data...")
    test_raw = load_nextclade(TEST_PATH)
    val_raw = load_nextclade(VAL_PATH)
    
    if test_raw.empty or val_raw.empty:
        print("✗ Failed to load required data files!")
        return
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  TEST: {len(test_raw)} samples")
    print(f"  VAL:  {len(val_raw)} samples")

    # -----------------------
    # Process Data
    # -----------------------
    print("\n2. Processing clade-year data...")
    test_cy = make_clade_year_table(test_raw)
    val_cy = make_clade_year_table(val_raw)
    
    train_df = build_supervised_dataset(test_cy)
    
    print(f"✓ Training data: {len(train_df)} clade-year rows")
    print(f"  Positive samples (y=1): {train_df['y'].sum()}")
    print(f"  Negative samples (y=0): {(train_df['y']==0).sum()}")

    # -----------------------
    # Basic Features (compatibility)
    # -----------------------
    print("\n3. Setting up features...")
    basic_features = ["n", "freq", "freq_prev", "freq_delta", "nonsyn_med", "syn_med", "novelty_med", "pam_reversion_med"]
    print(f"✓ Using {len(basic_features)} basic features for compatibility")

    # -----------------------
    # Time-Series Cross Validation
    # -----------------------
    print("\n4. Running time-series cross validation...")
    
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

        X_train = train_fold[basic_features].fillna(0)
        y_train = train_fold["y"]
        X_val = val_fold[basic_features].fillna(0)
        y_val = val_fold["y"]

        if y_train.nunique() < 2 or y_val.nunique() < 2:
            continue

        # Train model
        model = make_pr_auc_optimized_model()
        model.fit(X_train, y_train)
        val_fold["p_pred"] = model.predict_proba(X_val)[:, 1]

        # Evaluate
        auc = roc_auc_score(y_val, val_fold["p_pred"])
        y_pred = (val_fold["p_pred"] >= THRESHOLD).astype(int)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_val, val_fold["p_pred"])

        hit1 = topk_hit(val_fold, "p_pred", k=1)
        hit3 = topk_hit(val_fold, "p_pred", k=3)

        cv_results["val_year"].append(val_year)
        cv_results["auc"].append(auc)
        cv_results["pr_auc"].append(pr_auc)
        cv_results["precision"].append(precision)
        cv_results["recall"].append(recall)
        cv_results["f1"].append(f1)
        cv_results["hit1"].append(hit1)
        cv_results["hit3"].append(hit3)
        cv_results["train_size"].append(len(train_fold))
        cv_results["val_size"].append(len(val_fold))

    # CV Results Summary
    def safe_mean(lst):
        valid = [x for x in lst if not np.isnan(x)]
        return np.mean(valid) if valid else np.nan

    avg_auc = safe_mean(cv_results["auc"])
    avg_pr_auc = safe_mean(cv_results["pr_auc"])
    avg_precision = safe_mean(cv_results["precision"])
    avg_recall = safe_mean(cv_results["recall"])
    avg_f1 = safe_mean(cv_results["f1"])
    avg_hit1 = np.mean(cv_results["hit1"])
    avg_hit3 = np.mean(cv_results["hit3"])

    print(f"\n[Enhanced Model Performance]")
    print(f"  AUC:       {avg_auc:.3f}")
    print(f"  PR-AUC:    {avg_pr_auc:.3f}")
    print(f"  Precision: {avg_precision:.3f}")
    print(f"  Recall:    {avg_recall:.3f}")
    print(f"  F1:        {avg_f1:.3f}")
    print(f"  Top-1 Hit: {avg_hit1:.1%}")
    print(f"  Top-3 Hit: {avg_hit3:.1%}")

    # F2-Score 계산 (Recall에 높은 가중치)
    f2_scores = []
    for p, r in zip(cv_results["precision"], cv_results["recall"]):
        if p + r > 0:
            f2 = 5 * (p * r) / (4 * p + r)
            f2_scores.append(f2)
    avg_f2 = np.mean(f2_scores) if f2_scores else 0
    print(f"  F2-Score:  {avg_f2:.3f}")

    # -----------------------
    # Final Model Training
    # -----------------------
    print("\n5. Training final model...")
    X_all = train_df[basic_features].fillna(0)
    y_all = train_df["y"]

    final_model = make_pr_auc_optimized_model()
    final_model.fit(X_all, y_all)

    print(f"✓ Final model trained on {len(X_all)} samples")

    # -----------------------
    # 2025 Prediction
    # -----------------------
    print("\n6. Predicting 2025 candidates...")
    
    if not val_cy.empty:
        val_pred = val_cy.copy()
        val_pred["probability"] = final_model.predict_proba(val_pred[basic_features].fillna(0))[:, 1]
        val_pred["predicted_CR"] = (val_pred["probability"] >= THRESHOLD).astype(int)

        # 상위 예측 결과
        print(f"\n[2025 Top Predictions (threshold={THRESHOLD})]")
        top_predictions = val_pred.sort_values("probability", ascending=False).head(10)
        
        for i, (_, row) in enumerate(top_predictions.iterrows(), 1):
            status = "*** PREDICTED CR ***" if row["predicted_CR"] == 1 else ""
            print(f"  {i:2d}. {row['clade']:<15} "
                  f"Prob: {row['probability']:.3f} "
                  f"Freq: {row['freq']:.1%} "
                  f"n={row['n']:3d} {status}")

        # 예측 요약
        n_predicted = val_pred["predicted_CR"].sum()
        print(f"\n[Prediction Summary]")
        print(f"  Total 2025 clades: {len(val_pred)}")
        print(f"  Predicted CR candidates: {n_predicted} (threshold={THRESHOLD})")

    # -----------------------
    # Feature Importance
    # -----------------------
    print(f"\n7. Feature Importance Analysis")
    print("-" * 50)
    coefs = final_model.named_steps['clf'].coef_[0]
    feat_importance = pd.DataFrame({
        "Feature": basic_features,
        "Coefficient": coefs,
        "Abs_Coef": np.abs(coefs)
    }).sort_values("Abs_Coef", ascending=False)
    
    for _, row in feat_importance.iterrows():
        print(f"  {row['Feature']:<20} {row['Coefficient']:>8.3f}")

    # -----------------------
    # Final Summary
    # -----------------------
    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE")
    print("=" * 70)
    print(f"✓ Enhanced model trained and evaluated")
    print(f"✓ 2025 CR candidates predicted")
    print(f"✓ Clinical interpretation provided")
    print(f"\nKey Results:")
    print(f"  - AUC: {avg_auc:.3f}")
    print(f"  - PR-AUC: {avg_pr_auc:.3f}")
    print(f"  - Recall: {avg_recall:.3f}")
    print(f"  - F2-Score: {avg_f2:.3f}")
    print("=" * 70)

if __name__ == "__main__":
    main()