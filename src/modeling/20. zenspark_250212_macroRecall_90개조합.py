"""
H3N2 Influenza Clade Prediction Model - Macro Recall ~0.9 최적화
================================================================
목적: Macro Recall을 1.0에서 ~0.9로 낮추되, AUC / PR-AUC를 높이는 최적 조합 탐색
방법: C, l1_ratio, class_weight, min_recall 그리드서치
데이터: 한국+중국+일본 Nextclade 데이터 (2005-2025)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# 1. 설정 (Configuration)
# ================================================================


TEST_PATH = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\01. TEST_KJC.csv"
VAL_PATH  = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\02. VAL_KJC.csv"


FEATURES = [
    "n", "freq", "freq_prev", "freq_delta",
    "nonsyn_med", "syn_med", "novelty_med", "pam_reversion_med",
]

# 그리드서치 대상 하이퍼파라미터
GRID = {
    "C":            [0.3, 0.5, 0.8, 1.0, 2.0],
    "l1_ratio":     [0.5, 0.7, 0.9],
    "class_weight": ["balanced", {0:1, 1:5}, {0:1, 1:8}],
    "min_recall":   [0.0, 0.5],   # 0.0 = 제약 없음(순수 F1 최적화)
}


# ================================================================
# 2. 데이터 로딩 (Data Loading)
# ================================================================

def load_nextclade(path: str) -> pd.DataFrame:
    """Nextclade CSV 로드 + QC필터 + 연도추출 + 돌연변이 피처 생성"""
    try:
        df = pd.read_csv(path, sep=";", low_memory=False)
    except Exception as e:
        print(f"  [오류] {e}")
        return pd.DataFrame()

    if "qc.overallStatus" in df.columns:
        df = df[df["qc.overallStatus"].astype(str).str.lower() == "good"].copy()

    s = df["seqName"].astype(str)
    year_from_name = s.str.extract(r'/(\d{4})\|')[0].pipe(pd.to_numeric, errors='coerce')
    year_from_tail = s.str.split('|').str[-1].pipe(pd.to_numeric, errors='coerce')
    df["year"] = year_from_name.where(
        year_from_name.between(2005, 2025),
        year_from_tail.where(year_from_tail.between(2005, 2025))
    )
    df = df[(df["year"] >= 2005) & (df["year"] <= 2025)].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df["nonsyn"]        = pd.to_numeric(df.get("totalAminoacidSubstitutions"), errors="coerce")
    df["total_subs"]    = pd.to_numeric(df.get("totalSubstitutions"), errors="coerce")
    df["syn_proxy"]     = df["total_subs"] - df["nonsyn"]
    df["novelty"]       = df["nonsyn"].fillna(0) + 0.2 * df["syn_proxy"].fillna(0)
    df["pam_reversion"] = pd.to_numeric(
        df.get("privateAaMutations.totalReversionSubstitutions"), errors="coerce"
    )

    keep = ["seqName", "clade", "year", "nonsyn", "syn_proxy", "novelty", "pam_reversion"]
    df = df[keep].dropna(subset=["clade", "year"]).copy()
    df["year"] = df["year"].astype(int)
    return df


# ================================================================
# 3. 피처 엔지니어링 (Feature Engineering)
# ================================================================

def make_clade_year_table(df):
    if df.empty:
        return pd.DataFrame(columns=["year", "clade"] + FEATURES)
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
    g["freq_prev"]  = g.groupby("clade")["freq"].shift(1).fillna(0)
    g["freq_delta"] = (g["freq"] - g["freq_prev"]).fillna(0)
    return g


# ================================================================
# 4. 라벨 생성 (Label Creation)
# ================================================================

def get_dominant_clade_by_year(clade_year):
    idx = clade_year.groupby("year")["n"].idxmax()
    return clade_year.loc[idx, ["year", "clade"]].set_index("year")["clade"]


def build_supervised_dataset(clade_year):
    cr = get_dominant_clade_by_year(clade_year)
    df = clade_year.copy()
    df["CR_next"] = df["year"].map(lambda y: cr.get(y + 1, None))
    df = df.dropna(subset=["CR_next"]).copy()
    df["y"] = (df["clade"] == df["CR_next"]).astype(int)
    return df


# ================================================================
# 5. 모델 & 평가 (Model & Evaluation)
# ================================================================

def find_best_threshold(y_true, y_proba, min_recall=0.0):
    """F1을 최대화하는 threshold 탐색 (min_recall 제약 적용)"""
    best_threshold, best_f1 = 0.5, 0
    best_metrics = {"threshold": 0.5, "precision": 0, "recall": 0, "f1": 0}

    for thr in np.arange(0.10, 0.90, 0.05):
        y_pred = (y_proba >= thr).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        if rec < min_recall:
            continue
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr
            best_metrics = {"threshold": thr, "precision": prec, "recall": rec, "f1": f1}

    return best_threshold, best_metrics


def topk_hit(df_year, proba_col, k=3):
    true_clade = df_year["CR_next"].iloc[0]
    topk = df_year.sort_values(proba_col, ascending=False).head(k)["clade"].tolist()
    return int(true_clade in topk)


def evaluate_config(train_df, C, l1_ratio, class_weight, min_recall):
    """주어진 하이퍼파라미터 조합으로 CV 수행 후 성능 반환"""
    years = sorted(train_df["year"].unique())
    MIN_TRAIN_YEARS = 3

    fold_metrics = defaultdict(list)
    all_y_true, all_y_proba = [], []

    for i in range(MIN_TRAIN_YEARS, len(years)):
        val_year = years[i]
        train_fold = train_df[train_df["year"].isin(years[:i])]
        val_fold   = train_df[train_df["year"] == val_year]
        if len(val_fold) == 0:
            continue

        X_train, y_train = train_fold[FEATURES].fillna(0), train_fold["y"]
        X_val,   y_val   = val_fold[FEATURES].fillna(0),   val_fold["y"]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga",
                l1_ratio=l1_ratio, C=C,
                class_weight=class_weight, max_iter=3000
            ))
        ])
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]

        all_y_true.extend(y_val.tolist())
        all_y_proba.extend(y_proba.tolist())

        opt_thr, m = find_best_threshold(y_val, y_proba, min_recall=min_recall)
        auc    = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)

        vf = val_fold.copy()
        vf["p_pred"] = y_proba
        hit1 = topk_hit(vf, "p_pred", k=1)
        hit3 = topk_hit(vf, "p_pred", k=3)

        fold_metrics["auc"].append(auc)
        fold_metrics["pr_auc"].append(pr_auc)
        fold_metrics["precision"].append(m["precision"])
        fold_metrics["recall"].append(m["recall"])
        fold_metrics["f1"].append(m["f1"])
        fold_metrics["threshold"].append(opt_thr)
        fold_metrics["hit1"].append(hit1)
        fold_metrics["hit3"].append(hit3)

    if not fold_metrics["f1"]:
        return None

    macro = {k: np.mean(v) for k, v in fold_metrics.items()}

    # Micro
    all_y_true  = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)
    micro_thr, _ = find_best_threshold(all_y_true, all_y_proba, min_recall=min_recall)
    all_y_pred = (all_y_proba >= micro_thr).astype(int)
    micro = {
        "precision": precision_score(all_y_true, all_y_pred, zero_division=0),
        "recall":    recall_score(all_y_true, all_y_pred, zero_division=0),
        "f1":        f1_score(all_y_true, all_y_pred, zero_division=0),
        "threshold": micro_thr,
    }

    return {"macro": macro, "micro": micro, "fold_metrics": fold_metrics}


# ================================================================
# 6. 메인 실행 (Main)
# ================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("H3N2 Clade Prediction - Grid Search for Macro Recall ~0.9")
    print("=" * 70)

    # ── 데이터 로드 ──────────────────────────────────────────
    test_raw = load_nextclade(TEST_PATH)
    val_raw  = load_nextclade(VAL_PATH)
    if test_raw.empty or val_raw.empty:
        print("데이터 로드 실패"); exit()

    test_cy  = make_clade_year_table(test_raw)
    val_cy   = make_clade_year_table(val_raw)
    train_df = build_supervised_dataset(test_cy)

    print(f"TEST: {len(test_raw)} samples | VAL: {len(val_raw)} samples")
    print(f"Training: {len(train_df)} rows (y=1: {train_df['y'].sum()}, y=0: {(train_df['y']==0).sum()})")

    # ── 그리드서치 ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Grid Search")
    print("=" * 70)

    total_combos = (len(GRID["C"]) * len(GRID["l1_ratio"])
                    * len(GRID["class_weight"]) * len(GRID["min_recall"]))
    print(f"총 {total_combos}개 조합 탐색...\n")

    results = []

    for C in GRID["C"]:
        for l1 in GRID["l1_ratio"]:
            for cw in GRID["class_weight"]:
                for mr in GRID["min_recall"]:
                    res = evaluate_config(train_df, C, l1, cw, mr)
                    if res is None:
                        continue

                    macro = res["macro"]
                    micro = res["micro"]

                    # class_weight를 보기 좋게 변환
                    cw_str = "balanced" if cw == "balanced" else f"1:{cw[1]}"

                    results.append({
                        "C": C, "l1_ratio": l1, "class_weight": cw_str,
                        "min_recall": mr,
                        "macro_auc": macro["auc"],
                        "macro_pr_auc": macro["pr_auc"],
                        "macro_prec": macro["precision"],
                        "macro_recall": macro["recall"],
                        "macro_f1": macro["f1"],
                        "macro_thr": macro["threshold"],
                        "macro_hit1": macro["hit1"],
                        "macro_hit3": macro["hit3"],
                        "micro_prec": micro["precision"],
                        "micro_recall": micro["recall"],
                        "micro_f1": micro["f1"],
                    })

    results_df = pd.DataFrame(results)

    # ── 전체 결과 출력 ───────────────────────────────────────
    print("\n[All Results - sorted by macro_f1]")
    show_cols = ["C", "l1_ratio", "class_weight", "min_recall",
                 "macro_auc", "macro_pr_auc", "macro_prec", "macro_recall",
                 "macro_f1", "macro_hit1", "macro_hit3",
                 "micro_prec", "micro_recall"]
    print(results_df.sort_values("macro_f1", ascending=False)[show_cols]
          .to_string(index=False))

    # ── Recall 0.85~0.95 필터 ────────────────────────────────
    print("\n" + "=" * 70)
    print("Filtered: Macro Recall 0.85 ~ 0.95 (목표 범위)")
    print("=" * 70)

    filtered = results_df[
        (results_df["macro_recall"] >= 0.85) &
        (results_df["macro_recall"] <= 0.95)
    ].copy()

    if len(filtered) == 0:
        # 범위를 넓혀서 재시도
        print("  0.85~0.95 범위 결과 없음 -> 0.80~1.00 범위로 확장")
        filtered = results_df[
            (results_df["macro_recall"] >= 0.80) &
            (results_df["macro_recall"] <= 1.00)
        ].copy()

    if len(filtered) == 0:
        print("  적합한 결과 없음. 전체 결과에서 최적 선택.")
        filtered = results_df.copy()

    # AUC + PR-AUC 합산 점수로 정렬
    filtered["score"] = filtered["macro_auc"] + filtered["macro_pr_auc"]
    filtered = filtered.sort_values("score", ascending=False)

    print(f"\n  {len(filtered)}개 조합 발견")
    print(filtered[show_cols].head(10).to_string(index=False))

    # ── 최적 조합 선택 ──────────────────────────────────────
    best = filtered.iloc[0]

    print("\n" + "=" * 70)
    print("BEST Configuration")
    print("=" * 70)
    print(f"  C:            {best['C']}")
    print(f"  l1_ratio:     {best['l1_ratio']}")
    print(f"  class_weight: {best['class_weight']}")
    print(f"  min_recall:   {best['min_recall']}")
    print(f"  threshold:    {best['macro_thr']:.3f}")

    print(f"\n[Performance - Macro Average]")
    print(f"  AUC:          {best['macro_auc']:.3f}")
    print(f"  PR-AUC:       {best['macro_pr_auc']:.3f}")
    print(f"  Precision:    {best['macro_prec']:.3f}")
    print(f"  Recall:       {best['macro_recall']:.3f}")
    print(f"  F1:           {best['macro_f1']:.3f}")
    print(f"  Top-1 Hit:    {best['macro_hit1']:.2f}")
    print(f"  Top-3 Hit:    {best['macro_hit3']:.2f}")

    print(f"\n[Performance - Micro Average]")
    print(f"  Precision:    {best['micro_prec']:.3f}")
    print(f"  Recall:       {best['micro_recall']:.3f}")
    print(f"  F1:           {best['micro_f1']:.3f}")

    # ── 최적 조합으로 최종 모델 학습 ─────────────────────────
    print("\n" + "=" * 70)
    print("Final Model: Train on ALL data with BEST config")
    print("=" * 70)

    best_cw = "balanced" if best["class_weight"] == "balanced" else {0: 1, 1: int(best["class_weight"].split(":")[1])}

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=best["l1_ratio"], C=best["C"],
            class_weight=best_cw, max_iter=3000
        ))
    ])

    X_all = train_df[FEATURES].fillna(0)
    y_all = train_df["y"]
    final_model.fit(X_all, y_all)

    y_proba_all = final_model.predict_proba(X_all)[:, 1]
    opt_threshold, thr_metrics = find_best_threshold(
        y_all, y_proba_all, min_recall=best["min_recall"]
    )

    print(f"Model trained on {len(X_all)} samples")
    print(f"Optimized threshold: {opt_threshold:.3f}")

    # Feature Coefficients
    print("\n[Feature Coefficients]")
    coefs = final_model.named_steps["clf"].coef_[0]
    feat_coef = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": coefs,
        "Abs_Coef": np.abs(coefs),
    }).sort_values("Abs_Coef", ascending=False)
    print(feat_coef[["Feature", "Coefficient"]].to_string(index=False))

    # ── 2025 Prediction ──────────────────────────────────────
    THRESHOLD = opt_threshold

    print("\n" + "=" * 70)
    print(f"2025 Prediction (2026 CR Candidates) - threshold={THRESHOLD:.3f}")
    print("=" * 70)

    if not val_cy.empty:
        val_pred = val_cy.copy()
        val_pred["probability"]  = final_model.predict_proba(val_pred[FEATURES].fillna(0))[:, 1]
        val_pred["predicted_CR"] = (val_pred["probability"] >= THRESHOLD).astype(int)
        val_pred["dN/dS"] = (val_pred["nonsyn_med"] / val_pred["syn_med"]).round(3)

        print("\n[2025 Clade Analysis - Top 10]")
        result = val_pred.sort_values("probability", ascending=False)[
            ["clade", "n", "freq", "nonsyn_med", "syn_med", "dN/dS",
             "pam_reversion_med", "probability", "predicted_CR"]
        ].head(10)
        print(result.to_string(index=False))

        n_predicted  = val_pred["predicted_CR"].sum()
        total_clades = len(val_pred)
        print(f"\n[Prediction Summary]")
        print(f"  Total clades in 2025: {total_clades}")
        print(f"  Predicted as 2026 CR: {n_predicted} clades (threshold={THRESHOLD:.3f})")

        print(f"\n[2026 CR Prediction - Top 3 Candidates]")
        print("-" * 50)
        top3 = val_pred.sort_values("probability", ascending=False).head(3)
        for i, (_, row) in enumerate(top3.iterrows(), 1):
            status = "*** PREDICTED CR ***" if row["predicted_CR"] == 1 else ""
            print(f"  #{i}: {row['clade']}")
            print(f"      Probability: {row['probability']:.3f}")
            print(f"      Frequency:   {row['freq']:.1%} ({row['n']:.0f} samples)")
            print(f"      dN/dS:       {row['dN/dS']}")
            print(f"      AA Reversion: {row['pam_reversion_med']}")
            print(f"      {status}")
            print()
    else:
        print("\n[WARNING] No validation data available.")

    # ── Historical CR Trend ──────────────────────────────────
    print("=" * 70)
    print("Historical CR Trend (Recent 5 Years)")
    print("=" * 70)

    cr_history = get_dominant_clade_by_year(test_cy)
    for year in sorted(cr_history.index)[-5:]:
        clade = cr_history[year]
        year_data = test_raw[test_raw["year"] == year]
        total = len(year_data)
        clade_count = len(year_data[year_data["clade"] == clade])
        pct = clade_count / total * 100 if total > 0 else 0
        print(f"  {year}: {clade} ({clade_count}/{total}, {pct:.1f}%)")

    # ── MODEL SUMMARY ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"""
  Algorithm:     ElasticNet Logistic Regression
  C:             {best['C']}
  l1_ratio:      {best['l1_ratio']}
  class_weight:  {best['class_weight']}
  Threshold:     {opt_threshold:.3f} (min_recall={best['min_recall']})
  Features:      {len(FEATURES)} features

  Performance (Macro Average):
    - AUC:          {best['macro_auc']:.3f}
    - PR-AUC:       {best['macro_pr_auc']:.3f}
    - Precision:    {best['macro_prec']:.3f}
    - Recall:       {best['macro_recall']:.3f}
    - F1:           {best['macro_f1']:.3f}
    - Top-1 Hit:    {best['macro_hit1']:.0%}
    - Top-3 Hit:    {best['macro_hit3']:.0%}

  Performance (Micro Average):
    - Precision:    {best['micro_prec']:.3f}
    - Recall:       {best['micro_recall']:.3f}
    - F1:           {best['micro_f1']:.3f}
""")
    print("=" * 70)
    print("Done!")
    print("=" * 70)
