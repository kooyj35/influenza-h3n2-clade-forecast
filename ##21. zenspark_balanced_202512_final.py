"""
H3N2 Influenza Clade Prediction Model (Balanced, l1_ratio=0.8)
================================================================
목적: 다음 해 지배 clade(CR: Candidate Recommendation)를 예측
전략: Balanced 가중치 + Threshold 최적화 + ElasticNet(l1=0.8, C=0.4)
데이터: 한국+중국+일본 Nextclade 데이터 (2005-2025)

재현성: random_state=42 고정 (매 실행 동일 결과 보장)
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

# 데이터 경로
TEST_PATH = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\01. TEST_KJC.csv"
VAL_PATH  = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\02. VAL_KJC.csv"

# 모델 하이퍼파라미터
MODEL_CONFIG = {
    "l1_ratio": 0.8,       # ElasticNet L1 비율 (0=Ridge, 1=Lasso)
    "C": 0.4,              # 정규화 강도 (작을수록 강한 정규화)
    "max_iter": 3000,      # 최적화 최대 반복 수
    "threshold": 0.25,     # 초기 분류 임계값 (학습 중 최적화됨)
}

# 사용할 피처 목록
FEATURES = [
    "n",                   # 해당 clade의 시퀀스 수
    "freq",                # 해당 연도 내 빈도
    "freq_prev",           # 전년도 빈도
    "freq_delta",          # 빈도 변화량 (올해 - 작년)
    "nonsyn_med",          # 비동의 치환 중앙값
    "syn_med",             # 동의 치환 중앙값
    "novelty_med",         # 새로움 점수 중앙값
    "pam_reversion_med",   # 복귀 돌연변이 중앙값
]


# ================================================================
# 2. 데이터 로딩 (Data Loading)
# ================================================================

def load_nextclade(path: str) -> pd.DataFrame:
    """Nextclade CSV 파일을 로드하고 전처리하여 반환한다.

    처리 과정:
      1) QC 필터링 (good만 유지)
      2) seqName에서 연도 추출
      3) 돌연변이 관련 피처 생성
    """
    # --- 파일 읽기 ---
    try:
        df = pd.read_csv(path, sep=";", low_memory=False)
        print(f"  [로드 완료] {path}")
        print(f"    원본 데이터: {df.shape[0]}개 시퀀스")
    except FileNotFoundError:
        print(f"  [오류] 파일 없음: {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  [오류] {e}")
        return pd.DataFrame()

    # --- QC 필터링: 품질 'good'인 시퀀스만 유지 ---
    if "qc.overallStatus" in df.columns:
        before = len(df)
        df = df[df["qc.overallStatus"].astype(str).str.lower() == "good"].copy()
        print(f"    QC 필터 후: {len(df)}/{before}개")

    # --- 연도 추출 (seqName에서 파싱) ---
    s = df["seqName"].astype(str)
    year_from_name = s.str.extract(r'/(\d{4})\|')[0].pipe(pd.to_numeric, errors='coerce')
    year_from_tail = s.str.split('|').str[-1].pipe(pd.to_numeric, errors='coerce')
    df["year"] = year_from_name.where(
        year_from_name.between(2005, 2025),
        year_from_tail.where(year_from_tail.between(2005, 2025))
    )
    df = df[(df["year"] >= 2005) & (df["year"] <= 2025)].copy()

    if len(df) == 0:
        print("    [경고] 필터링 후 데이터 없음!")
        return pd.DataFrame()

    # --- 돌연변이 피처 생성 ---
    df["nonsyn"]    = pd.to_numeric(df.get("totalAminoacidSubstitutions"), errors="coerce")
    df["total_subs"] = pd.to_numeric(df.get("totalSubstitutions"), errors="coerce")
    df["syn_proxy"]  = df["total_subs"] - df["nonsyn"]                         # 동의 치환 근사값
    df["novelty"]    = df["nonsyn"].fillna(0) + 0.2 * df["syn_proxy"].fillna(0)  # 새로움 점수
    df["pam_reversion"] = pd.to_numeric(
        df.get("privateAaMutations.totalReversionSubstitutions"), errors="coerce"
    )

    # --- 필요한 컬럼만 유지 ---
    keep = ["seqName", "clade", "year", "nonsyn", "syn_proxy", "novelty", "pam_reversion"]
    df = df[keep].dropna(subset=["clade", "year"]).copy()
    df["year"] = df["year"].astype(int)

    print(f"    최종 데이터: {len(df)}개 시퀀스")
    return df


# ================================================================
# 3. 피처 엔지니어링 (Feature Engineering)
# ================================================================

def make_clade_year_table(df: pd.DataFrame) -> pd.DataFrame:
    """시퀀스 데이터를 clade-연도 단위로 집계한다.

    생성 피처:
      - n: 시퀀스 수
      - freq: 연도 내 빈도 비율
      - freq_prev: 전년도 빈도
      - freq_delta: 빈도 변화량
      - nonsyn_med, syn_med, novelty_med, pam_reversion_med: 중앙값
    """
    if df.empty:
        return pd.DataFrame(columns=["year", "clade"] + FEATURES)

    # clade-연도별 집계
    g = (df.groupby(["year", "clade"])
           .agg(
               n=("seqName", "count"),
               nonsyn_med=("nonsyn", "median"),
               syn_med=("syn_proxy", "median"),
               novelty_med=("novelty", "median"),
               pam_reversion_med=("pam_reversion", "median"),
           )
           .reset_index())

    # 연도별 전체 시퀀스 수 -> 빈도 계산
    totals = g.groupby("year")["n"].sum().reset_index(name="year_total")
    g = g.merge(totals, on="year", how="left")
    g["freq"] = g["n"] / g["year_total"]

    # 전년도 빈도 및 변화량
    g = g.sort_values(["clade", "year"])
    g["freq_prev"]  = g.groupby("clade")["freq"].shift(1).fillna(0)
    g["freq_delta"] = (g["freq"] - g["freq_prev"]).fillna(0)

    return g


# ================================================================
# 4. 라벨 생성 (Label Creation)
# ================================================================

def get_dominant_clade_by_year(clade_year: pd.DataFrame) -> pd.Series:
    """각 연도에서 가장 많은 시퀀스를 가진 clade(지배 clade)를 반환한다."""
    idx = clade_year.groupby("year")["n"].idxmax()
    return clade_year.loc[idx, ["year", "clade"]].set_index("year")["clade"]


def build_supervised_dataset(clade_year: pd.DataFrame) -> pd.DataFrame:
    """지도학습용 데이터셋을 구성한다.

    라벨 정의:
      y=1: 해당 clade가 다음 해의 지배 clade와 일치
      y=0: 불일치
    """
    cr = get_dominant_clade_by_year(clade_year)

    df = clade_year.copy()
    df["CR_next"] = df["year"].map(lambda y: cr.get(y + 1, None))
    df = df.dropna(subset=["CR_next"]).copy()
    df["y"] = (df["clade"] == df["CR_next"]).astype(int)

    return df


# ================================================================
# 5. 모델 학습 및 평가 (Model Training & Evaluation)
# ================================================================

def build_model():
    """ElasticNet 로지스틱 회귀 파이프라인을 생성한다.
    StandardScaler로 정규화 후, balanced 가중치를 적용한다.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=MODEL_CONFIG["l1_ratio"],
            C=MODEL_CONFIG["C"],
            class_weight="balanced",   # 클래스 불균형 자동 보정
            max_iter=MODEL_CONFIG["max_iter"],
            random_state=42,           # 재현성 보장
        )),
    ])


def find_best_threshold(y_true, y_proba, min_recall=0.7):
    """F1 점수를 최대화하는 분류 임계값을 탐색한다.

    Args:
        y_true: 실제 라벨
        y_proba: 예측 확률
        min_recall: 최소 recall 제약 (이 값 미만인 임계값은 제외)

    Returns:
        (best_threshold, metrics_dict)
    """
    best_threshold, best_f1 = 0.5, 0
    best_metrics = {}

    for threshold in np.arange(0.10, 0.90, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        if y_pred.sum() == 0:
            continue

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)

        if rec < min_recall:
            continue

        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {"threshold": threshold, "precision": prec, "recall": rec, "f1": f1}

    return best_threshold, best_metrics


def topk_hit(df_year: pd.DataFrame, proba_col: str, k: int = 3) -> int:
    """예측 확률 상위 k개 안에 실제 지배 clade가 포함되는지 확인한다."""
    true_clade = df_year["CR_next"].iloc[0]
    topk = df_year.sort_values(proba_col, ascending=False).head(k)["clade"].tolist()
    return int(true_clade in topk)


def cross_validate(train_df: pd.DataFrame):
    """시간순 교차검증(Expanding Window)을 수행한다.

    - 최소 3년 학습 후부터 검증 시작
    - 각 fold에서 threshold 최적화, AUC, PR-AUC, Top-K Hit 등 측정
    - Macro 평균과 Micro 평균을 모두 계산

    Returns:
        (cv_results_dict, macro_dict, micro_dict)
    """
    years = sorted(train_df["year"].unique())
    MIN_TRAIN_YEARS = 3

    # fold별 결과 저장
    cv = defaultdict(list)
    # Micro 평균 계산용: 전체 fold의 예측값을 모음
    all_y_true, all_y_proba = [], []

    for i in range(MIN_TRAIN_YEARS, len(years)):
        val_year = years[i]
        train_years = years[:i]

        train_fold = train_df[train_df["year"].isin(train_years)]
        val_fold   = train_df[train_df["year"] == val_year]

        if len(val_fold) == 0:
            continue

        X_train = train_fold[FEATURES].fillna(0)
        y_train = train_fold["y"]
        X_val   = val_fold[FEATURES].fillna(0)
        y_val   = val_fold["y"]

        # 학습/검증 세트에 양성+음성이 모두 있어야 평가 가능
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        # 모델 학습 및 예측
        model = build_model()
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]

        # Micro 평균용 누적
        all_y_true.extend(y_val.tolist())
        all_y_proba.extend(y_proba.tolist())

        # Threshold 최적화 (이 fold 기준)
        opt_thr, metrics = find_best_threshold(y_val, y_proba, min_recall=0.7)

        # AUC / PR-AUC
        auc    = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)

        # Top-K Hit Rate
        vf = val_fold.copy()
        vf["p_pred"] = y_proba
        hit1 = topk_hit(vf, "p_pred", k=1)
        hit3 = topk_hit(vf, "p_pred", k=3)

        # 결과 기록
        cv["val_year"].append(val_year)
        cv["train_size"].append(len(train_fold))
        cv["val_size"].append(len(val_fold))
        cv["threshold"].append(opt_thr)
        cv["auc"].append(auc)
        cv["pr_auc"].append(pr_auc)
        cv["precision"].append(metrics.get("precision", 0))
        cv["recall"].append(metrics.get("recall", 0))
        cv["f1"].append(metrics.get("f1", 0))
        cv["hit1"].append(hit1)
        cv["hit3"].append(hit3)

    # --- Macro 평균 (fold별 평균) ---
    metric_keys = ["threshold", "auc", "pr_auc", "precision", "recall", "f1", "hit1", "hit3"]
    macro = {k: np.mean(cv[k]) for k in metric_keys}

    # --- Micro 평균 (전체 예측을 한꺼번에 평가) ---
    all_y_true  = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)
    micro_thr, _ = find_best_threshold(all_y_true, all_y_proba, min_recall=0.7)
    all_y_pred = (all_y_proba >= micro_thr).astype(int)

    micro = {
        "precision": precision_score(all_y_true, all_y_pred, zero_division=0),
        "recall":    recall_score(all_y_true, all_y_pred, zero_division=0),
        "f1":        f1_score(all_y_true, all_y_pred, zero_division=0),
        "threshold": micro_thr,
    }

    return cv, macro, micro


# ================================================================
# 6. 메인 실행 (Main)
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("H3N2 Clade Prediction - Balanced + Threshold 최적화")
    print("=" * 60)

    # ================================================================
    # [1] 데이터 로드
    # ================================================================
    print("\n[1] 데이터 로드")
    test_raw = load_nextclade(TEST_PATH)
    val_raw  = load_nextclade(VAL_PATH)

    if test_raw.empty or val_raw.empty:
        print("  데이터 로드 실패. 종료합니다.")
        exit()

    print(f"\nTEST data: {len(test_raw)} samples")
    print(f"VAL data:  {len(val_raw)} samples")

    # ================================================================
    # [2] 피처 엔지니어링
    # ================================================================
    test_cy  = make_clade_year_table(test_raw)
    val_cy   = make_clade_year_table(val_raw)
    train_df = build_supervised_dataset(test_cy)

    print(f"\nTraining data: {len(train_df)} clade-year rows")
    print(f"y=1 (minority): {train_df['y'].sum()}, y=0 (majority): {(train_df['y']==0).sum()}")

    # ================================================================
    # [3] Model Configuration
    # ================================================================
    print("\n" + "=" * 60)
    print("[3] Model Configuration")
    print("=" * 60)
    print(f"  Algorithm:    ElasticNet Logistic Regression")
    print(f"  l1_ratio:     {MODEL_CONFIG['l1_ratio']}")
    print(f"  C:            {MODEL_CONFIG['C']}")
    print(f"  class_weight: balanced (sklearn 자동 보정)")
    print(f"  threshold:    최적화 (min_recall >= 0.7)")
    print(f"  Features ({len(FEATURES)}): {FEATURES}")

    # ================================================================
    # [4] 시간순 교차검증 (Expanding Window)
    # ================================================================
    print("\n" + "=" * 60)
    print("[4] Time-Series Cross Validation (Expanding Window)")
    print("=" * 60)

    cv, macro, micro = cross_validate(train_df)

    # CV 결과 테이블 출력 (13기반 스타일)
    cv_df = pd.DataFrame(cv)
    print("\n[CV Results by Year]")
    print(cv_df.to_string(index=False))

    # ================================================================
    # [5] 최종 모델 학습 (Final Model)
    # ================================================================
    print("\n" + "=" * 60)
    print("[5] Final Model: Train on ALL data")
    print("=" * 60)

    X_all = train_df[FEATURES].fillna(0)
    y_all = train_df["y"]

    final_model = build_model()
    final_model.fit(X_all, y_all)

    # 전체 데이터에서 threshold 최적화
    y_proba_all = final_model.predict_proba(X_all)[:, 1]
    opt_threshold, thr_metrics = find_best_threshold(y_all, y_proba_all, min_recall=0.7)
    MODEL_CONFIG["threshold"] = opt_threshold

    print(f"Model trained on {len(X_all)} samples with {len(FEATURES)} features")
    print(f"Optimized threshold: {opt_threshold:.3f}")

    # Feature Coefficients (13기반 스타일)
    print("\n[Feature Coefficients]")
    coefs = final_model.named_steps["clf"].coef_[0]
    feat_coef = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": coefs,
        "Abs_Coef": np.abs(coefs),
    }).sort_values("Abs_Coef", ascending=False)
    print(feat_coef[["Feature", "Coefficient"]].to_string(index=False))

    # ================================================================
    # [6] Performance Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("[6] Performance Summary")
    print("=" * 60)

    # 평균
    print(f"\n[Average (fold별 평균)]")
    print(f"  AUC:          {macro['auc']:.3f}")
    print(f"  PR-AUC:       {macro['pr_auc']:.3f}")
    print(f"  Precision:    {macro['precision']:.3f}")
    print(f"  Recall:       {macro['recall']:.3f}")
    print(f"  F1:           {macro['f1']:.3f}")
    print(f"  Top-1 Hit:    {macro['hit1']:.2f}")
    print(f"  Top-3 Hit:    {macro['hit3']:.2f}")
    print(f"  Threshold:    {macro['threshold']:.3f}")

    # ================================================================
    # [7] 2025 Prediction
    # ================================================================
    THRESHOLD = opt_threshold

    print("\n" + "=" * 60)
    print(f"[7] 2025 Prediction (2026 CR Candidates) - threshold={THRESHOLD:.3f}")
    print("=" * 60)

    if not val_cy.empty:
        val_pred = val_cy.copy()
        val_pred["probability"]  = final_model.predict_proba(val_pred[FEATURES].fillna(0))[:, 1]
        val_pred["predicted_CR"] = (val_pred["probability"] >= THRESHOLD).astype(int)

        # dN/dS ratio
        val_pred["dN/dS"] = (val_pred["nonsyn_med"] / val_pred["syn_med"]).round(3)

        # Top 10 테이블 출력 (넘버링 + CR 강조)
        print("\n[2025 Clade Analysis - Top 10]")
        result = val_pred.sort_values("probability", ascending=False)[
            ["clade", "n", "freq", "nonsyn_med", "syn_med", "dN/dS",
             "pam_reversion_med", "probability", "predicted_CR"]
        ].head(10).reset_index(drop=True)
        result.index = [f"#{i+1}" for i in range(len(result))]
        result["predicted_CR"] = result["predicted_CR"].map({1: "<<< CR >>>", 0: ""})
        result.index.name = "rank"
        print(result.to_string())

        # Prediction Summary
        n_predicted  = val_pred["predicted_CR"].sum()
        total_clades = len(val_pred)
        print(f"\n[Prediction Summary]")
        print(f"  Total clades in 2025: {total_clades}")
        print(f"  Predicted as 2026 CR: {n_predicted} clades (threshold={THRESHOLD:.3f})")

        # Top 3 Candidates 상세 출력 (13기반 스타일)
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

    # ================================================================
    # [8] Historical CR Trend (최근 5년)
    # ================================================================
    print("=" * 60)
    print("[8] Historical CR Trend (Recent 5 Years)")
    print("=" * 60)

    cr_history = get_dominant_clade_by_year(test_cy)
    for year in sorted(cr_history.index)[-5:]:
        clade = cr_history[year]
        year_data = test_raw[test_raw["year"] == year]
        total = len(year_data)
        clade_count = len(year_data[year_data["clade"] == clade])
        pct = clade_count / total * 100 if total > 0 else 0
        print(f"  {year}: {clade} ({clade_count}/{total}, {pct:.1f}%)")

    # ================================================================
    # [9] Backtesting (2008 ~ 2024)
    # ================================================================
    print("\n" + "=" * 60)
    print("[9] Backtesting (Expanding Window)")
    print("=" * 60)

    cr_all = get_dominant_clade_by_year(test_cy)  # 연도별 실제 CR
    bt_results = []
    skipped = []

    for val_year in range(2008, 2025):
        # 학습 데이터: val_year 이전 연도만
        train_cy_bt = test_cy[test_cy["year"] < val_year].copy()
        val_cy_bt   = test_cy[test_cy["year"] == val_year].copy()

        if val_cy_bt.empty or len(train_cy_bt) < 3:
            skipped.append(val_year)
            continue

        # 지도학습 데이터 구성
        cr_bt = get_dominant_clade_by_year(train_cy_bt)
        sup_bt = train_cy_bt.copy()
        sup_bt["CR_next"] = sup_bt["year"].map(lambda y: cr_bt.get(y + 1, None))
        sup_bt = sup_bt.dropna(subset=["CR_next"]).copy()
        sup_bt["y"] = (sup_bt["clade"] == sup_bt["CR_next"]).astype(int)

        if len(sup_bt) < 3 or sup_bt["y"].nunique() < 2:
            skipped.append(val_year)
            continue

        # 모델 학습 및 예측
        X_bt = sup_bt[FEATURES].fillna(0)
        y_bt = sup_bt["y"]
        model_bt = build_model()
        model_bt.fit(X_bt, y_bt)

        val_cy_bt["probability"] = model_bt.predict_proba(val_cy_bt[FEATURES].fillna(0))[:, 1]
        ranked = val_cy_bt.sort_values("probability", ascending=False)
        top3_clades = ranked.head(3)["clade"].tolist()

        actual = cr_all.get(val_year, "?")
        hit1 = "O" if len(top3_clades) > 0 and top3_clades[0] == actual else "X"
        hit3 = "O" if actual in top3_clades else "X"

        bt_results.append({
            "year": val_year, "actual_CR": actual,
            "pred_1": top3_clades[0] if len(top3_clades) > 0 else "",
            "pred_2": top3_clades[1] if len(top3_clades) > 1 else "",
            "pred_3": top3_clades[2] if len(top3_clades) > 2 else "",
            "hit1": hit1, "hit3": hit3,
        })

    bt_df = pd.DataFrame(bt_results)
    n_total = len(bt_df)
    n_hit1 = (bt_df["hit1"] == "O").sum()
    n_hit3 = (bt_df["hit3"] == "O").sum()

    if skipped:
        print(f"\n  * Skipped: {skipped} (학습 데이터 부족)")

    # 2025: VAL 데이터에서 실제 CR 확인 + 모델 예측과 비교
    val_cr_2025 = get_dominant_clade_by_year(val_cy).get(2025, "?") if not val_cy.empty else "?"
    top3_2025 = val_pred.sort_values("probability", ascending=False).head(3)["clade"].tolist() if not val_cy.empty else []
    hit1_2025 = "O" if len(top3_2025) > 0 and top3_2025[0] == val_cr_2025 else "X"
    hit3_2025 = "O" if val_cr_2025 in top3_2025 else "X"
    bt_results.append({
        "year": 2025, "actual_CR": val_cr_2025,
        "pred_1": top3_2025[0] if len(top3_2025) > 0 else "",
        "pred_2": top3_2025[1] if len(top3_2025) > 1 else "",
        "pred_3": top3_2025[2] if len(top3_2025) > 2 else "",
        "hit1": hit1_2025, "hit3": hit3_2025,
    })

    # 2026: 실제 데이터 없음 (예측만)
    bt_results.append({
        "year": 2026, "actual_CR": "?",
        "pred_1": top3_2025[0] if len(top3_2025) > 0 else "",
        "pred_2": top3_2025[1] if len(top3_2025) > 1 else "",
        "pred_3": top3_2025[2] if len(top3_2025) > 2 else "",
        "hit1": "?", "hit3": "?",
    })

    bt_df = pd.DataFrame(bt_results)

    # Summary는 2025까지만 (실제 검증 가능한 연도)
    bt_verified = bt_df[bt_df["hit1"].isin(["O", "X"])]
    n_total = len(bt_verified)
    n_hit1 = (bt_verified["hit1"] == "O").sum()
    n_hit3 = (bt_verified["hit3"] == "O").sum()

    print(f"\n  {'Year':<6} {'Actual CR':<10} {'#1 Pred':<10} {'#2 Pred':<10} {'#3 Pred':<10} {'H@1':>4} {'H@3':>4}")
    print(f"  {'-'*58}")
    for _, r in bt_df.iterrows():
        yr = int(r["year"])
        if yr == 2026:
            # 구분선 + 2026 예측
            print(f"  {'─'*58}")
            print(f"  {yr:<6} {'?':<10} {r['pred_1']:<10} {r['pred_2']:<10} {r['pred_3']:<10} {'?':>4} {'?':>4} (예측)")
        elif yr == 2025:
            print(f"  {'─'*58}")
            mark = " <<<" if r["hit1"] == "O" else (" *" if r["hit3"] == "O" else "")
            print(f"  {yr:<6} {r['actual_CR']:<10} {r['pred_1']:<10} {r['pred_2']:<10} {r['pred_3']:<10} {r['hit1']:>4} {r['hit3']:>4}{mark}")
        else:
            mark = " <<<" if r["hit1"] == "O" else (" *" if r["hit3"] == "O" else "")
            print(f"  {yr:<6} {r['actual_CR']:<10} {r['pred_1']:<10} {r['pred_2']:<10} {r['pred_3']:<10} {r['hit1']:>4} {r['hit3']:>4}{mark}")

    print(f"\n  [Backtest Summary] ({n_total}개년 검증, {len(skipped)}개년 제외)")
    print(f"    Top-1 Hit: {n_hit1}/{n_total} ({n_hit1/n_total:.0%})")
    print(f"    Top-3 Hit: {n_hit3}/{n_total} ({n_hit3/n_total:.0%})")

    # ================================================================
    # [10] MODEL SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("[10] MODEL SUMMARY")
    print("=" * 60)
    print(f"""
  Algorithm:     ElasticNet Logistic Regression
  Optimization:  Balanced weight + Threshold 최적화
  Features:      {len(FEATURES)} features

  Performance (Average):
    - AUC:          {macro['auc']:.3f}
    - PR-AUC:       {macro['pr_auc']:.3f}
    - Precision:    {macro['precision']:.3f}
    - Recall:       {macro['recall']:.3f}
    - F1:           {macro['f1']:.3f}
    - Top-1 Hit:    {macro['hit1']:.0%}
    - Top-3 Hit:    {macro['hit3']:.0%}

  Threshold:       {opt_threshold:.3f} (F1 최대화, min_recall >= 0.7)
""")
    print("=" * 60)
    print("Done!")
    print("=" * 60)
