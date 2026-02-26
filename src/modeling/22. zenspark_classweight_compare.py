"""
H3N2 Clade Prediction - class_weight 비교 실험
================================================================
목적: balanced vs 커스텀 class_weight 조합별 성능 비교
기반: _##zenspark_F2_balanced_202512_final.py
비교 대상:
  A) balanced (sklearn 자동)
  B) {0:1, 1:5}  - balanced와 유사한 수준
  C) {0:1, 1:3}  - 중간 보정
  D) {0:1, 1:2}  - 약한 보정
  E) {0:1, 1:1.5} - precision 중시
  F) None         - 가중치 없음 (baseline)

재현성: random_state=42 고정
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
# 1. 설정
# ================================================================


TEST_PATH = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\01. TEST_KJC.csv"
VAL_PATH  = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\02. VAL_KJC.csv"

MODEL_CONFIG = {
    "l1_ratio": 0.8,
    "C": 0.4,
    "max_iter": 3000,
}

FEATURES = [
    "n", "freq", "freq_prev", "freq_delta",
    "nonsyn_med", "syn_med", "novelty_med", "pam_reversion_med",
]

# ================================================================
# 비교할 class_weight 조합 정의
# ================================================================
WEIGHT_CONFIGS = {
    "A) balanced":   "balanced",
    "B) {0:1,1:5}":  {0: 1, 1: 5},
    "C) {0:1,1:3}":  {0: 1, 1: 3},
    "D) {0:1,1:2}":  {0: 1, 1: 2},
    "E) {0:1,1:1.5}": {0: 1, 1: 1.5},
    "F) None":       None,
}


# ================================================================
# 2. 데이터 로딩 (원본과 동일)
# ================================================================

def load_nextclade(path: str) -> pd.DataFrame:
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

    if "qc.overallStatus" in df.columns:
        before = len(df)
        df = df[df["qc.overallStatus"].astype(str).str.lower() == "good"].copy()
        print(f"    QC 필터 후: {len(df)}/{before}개")

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

    df["nonsyn"]    = pd.to_numeric(df.get("totalAminoacidSubstitutions"), errors="coerce")
    df["total_subs"] = pd.to_numeric(df.get("totalSubstitutions"), errors="coerce")
    df["syn_proxy"]  = df["total_subs"] - df["nonsyn"]
    df["novelty"]    = df["nonsyn"].fillna(0) + 0.2 * df["syn_proxy"].fillna(0)
    df["pam_reversion"] = pd.to_numeric(
        df.get("privateAaMutations.totalReversionSubstitutions"), errors="coerce"
    )

    keep = ["seqName", "clade", "year", "nonsyn", "syn_proxy", "novelty", "pam_reversion"]
    df = df[keep].dropna(subset=["clade", "year"]).copy()
    df["year"] = df["year"].astype(int)
    print(f"    최종 데이터: {len(df)}개 시퀀스")
    return df


# ================================================================
# 3. 피처 엔지니어링 (원본과 동일)
# ================================================================

def make_clade_year_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["year", "clade"] + FEATURES)

    g = (df.groupby(["year", "clade"])
           .agg(
               n=("seqName", "count"),
               nonsyn_med=("nonsyn", "median"),
               syn_med=("syn_proxy", "median"),
               novelty_med=("novelty", "median"),
               pam_reversion_med=("pam_reversion", "median"),
           )
           .reset_index())

    totals = g.groupby("year")["n"].sum().reset_index(name="year_total")
    g = g.merge(totals, on="year", how="left")
    g["freq"] = g["n"] / g["year_total"]

    g = g.sort_values(["clade", "year"])
    g["freq_prev"]  = g.groupby("clade")["freq"].shift(1).fillna(0)
    g["freq_delta"] = (g["freq"] - g["freq_prev"]).fillna(0)

    return g


def get_dominant_clade_by_year(clade_year: pd.DataFrame) -> pd.Series:
    idx = clade_year.groupby("year")["n"].idxmax()
    return clade_year.loc[idx, ["year", "clade"]].set_index("year")["clade"]


def build_supervised_dataset(clade_year: pd.DataFrame) -> pd.DataFrame:
    cr = get_dominant_clade_by_year(clade_year)
    df = clade_year.copy()
    df["CR_next"] = df["year"].map(lambda y: cr.get(y + 1, None))
    df = df.dropna(subset=["CR_next"]).copy()
    df["y"] = (df["clade"] == df["CR_next"]).astype(int)
    return df


# ================================================================
# 4. 모델 (class_weight를 파라미터로 받음)
# ================================================================

def build_model(class_weight="balanced"):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=MODEL_CONFIG["l1_ratio"],
            C=MODEL_CONFIG["C"],
            class_weight=class_weight,
            max_iter=MODEL_CONFIG["max_iter"],
            random_state=42,
        )),
    ])


def find_best_threshold(y_true, y_proba, min_recall=0.7):
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
    true_clade = df_year["CR_next"].iloc[0]
    topk = df_year.sort_values(proba_col, ascending=False).head(k)["clade"].tolist()
    return int(true_clade in topk)


# ================================================================
# 5. CV + Backtest를 하나의 class_weight로 실행
# ================================================================

def run_cv(train_df, class_weight):
    """시간순 교차검증 실행 후 macro 평균 반환"""
    years = sorted(train_df["year"].unique())
    MIN_TRAIN_YEARS = 3

    cv = defaultdict(list)
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

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        model = build_model(class_weight)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]

        all_y_true.extend(y_val.tolist())
        all_y_proba.extend(y_proba.tolist())

        opt_thr, metrics = find_best_threshold(y_val, y_proba, min_recall=0.7)

        auc    = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)

        vf = val_fold.copy()
        vf["p_pred"] = y_proba
        hit1 = topk_hit(vf, "p_pred", k=1)
        hit3 = topk_hit(vf, "p_pred", k=3)

        cv["val_year"].append(val_year)
        cv["threshold"].append(opt_thr)
        cv["auc"].append(auc)
        cv["pr_auc"].append(pr_auc)
        cv["precision"].append(metrics.get("precision", 0))
        cv["recall"].append(metrics.get("recall", 0))
        cv["f1"].append(metrics.get("f1", 0))
        cv["hit1"].append(hit1)
        cv["hit3"].append(hit3)

    metric_keys = ["threshold", "auc", "pr_auc", "precision", "recall", "f1", "hit1", "hit3"]
    macro = {k: np.mean(cv[k]) if cv[k] else 0 for k in metric_keys}

    return cv, macro


def run_backtest(test_cy, class_weight):
    """Backtest (2008~2024) 실행 후 Hit Rate 반환"""
    cr_all = get_dominant_clade_by_year(test_cy)
    bt_results = []
    skipped = []

    for val_year in range(2008, 2025):
        train_cy_bt = test_cy[test_cy["year"] < val_year].copy()
        val_cy_bt   = test_cy[test_cy["year"] == val_year].copy()

        if val_cy_bt.empty or len(train_cy_bt) < 3:
            skipped.append(val_year)
            continue

        cr_bt = get_dominant_clade_by_year(train_cy_bt)
        sup_bt = train_cy_bt.copy()
        sup_bt["CR_next"] = sup_bt["year"].map(lambda y: cr_bt.get(y + 1, None))
        sup_bt = sup_bt.dropna(subset=["CR_next"]).copy()
        sup_bt["y"] = (sup_bt["clade"] == sup_bt["CR_next"]).astype(int)

        if len(sup_bt) < 3 or sup_bt["y"].nunique() < 2:
            skipped.append(val_year)
            continue

        X_bt = sup_bt[FEATURES].fillna(0)
        y_bt = sup_bt["y"]
        model_bt = build_model(class_weight)
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
    n_hit1 = (bt_df["hit1"] == "O").sum() if n_total > 0 else 0
    n_hit3 = (bt_df["hit3"] == "O").sum() if n_total > 0 else 0

    return bt_df, n_hit1, n_hit3, n_total, skipped


def run_2025_prediction(train_df, val_cy, class_weight):
    """전체 학습 후 2025 예측"""
    X_all = train_df[FEATURES].fillna(0)
    y_all = train_df["y"]

    model = build_model(class_weight)
    model.fit(X_all, y_all)

    y_proba_all = model.predict_proba(X_all)[:, 1]
    opt_thr, _ = find_best_threshold(y_all, y_proba_all, min_recall=0.7)

    if not val_cy.empty:
        val_pred = val_cy.copy()
        val_pred["probability"] = model.predict_proba(val_pred[FEATURES].fillna(0))[:, 1]
        val_pred["predicted_CR"] = (val_pred["probability"] >= opt_thr).astype(int)
        top3 = val_pred.sort_values("probability", ascending=False).head(3)
        return model, opt_thr, val_pred, top3
    return model, opt_thr, None, None


# ================================================================
# 6. 메인 실행
# ================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("H3N2 Clade Prediction - class_weight 비교 실험")
    print("=" * 70)

    # ─── 데이터 로드 ───
    print("\n[1] 데이터 로드")
    test_raw = load_nextclade(TEST_PATH)
    val_raw  = load_nextclade(VAL_PATH)

    if test_raw.empty or val_raw.empty:
        print("  데이터 로드 실패. 종료합니다.")
        exit()

    # ─── 피처 엔지니어링 ───
    test_cy  = make_clade_year_table(test_raw)
    val_cy   = make_clade_year_table(val_raw)
    train_df = build_supervised_dataset(test_cy)

    n_pos = train_df["y"].sum()
    n_neg = (train_df["y"] == 0).sum()
    n_total_samples = len(train_df)
    print(f"\nTraining data: {n_total_samples} rows (y=1: {n_pos}, y=0: {n_neg})")

    # balanced가 실제로 적용하는 가중치 계산
    w0_balanced = n_total_samples / (2 * n_neg)
    w1_balanced = n_total_samples / (2 * n_pos)
    print(f"\n[참고] balanced 자동 가중치:")
    print(f"  weight(y=0) = {n_total_samples} / (2 × {n_neg}) = {w0_balanced:.3f}")
    print(f"  weight(y=1) = {n_total_samples} / (2 × {n_pos}) = {w1_balanced:.3f}")
    print(f"  y=1/y=0 비율 = {w1_balanced/w0_balanced:.1f}x")

    # ================================================================
    # [2] 각 class_weight 조합별 실험
    # ================================================================
    print("\n" + "=" * 70)
    print("[2] class_weight 조합별 CV + Backtest 실행")
    print("=" * 70)

    all_results = []

    for name, cw in WEIGHT_CONFIGS.items():
        print(f"\n{'─'*70}")
        print(f"  실험: {name}")
        print(f"{'─'*70}")

        # --- CV ---
        cv, macro = run_cv(train_df, cw)
        cv_df = pd.DataFrame(cv)

        print(f"\n  [CV Results by Year]")
        if not cv_df.empty:
            print("  " + cv_df.to_string(index=False).replace("\n", "\n  "))

        # --- Backtest ---
        bt_df, bt_hit1, bt_hit3, bt_total, bt_skipped = run_backtest(test_cy, cw)

        # --- 2025 Prediction ---
        model, opt_thr, val_pred, top3 = run_2025_prediction(train_df, val_cy, cw)

        # 2025 검증 (VAL 데이터의 실제 CR과 비교)
        val_cr_2025 = get_dominant_clade_by_year(val_cy).get(2025, "?") if not val_cy.empty else "?"
        top3_clades_2025 = []
        if top3 is not None:
            top3_clades_2025 = top3["clade"].tolist()

        bt_hit1_with_2025 = bt_hit1 + (1 if len(top3_clades_2025) > 0 and top3_clades_2025[0] == val_cr_2025 else 0)
        bt_hit3_with_2025 = bt_hit3 + (1 if val_cr_2025 in top3_clades_2025 else 0)
        bt_total_with_2025 = bt_total + 1

        # Top-3 예측 결과 출력
        if top3 is not None:
            print(f"\n  [2025 → 2026 CR Top-3 예측]")
            for i, (_, row) in enumerate(top3.iterrows(), 1):
                cr_mark = " *** CR ***" if row["predicted_CR"] == 1 else ""
                print(f"    #{i}: {row['clade']} (prob={row['probability']:.3f}){cr_mark}")

        # 결과 저장
        all_results.append({
            "config": name,
            "AUC": macro["auc"],
            "PR-AUC": macro["pr_auc"],
            "Precision": macro["precision"],
            "Recall": macro["recall"],
            "F1": macro["f1"],
            "CV_Hit@1": macro["hit1"],
            "CV_Hit@3": macro["hit3"],
            "Threshold": opt_thr,
            "BT_Hit@1": f"{bt_hit1_with_2025}/{bt_total_with_2025}",
            "BT_Hit@1%": bt_hit1_with_2025 / bt_total_with_2025 if bt_total_with_2025 > 0 else 0,
            "BT_Hit@3": f"{bt_hit3_with_2025}/{bt_total_with_2025}",
            "BT_Hit@3%": bt_hit3_with_2025 / bt_total_with_2025 if bt_total_with_2025 > 0 else 0,
            "pred_1": top3_clades_2025[0] if len(top3_clades_2025) > 0 else "",
            "pred_2": top3_clades_2025[1] if len(top3_clades_2025) > 1 else "",
            "pred_3": top3_clades_2025[2] if len(top3_clades_2025) > 2 else "",
        })

    # ================================================================
    # [3] 종합 비교 테이블
    # ================================================================
    print("\n" + "=" * 70)
    print("[3] 종합 비교 테이블")
    print("=" * 70)

    comp_df = pd.DataFrame(all_results)

    # --- CV 성능 비교 ---
    print("\n[CV Performance (Macro Average)]")
    cv_cols = ["config", "AUC", "PR-AUC", "Precision", "Recall", "F1", "Threshold"]
    cv_display = comp_df[cv_cols].copy()
    for col in ["AUC", "PR-AUC", "Precision", "Recall", "F1", "Threshold"]:
        cv_display[col] = cv_display[col].map(lambda x: f"{x:.3f}")
    print(cv_display.to_string(index=False))

    # --- CV Hit Rate 비교 ---
    print("\n[CV Hit Rate (Macro Average)]")
    hit_cols = ["config", "CV_Hit@1", "CV_Hit@3"]
    hit_display = comp_df[hit_cols].copy()
    for col in ["CV_Hit@1", "CV_Hit@3"]:
        hit_display[col] = hit_display[col].map(lambda x: f"{x:.0%}")
    print(hit_display.to_string(index=False))

    # --- Backtest 비교 ---
    print("\n[Backtest Hit Rate (2008~2025)]")
    bt_cols = ["config", "BT_Hit@1", "BT_Hit@1%", "BT_Hit@3", "BT_Hit@3%"]
    bt_display = comp_df[bt_cols].copy()
    bt_display["BT_Hit@1%"] = bt_display["BT_Hit@1%"].map(lambda x: f"{x:.0%}")
    bt_display["BT_Hit@3%"] = bt_display["BT_Hit@3%"].map(lambda x: f"{x:.0%}")
    print(bt_display.to_string(index=False))

    # --- 2026 CR 예측 비교 ---
    print("\n[2026 CR 예측 (2025 데이터 기반)]")
    pred_cols = ["config", "Threshold", "pred_1", "pred_2", "pred_3"]
    pred_display = comp_df[pred_cols].copy()
    pred_display["Threshold"] = pred_display["Threshold"].map(lambda x: f"{x:.3f}")
    pred_display.columns = ["config", "Threshold", "#1 Pred", "#2 Pred", "#3 Pred"]
    print(pred_display.to_string(index=False))

    # ================================================================
    # [4] 최적 조합 추천
    # ================================================================
    print("\n" + "=" * 70)
    print("[4] 최적 조합 추천")
    print("=" * 70)

    # F1 기준 최적
    best_f1_idx = comp_df["F1"].astype(float).idxmax()
    best_f1_row = comp_df.loc[best_f1_idx]
    print(f"\n  [F1 최고]  {best_f1_row['config']}")
    print(f"    F1={best_f1_row['F1']:.3f}, Prec={best_f1_row['Precision']:.3f}, "
          f"Rec={best_f1_row['Recall']:.3f}, BT_H@1={best_f1_row['BT_Hit@1']}")

    # Precision 기준 최적 (recall >= 0.7 조건)
    valid = comp_df[comp_df["Recall"].astype(float) >= 0.7]
    if not valid.empty:
        best_prec_idx = valid["Precision"].astype(float).idxmax()
        best_prec_row = valid.loc[best_prec_idx]
        print(f"\n  [Precision 최고 (Recall≥0.7)]  {best_prec_row['config']}")
        print(f"    Prec={best_prec_row['Precision']:.3f}, Rec={best_prec_row['Recall']:.3f}, "
              f"F1={best_prec_row['F1']:.3f}, BT_H@1={best_prec_row['BT_Hit@1']}")

    # Backtest Hit@1 기준 최적
    best_bt_idx = comp_df["BT_Hit@1%"].astype(float).idxmax()
    best_bt_row = comp_df.loc[best_bt_idx]
    print(f"\n  [Backtest Top-1 Hit 최고]  {best_bt_row['config']}")
    print(f"    BT_H@1={best_bt_row['BT_Hit@1']} ({best_bt_row['BT_Hit@1%']:.0%}), "
          f"F1={best_bt_row['F1']:.3f}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
