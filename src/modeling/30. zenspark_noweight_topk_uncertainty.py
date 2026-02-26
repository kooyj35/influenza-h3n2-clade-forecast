import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ================================================================
# 1. Configuration
# ================================================================

TEST_PATH = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\01. TEST_KJC.csv"
VAL_PATH = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\02. VAL_KJC.csv"

FEATURES = [
    "n",
    "freq",
    "freq_prev",
    "freq_delta",
    "freq_delta_2y",
    "freq_accel",
    "rolling_median_freq",
    "nonsyn_med",
    "syn_med",
    "novelty_med",
    "pam_reversion_med",
]

MODEL_GRID = {
    "l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0],
    "C": [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0],
}
MIN_RECALL_GRID = [0.60, 0.70, 0.80]
THRESHOLD_GRID = np.arange(0.10, 0.91, 0.05)
MAX_ITER = 5000
RANDOM_STATE = 42
MIN_TRAIN_YEARS = 3
RUN_NESTED_DIAGNOSTIC = False


# ================================================================
# 2. Data Loading + Feature Engineering
# ================================================================

def load_nextclade(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";", low_memory=False)
        print(f"  [Loaded] {path}")
        print(f"    Raw rows: {df.shape[0]}")
    except FileNotFoundError:
        print(f"  [Error] File not found: {path}")
        return pd.DataFrame()
    except Exception as exc:
        print(f"  [Error] {exc}")
        return pd.DataFrame()

    if "qc.overallStatus" in df.columns:
        before = len(df)
        df = df[df["qc.overallStatus"].astype(str).str.lower() == "good"].copy()
        print(f"    After QC(good): {len(df)}/{before}")

    s = df["seqName"].astype(str)
    year_from_name = s.str.extract(r"/(\d{4})\|")[0].pipe(pd.to_numeric, errors="coerce")
    year_from_tail = s.str.split("|").str[-1].pipe(pd.to_numeric, errors="coerce")
    df["year"] = year_from_name.where(
        year_from_name.between(2005, 2025),
        year_from_tail.where(year_from_tail.between(2005, 2025)),
    )
    df = df[(df["year"] >= 2005) & (df["year"] <= 2025)].copy()

    if df.empty:
        print("    [Warn] No rows after year filter")
        return pd.DataFrame()

    df["nonsyn"] = pd.to_numeric(df.get("totalAminoacidSubstitutions"), errors="coerce")
    df["total_subs"] = pd.to_numeric(df.get("totalSubstitutions"), errors="coerce")
    df["syn_proxy"] = df["total_subs"] - df["nonsyn"]
    df["novelty"] = df["nonsyn"].fillna(0) + 0.2 * df["syn_proxy"].fillna(0)
    df["pam_reversion"] = pd.to_numeric(
        df.get("privateAaMutations.totalReversionSubstitutions"), errors="coerce"
    )

    keep = ["seqName", "clade", "year", "nonsyn", "syn_proxy", "novelty", "pam_reversion"]
    df = df[keep].dropna(subset=["clade", "year"]).copy()
    df["year"] = df["year"].astype(int)

    print(f"    Final rows: {len(df)}")
    return df


def make_clade_year_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["year", "clade"] + FEATURES)

    g = (
        df.groupby(["year", "clade"])
        .agg(
            n=("seqName", "count"),
            nonsyn_med=("nonsyn", "median"),
            syn_med=("syn_proxy", "median"),
            novelty_med=("novelty", "median"),
            pam_reversion_med=("pam_reversion", "median"),
        )
        .reset_index()
    )

    totals = g.groupby("year")["n"].sum().reset_index(name="year_total")
    g = g.merge(totals, on="year", how="left")
    g["freq"] = g["n"] / g["year_total"]

    g = g.sort_values(["clade", "year"])
    g["freq_prev"] = g.groupby("clade")["freq"].shift(1).fillna(0)
    g["freq_delta"] = (g["freq"] - g["freq_prev"]).fillna(0)
    g["freq_prev_2y"] = g.groupby("clade")["freq"].shift(2).fillna(0)
    g["freq_delta_2y"] = (g["freq"] - g["freq_prev_2y"]).fillna(0)
    g["freq_accel"] = (g["freq_delta"] - g.groupby("clade")["freq_delta"].shift(1).fillna(0)).fillna(0)
    g["rolling_median_freq"] = (
        g.groupby("clade")["freq"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).median())
        .fillna(0)
    )
    g = g.drop(columns=["freq_prev_2y"])

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
# 3. Modeling Helpers
# ================================================================

def build_model(l1_ratio: float, c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=l1_ratio,
                    C=c_value,
                    class_weight=None,
                    max_iter=MAX_ITER,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def topk_hit(df_year: pd.DataFrame, proba_col: str, k: int) -> int:
    true_clade = df_year["CR_next"].iloc[0]
    topk = df_year.sort_values(proba_col, ascending=False).head(k)["clade"].tolist()
    return int(true_clade in topk)


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, min_recall: float):
    best_thr = 0.5
    best_prf = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for thr in THRESHOLD_GRID:
        y_pred = (y_proba >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec < min_recall:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_prf["f1"]:
            best_thr = float(thr)
            best_prf = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

    return best_thr, best_prf


def find_conservative_flag_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    base_threshold: float,
    min_recall: float,
    min_precision: float = 0.60,
):
    best_thr = base_threshold
    best_f1 = -1.0
    found = False

    for thr in THRESHOLD_GRID:
        y_pred = (y_proba >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec < min_recall or prec < min_precision:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            found = True

    if not found:
        return float(base_threshold)
    return float(max(base_threshold, best_thr))


@dataclass
class SearchResult:
    l1_ratio: float
    c_value: float
    min_recall: float
    threshold: float
    top1: float
    top3: float
    auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    n_folds: int
    backtest_hit1: int
    backtest_hit3: int
    backtest_n: int


def run_expanding_cv(train_df: pd.DataFrame, l1_ratio: float, c_value: float, min_recall: float, min_train_years: int):
    years = sorted(train_df["year"].unique())
    rows = []
    for i in range(min_train_years, len(years)):
        val_year = years[i]
        train_years = years[:i]

        train_fold = train_df[train_df["year"].isin(train_years)]
        val_fold = train_df[train_df["year"] == val_year]

        if train_fold.empty or val_fold.empty:
            continue
        if train_fold["y"].nunique() < 2 or val_fold["y"].nunique() < 2:
            continue

        model = build_model(l1_ratio=l1_ratio, c_value=c_value)
        model.fit(train_fold[FEATURES].fillna(0), train_fold["y"])
        y_proba = model.predict_proba(val_fold[FEATURES].fillna(0))[:, 1]

        vf = val_fold.copy()
        vf["p_pred"] = y_proba

        rows.append(
            {
                "val_year": val_year,
                "train_size": len(train_fold),
                "val_size": len(val_fold),
                "auc": roc_auc_score(val_fold["y"], y_proba),
                "pr_auc": average_precision_score(val_fold["y"], y_proba),
                "hit1": topk_hit(vf, "p_pred", 1),
                "hit3": topk_hit(vf, "p_pred", 3),
                "y_true": val_fold["y"].to_numpy(),
                "y_proba": y_proba,
            }
        )

    if len(rows) == 0:
        return None

    thresholds = []
    precision_vals, recall_vals, f1_vals = [], [], []
    for r in rows:
        threshold, _ = find_best_threshold(r["y_true"], r["y_proba"], min_recall=min_recall)
        y_pred = (r["y_proba"] >= threshold).astype(int)
        p = precision_score(r["y_true"], y_pred, zero_division=0)
        rc = recall_score(r["y_true"], y_pred, zero_division=0)
        f = f1_score(r["y_true"], y_pred, zero_division=0)
        thresholds.append(float(threshold))
        precision_vals.append(float(p))
        recall_vals.append(float(rc))
        f1_vals.append(float(f))
        r["precision"] = p
        r["recall"] = rc
        r["f1"] = f
        r["threshold"] = threshold

    cv_df = pd.DataFrame(rows).drop(columns=["y_true", "y_proba"])

    summary = SearchResult(
        l1_ratio=float(l1_ratio),
        c_value=float(c_value),
        min_recall=float(min_recall),
        threshold=float(np.mean(thresholds)),
        top1=float(cv_df["hit1"].mean()),
        top3=float(cv_df["hit3"].mean()),
        auc=float(cv_df["auc"].mean()),
        pr_auc=float(cv_df["pr_auc"].mean()),
        precision=float(np.mean(precision_vals)),
        recall=float(np.mean(recall_vals)),
        f1=float(np.mean(f1_vals)),
        n_folds=int(len(cv_df)),
        backtest_hit1=-1,
        backtest_hit3=-1,
        backtest_n=-1,
    )

    return cv_df, summary


def run_backtest(test_cy: pd.DataFrame, val_cy: pd.DataFrame, l1_ratio: float, c_value: float):
    cr_all = get_dominant_clade_by_year(test_cy)
    bt = []

    for val_year in range(2008, 2025):
        train_cy_bt = test_cy[test_cy["year"] < val_year].copy()
        val_cy_bt = test_cy[test_cy["year"] == val_year].copy()

        if val_cy_bt.empty or len(train_cy_bt) < 3:
            continue

        cr_bt = get_dominant_clade_by_year(train_cy_bt)
        sup_bt = train_cy_bt.copy()
        sup_bt["CR_next"] = sup_bt["year"].map(lambda y: cr_bt.get(y + 1, None))
        sup_bt = sup_bt.dropna(subset=["CR_next"]).copy()
        sup_bt["y"] = (sup_bt["clade"] == sup_bt["CR_next"]).astype(int)
        if len(sup_bt) < 3 or sup_bt["y"].nunique() < 2:
            continue

        model_bt = build_model(l1_ratio, c_value)
        model_bt.fit(sup_bt[FEATURES].fillna(0), sup_bt["y"])
        val_cy_bt["probability"] = model_bt.predict_proba(val_cy_bt[FEATURES].fillna(0))[:, 1]
        top3 = val_cy_bt.sort_values("probability", ascending=False).head(3)["clade"].tolist()
        actual = cr_all.get(val_year, "?")
        bt.append((int(len(top3) > 0 and top3[0] == actual), int(actual in top3)))

    # 2025 verification on VAL
    sup_all = build_supervised_dataset(test_cy)
    model_all = build_model(l1_ratio, c_value)
    model_all.fit(sup_all[FEATURES].fillna(0), sup_all["y"])
    val_tmp = val_cy.copy()
    val_tmp["probability"] = model_all.predict_proba(val_tmp[FEATURES].fillna(0))[:, 1]
    top3_2025 = val_tmp.sort_values("probability", ascending=False).head(3)["clade"].tolist()
    actual_2025 = get_dominant_clade_by_year(val_cy).get(2025, "?") if not val_cy.empty else "?"
    bt.append((int(len(top3_2025) > 0 and top3_2025[0] == actual_2025), int(actual_2025 in top3_2025)))

    bt_arr = np.array(bt)
    return int(bt_arr[:, 0].sum()), int(bt_arr[:, 1].sum()), int(len(bt_arr))


def search_best_params(
    train_df: pd.DataFrame,
    test_cy: pd.DataFrame | None = None,
    val_cy: pd.DataFrame | None = None,
    min_train_years: int = MIN_TRAIN_YEARS,
):
    best_summary = None
    best_cv_df = None

    for l1_ratio, c_value, min_recall in product(
        MODEL_GRID["l1_ratio"], MODEL_GRID["C"], MIN_RECALL_GRID
    ):
        out = run_expanding_cv(train_df, l1_ratio, c_value, min_recall, min_train_years)
        if out is None:
            continue
        cv_df, summary = out
        if test_cy is not None and val_cy is not None:
            hit1, hit3, n_bt = run_backtest(test_cy, val_cy, l1_ratio, c_value)
            summary.backtest_hit1 = hit1
            summary.backtest_hit3 = hit3
            summary.backtest_n = n_bt

        if best_summary is None:
            best_summary, best_cv_df = summary, cv_df
            continue

        if test_cy is not None and val_cy is not None:
            current_key = (
                summary.backtest_hit1 >= 10 and summary.backtest_hit3 >= 12,
                summary.top3,
                summary.top1,
                summary.f1,
                summary.auc,
            )
            best_key = (
                best_summary.backtest_hit1 >= 10 and best_summary.backtest_hit3 >= 12,
                best_summary.top3,
                best_summary.top1,
                best_summary.f1,
                best_summary.auc,
            )
        else:
            current_key = (summary.top3, summary.top1, summary.f1, summary.auc)
            best_key = (best_summary.top3, best_summary.top1, best_summary.f1, best_summary.auc)
        if current_key > best_key:
            best_summary, best_cv_df = summary, cv_df

    if best_summary is None:
        raise RuntimeError("No valid parameter combination found.")

    return best_cv_df, best_summary


def nested_cv_diagnostic(train_df: pd.DataFrame):
    years = sorted(train_df["year"].unique())
    outer_rows = []

    for i in range(MIN_TRAIN_YEARS, len(years)):
        val_year = years[i]
        train_outer = train_df[train_df["year"].isin(years[:i])]
        val_outer = train_df[train_df["year"] == val_year]

        if train_outer.empty or val_outer.empty:
            continue
        if train_outer["y"].nunique() < 2 or val_outer["y"].nunique() < 2:
            continue

        try:
            _, best_inner = search_best_params(train_outer, min_train_years=2)
        except RuntimeError:
            continue

        model = build_model(best_inner.l1_ratio, best_inner.c_value)
        model.fit(train_outer[FEATURES].fillna(0), train_outer["y"])
        y_proba = model.predict_proba(val_outer[FEATURES].fillna(0))[:, 1]
        y_pred = (y_proba >= best_inner.threshold).astype(int)

        vf = val_outer.copy()
        vf["p_pred"] = y_proba
        outer_rows.append(
            {
                "val_year": val_year,
                "hit1": topk_hit(vf, "p_pred", 1),
                "hit3": topk_hit(vf, "p_pred", 3),
                "precision": precision_score(val_outer["y"], y_pred, zero_division=0),
                "recall": recall_score(val_outer["y"], y_pred, zero_division=0),
                "f1": f1_score(val_outer["y"], y_pred, zero_division=0),
                "auc": roc_auc_score(val_outer["y"], y_proba),
                "pr_auc": average_precision_score(val_outer["y"], y_proba),
            }
        )

    if len(outer_rows) == 0:
        return None, None

    outer_df = pd.DataFrame(outer_rows)
    macro = {k: float(outer_df[k].mean()) for k in ["auc", "pr_auc", "precision", "recall", "f1", "hit1", "hit3"]}
    return outer_df, macro


# ================================================================
# 4. Main
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("H3N2 Clade Prediction - Top-k Guided Tuning")
    print("=" * 60)

    print("\n[1] 데이터 로드")
    test_raw = load_nextclade(TEST_PATH)
    val_raw = load_nextclade(VAL_PATH)

    if test_raw.empty or val_raw.empty:
        raise SystemExit("Data loading failed.")

    test_cy = make_clade_year_table(test_raw)
    val_cy = make_clade_year_table(val_raw)
    train_df = build_supervised_dataset(test_cy)

    print(f"\nTraining rows: {len(train_df)}")
    print(f"Minority(y=1): {int(train_df['y'].sum())}, Majority(y=0): {int((train_df['y'] == 0).sum())}")

    print("\n" + "=" * 60)
    print("[2] 모델 설정")
    print("=" * 60)
    print("  Algorithm:    ElasticNet Logistic Regression")
    print("  Optimization: Top-k guided search + backtest constraints")
    print(f"  Features:     {len(FEATURES)}")
    print("  class_weight: None")
    print(f"  l1_ratio grid: {MODEL_GRID['l1_ratio']}")
    print(f"  C grid:        {MODEL_GRID['C']}")
    print(f"  min_recall grid: {MIN_RECALL_GRID}")

    print("\n" + "=" * 60)
    print("[3] 롤링 시계열 탐색 (운영 선택)")
    print("=" * 60)
    cv_df, best = search_best_params(
        train_df,
        test_cy=test_cy,
        val_cy=val_cy,
        min_train_years=MIN_TRAIN_YEARS,
    )
    print("\n[Best CV by Year]")
    print(cv_df.to_string(index=False))

    print("\nBest hyperparameters")
    print(
        f"  l1_ratio={best.l1_ratio}, C={best.c_value}, min_recall={best.min_recall}, "
        f"threshold={best.threshold:.3f}"
    )

    bt1_pct = (best.backtest_hit1 / best.backtest_n * 100) if best.backtest_n > 0 else 0.0
    bt3_pct = (best.backtest_hit3 / best.backtest_n * 100) if best.backtest_n > 0 else 0.0

    print("\n" + "=" * 60)
    print("[4] 최종 모델 학습 (전체 데이터)")
    print("=" * 60)
    final_model = build_model(best.l1_ratio, best.c_value)
    final_model.fit(train_df[FEATURES].fillna(0), train_df["y"])
    train_proba = final_model.predict_proba(train_df[FEATURES].fillna(0))[:, 1]
    flag_threshold = find_conservative_flag_threshold(
        train_df["y"].to_numpy(),
        train_proba,
        base_threshold=best.threshold,
        min_recall=best.min_recall,
        min_precision=max(0.60, best.precision),
    )

    coefs = final_model.named_steps["clf"].coef_[0]
    feat_coef = pd.DataFrame({"feature": FEATURES, "coef": coefs, "abs_coef": np.abs(coefs)}).sort_values(
        "abs_coef", ascending=False
    )
    print(feat_coef[["feature", "coef"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("[5] 성능 요약")
    print("=" * 60)
    print(f"  AUC:        {best.auc:.3f}")
    print(f"  PR-AUC:     {best.pr_auc:.3f}")
    print(f"  Precision:  {best.precision:.3f}")
    print(f"  Recall:     {best.recall:.3f}")
    print(f"  F1:         {best.f1:.3f}")
    print(f"  Top-1 Hit:  {best.top1:.2f}")
    print(f"  Top-3 Hit:  {best.top3:.2f}")
    print(f"  추천 임계값(탐색): {best.threshold:.3f}")
    print(f"  CR 플래그 임계값(보수): {flag_threshold:.3f}")
    print(
        f"  Backtest:   Top-1 {best.backtest_hit1}/{best.backtest_n} ({bt1_pct:.1f}%), "
        f"Top-3 {best.backtest_hit3}/{best.backtest_n} ({bt3_pct:.1f}%)"
    )

    print("\n" + "=" * 60)
    print("[6] 2026 CR 예측 - 상위 3개 후보")
    print("=" * 60)
    val_pred = val_cy.copy()
    val_pred["probability"] = final_model.predict_proba(val_pred[FEATURES].fillna(0))[:, 1]
    val_pred["predicted_CR"] = (val_pred["probability"] >= flag_threshold).astype(int)
    val_pred["dN/dS"] = (val_pred["nonsyn_med"] / val_pred["syn_med"]).replace([np.inf, -np.inf], np.nan).round(3)

    top3 = val_pred.sort_values("probability", ascending=False).head(3).reset_index(drop=True)
    for i, row in top3.iterrows():
        next_prob = top3.loc[i + 1, "probability"] if i + 1 < len(top3) else 0.0
        margin = float(row["probability"] - next_prob)
        if row["probability"] >= flag_threshold and margin >= 0.10:
            conf = "HIGH"
        elif row["probability"] >= best.threshold and margin >= 0.05:
            conf = "MEDIUM"
        else:
            conf = "LOW"
        print(f"  #{i+1}: {row['clade']}")
        print(f"      Probability: {row['probability']:.3f}")
        print(f"      Margin vs next: {margin:.3f}")
        print(f"      Confidence:  {conf}")
        print(f"      Frequency:   {row['freq']:.1%} ({row['n']:.0f} samples)")
        print(f"      dN/dS:       {row['dN/dS']}")
        print(f"      AA Reversion:{row['pam_reversion_med']}")
        print(f"      CR Flag:     {'YES' if int(row['predicted_CR']) == 1 else 'NO'}")
        print()
    print(f"\nPredicted as 2026 CR: {int(val_pred['predicted_CR'].sum())}/{len(val_pred)} clades")

    print("\n" + "=" * 60)
    print("[7] 최근 5개년 CR 추이")
    print("=" * 60)
    cr_history = get_dominant_clade_by_year(test_cy)
    for year in sorted(cr_history.index)[-5:]:
        clade = cr_history[year]
        year_data = test_raw[test_raw["year"] == year]
        total = len(year_data)
        clade_count = len(year_data[year_data["clade"] == clade])
        pct = clade_count / total * 100 if total > 0 else 0
        print(f"  {year}: {clade} ({clade_count}/{total}, {pct:.1f}%)")

    print("\n" + "=" * 60)
    print("[8] 백테스팅 (확장 윈도우)")
    print("=" * 60)
    print(f"  Top-1 Hit: {best.backtest_hit1}/{best.backtest_n} ({bt1_pct:.1f}%)")
    print(f"  Top-3 Hit: {best.backtest_hit3}/{best.backtest_n} ({bt3_pct:.1f}%)")

    print("\n" + "=" * 60)
    print("[9] 모델 요약")
    print("=" * 60)
    print(f"  Algorithm:     ElasticNet Logistic Regression")
    print(f"  Optimization:  Top-k guided search + backtest constraints")
    print(f"  Features:      {len(FEATURES)} features")
    print(f"  class_weight:  None")
    print(f"  Best hyperparameters:")
    print(f"    l1_ratio:    {best.l1_ratio}")
    print(f"    C:           {best.c_value}")
    print(f"    min_recall:  {best.min_recall}")
    print("  " + "-" * 30)
    print()
    print("  Performance 결과")
    print(f"  추천 임계값(탐색): {best.threshold:.3f}")
    print(f"  CR 플래그 임계값(보수): {flag_threshold:.3f}")
    print(f"  AUC:           {best.auc:.3f}")
    print(f"  PR-AUC:        {best.pr_auc:.3f}")
    print(f"  Precision:     {best.precision:.3f}")
    print(f"  Recall:        {best.recall:.3f}")
    print(f"  F1:            {best.f1:.3f}")
    print(f"  Top-1 Hit:     {best.top1:.2f}")
    print(f"  Top-3 Hit:     {best.top3:.2f}")
    print(f"  Backtest Top-1: {best.backtest_hit1}/{best.backtest_n} ({bt1_pct:.1f}%)")
    print(f"  Backtest Top-3: {best.backtest_hit3}/{best.backtest_n} ({bt3_pct:.1f}%)")

    if RUN_NESTED_DIAGNOSTIC:
        print("\n[Optional] Nested CV Diagnostic (conservative estimate)")
        nested_df, nested_macro = nested_cv_diagnostic(train_df)
        if nested_df is None:
            print("  Not enough data for nested diagnostic.")
        else:
            print(f"  AUC:        {nested_macro['auc']:.3f}")
            print(f"  PR-AUC:     {nested_macro['pr_auc']:.3f}")
            print(f"  Precision:  {nested_macro['precision']:.3f}")
            print(f"  Recall:     {nested_macro['recall']:.3f}")
            print(f"  F1:         {nested_macro['f1']:.3f}")
            print(f"  Top-1 Hit:  {nested_macro['hit1']:.2f}")
            print(f"  Top-3 Hit:  {nested_macro['hit3']:.2f}")

    print("\nDone.")
