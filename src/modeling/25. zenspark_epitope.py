"""
H3N2 Influenza Clade Prediction Model (Epitope + Glycosylation Features)
================================================================
목적: 다음 해 지배 clade(CR: Candidate Recommendation)를 예측
전략: class_weight=None + Threshold 최적화 + ElasticNet(l1=0.8, C=0.4)
데이터: 한국+중국+일본 Nextclade 데이터 (2005-2025)

변경점 (#2 → #5):
  - [추가] epitope_mut_med: HA1 Epitope(A,B,C,D,E) 위치 내 아미노산 치환 수 중앙값
    → 총 비동의 치환이 아닌, 면역 회피에 중요한 위치의 변이만 카운트
  - [추가] glyco_antigenic_med: 항원 부위 근처 당화(glycosylation) 부위 수 중앙값
    → 당 사슬이 항원 부위를 가려 면역 회피하는 메커니즘 반영
  - 기존 8개 피처 유지 + 2개 추가 = 총 10개 피처
  - "양(Quantity)에서 질(Quality)로" - 돌연변이의 위치적 중요성 반영

생물학적 근거:
  - Influenza H3N2의 면역 회피는 HA1 Epitope(A-E) 부위 돌연변이가 주도
  - 내부 단백질이나 비항원 부위의 변이는 면역 회피에 기여하지 않음
  - Glycosylation(당화)은 항원 부위를 물리적으로 차폐하여 항체 결합 방해
  - Epitope A, B가 면역 우성(immunodominant)으로 가장 중요

Epitope 위치 참조: Wiley et al. 1981, Wilson & Cox 1990
재현성: random_state=42 고정 (매 실행 동일 결과 보장)
"""

import pandas as pd
import numpy as np
import re
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

# 사용할 피처 목록 (기존 8개 + 2개 추가 = 10개)
FEATURES = [
    "n",                   # 해당 clade의 시퀀스 수
    "freq",                # 해당 연도 내 빈도
    "freq_prev",           # 전년도 빈도
    "freq_delta",          # 빈도 변화량 (올해 - 작년)
    "nonsyn_med",          # 비동의 치환 중앙값 (총 수량)
    "syn_med",             # 동의 치환 중앙값
    "novelty_med",         # 새로움 점수 중앙값
    "pam_reversion_med",   # 복귀 돌연변이 중앙값
    "epitope_mut_med",     # [NEW] Epitope(A-E) 위치 돌연변이 수 중앙값
    "glyco_antigenic_med", # [NEW] 항원 부위 근처 당화 부위 수 중앙값
]

# ================================================================
# 1-1. 생물학적 위치 정의 (Biological Position Definitions)
# ================================================================

# H3N2 HA1 Epitope 위치 (Wiley et al. 1981, Wilson & Cox 1990)
# 면역계가 인식하는 항원 결정기 부위 (총 131개 위치)
EPITOPE_POSITIONS = {
    # Site A: 면역우성 (immunodominant), 항체 결합의 주요 타겟
    "A": {122, 124, 126, 130, 131, 132, 133, 135, 137, 138,
          140, 142, 143, 144, 145, 146, 150, 152, 168},
    # Site B: 수용체 결합 부위(RBS) 근처, 면역우성
    "B": {128, 129, 155, 156, 157, 158, 159, 160, 163, 164,
          165, 186, 187, 188, 189, 190, 192, 193, 194, 196, 197, 198},
    # Site C: HA1 하부 및 C-terminal 영역
    "C": {44, 45, 46, 47, 48, 50, 51, 53, 54,
          273, 275, 276, 278, 279, 280, 294, 297, 299, 300,
          304, 305, 307, 308, 309, 310, 311, 312},
    # Site D: HA1 중간 영역, 광범위하게 분포
    "D": {96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174,
          175, 176, 177, 179, 182, 201, 203, 207, 208, 209,
          212, 213, 214, 215, 216, 217, 218, 219,
          226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248},
    # Site E: HA1 상부, 항체 접근 가능 영역
    "E": {57, 59, 62, 63, 67, 75, 78, 80, 81, 82, 83,
          86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265},
}
# 전체 Epitope 위치 합집합
ALL_EPITOPE_POS = set().union(*EPITOPE_POSITIONS.values())

# 항원 부위 근처 Glycosylation(당화) 위치
# → 이 위치에 N-linked glycosylation이 존재하면 항체 결합을 물리적으로 차단
GLYCO_ANTIGENIC_POSITIONS = {
    # position: 근접 epitope site
    38,    # C site 근처
    45,    # C site (Epitope C)
    63,    # E site (Epitope E) - 면역 회피 중요
    122,   # A site (Epitope A) - 매우 중요
    126,   # A site (Epitope A)
    133,   # A site (Epitope A) - 매우 중요
    144,   # A site (Epitope A) - B site도 차폐, 핵심 위치
    158,   # B site (Epitope B) - 면역 회피의 핵심
    165,   # B site (Epitope B) 근처
    246,   # D site (Epitope D)
}


# ================================================================
# 2. 돌연변이 파싱 함수 (Mutation Parsing Functions)
# ================================================================

def count_epitope_mutations(aa_subs_str):
    """aaSubstitutions 문자열에서 HA1 Epitope 위치 돌연변이 수를 카운트한다.

    Args:
        aa_subs_str: "HA1:G53D,HA1:G62E,HA1:E83K,..." 형식의 문자열

    Returns:
        int: Epitope A-E 위치 내 돌연변이 총 수
    """
    if pd.isna(aa_subs_str) or str(aa_subs_str).strip() == "":
        return 0

    count = 0
    for mut in str(aa_subs_str).split(","):
        mut = mut.strip()
        # HA1 유전자의 치환만 대상 (HA2, SigPep 등 제외)
        if not mut.startswith("HA1:"):
            continue
        # 위치 번호 추출: "HA1:G53D" → "G53D" → 53
        part = mut.split(":")[-1]
        pos_match = re.search(r'(\d+)', part)
        if pos_match:
            pos = int(pos_match.group(1))
            if pos in ALL_EPITOPE_POS:
                count += 1
    return count


def count_glyco_antigenic(glyco_str):
    """glycosylation 문자열에서 항원 부위 근처 당화 부위 수를 카운트한다.

    Args:
        glyco_str: "HA1:8:NST;HA1:22:NGT;HA1:158:NYT;..." 형식의 문자열

    Returns:
        int: 항원 부위 근처 glycosylation 부위 수
    """
    if pd.isna(glyco_str) or str(glyco_str).strip() == "":
        return 0

    count = 0
    for site in str(glyco_str).split(";"):
        site = site.strip()
        # HA1 유전자의 당화 부위만 대상
        if not site.startswith("HA1:"):
            continue
        # 위치 추출: "HA1:158:NYT" → 158
        parts = site.split(":")
        if len(parts) >= 2:
            try:
                pos = int(parts[1])
                if pos in GLYCO_ANTIGENIC_POSITIONS:
                    count += 1
            except ValueError:
                continue
    return count


# ================================================================
# 3. 데이터 로딩 (Data Loading)
# ================================================================

def load_nextclade(path: str) -> pd.DataFrame:
    """Nextclade CSV 파일을 로드하고 전처리하여 반환한다.

    처리 과정:
      1) QC 필터링 (good만 유지)
      2) seqName에서 연도 추출
      3) 돌연변이 관련 피처 생성
      4) [NEW] Epitope 돌연변이 & Glycosylation 파싱
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

    # --- 기존 돌연변이 피처 생성 ---
    df["nonsyn"]    = pd.to_numeric(df.get("totalAminoacidSubstitutions"), errors="coerce")
    df["total_subs"] = pd.to_numeric(df.get("totalSubstitutions"), errors="coerce")
    df["syn_proxy"]  = df["total_subs"] - df["nonsyn"]
    df["novelty"]    = df["nonsyn"].fillna(0) + 0.2 * df["syn_proxy"].fillna(0)
    df["pam_reversion"] = pd.to_numeric(
        df.get("privateAaMutations.totalReversionSubstitutions"), errors="coerce"
    )

    # --- [NEW] Epitope 돌연변이 카운트 ---
    aa_subs_col = "aaSubstitutions"
    if aa_subs_col in df.columns:
        df["epitope_mut"] = df[aa_subs_col].apply(count_epitope_mutations)
        print(f"    Epitope 돌연변이 파싱 완료 (평균: {df['epitope_mut'].mean():.1f}개/시퀀스)")
    else:
        print(f"    [경고] '{aa_subs_col}' 컬럼 없음 → epitope_mut = 0")
        df["epitope_mut"] = 0

    # --- [NEW] Glycosylation 항원부위 카운트 ---
    glyco_col = "glycosylation"
    if glyco_col in df.columns:
        df["glyco_antigenic"] = df[glyco_col].apply(count_glyco_antigenic)
        print(f"    Glycosylation 항원부위 파싱 완료 (평균: {df['glyco_antigenic'].mean():.1f}개/시퀀스)")
    else:
        print(f"    [경고] '{glyco_col}' 컬럼 없음 → glyco_antigenic = 0")
        df["glyco_antigenic"] = 0

    # --- 필요한 컬럼만 유지 ---
    keep = ["seqName", "clade", "year", "nonsyn", "syn_proxy", "novelty",
            "pam_reversion", "epitope_mut", "glyco_antigenic"]
    df = df[keep].dropna(subset=["clade", "year"]).copy()
    df["year"] = df["year"].astype(int)

    print(f"    최종 데이터: {len(df)}개 시퀀스")
    return df


# ================================================================
# 4. 피처 엔지니어링 (Feature Engineering)
# ================================================================

def make_clade_year_table(df: pd.DataFrame) -> pd.DataFrame:
    """시퀀스 데이터를 clade-연도 단위로 집계한다.

    생성 피처:
      - n: 시퀀스 수
      - freq: 연도 내 빈도 비율
      - freq_prev: 전년도 빈도
      - freq_delta: 빈도 변화량
      - nonsyn_med, syn_med, novelty_med, pam_reversion_med: 중앙값
      - [NEW] epitope_mut_med: Epitope 돌연변이 수 중앙값
      - [NEW] glyco_antigenic_med: 항원부위 당화 부위 수 중앙값
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
               epitope_mut_med=("epitope_mut", "median"),
               glyco_antigenic_med=("glyco_antigenic", "median"),
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
# 5. 라벨 생성 (Label Creation)
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
# 6. 모델 학습 및 평가 (Model Training & Evaluation)
# ================================================================

def build_model():
    """ElasticNet 로지스틱 회귀 파이프라인을 생성한다.
    StandardScaler로 정규화 후, 가중치 없이(None) 학습한다.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=MODEL_CONFIG["l1_ratio"],
            C=MODEL_CONFIG["C"],
            class_weight=None,
            max_iter=MODEL_CONFIG["max_iter"],
            random_state=42,
        )),
    ])


def find_best_threshold(y_true, y_proba, min_recall=0.7):
    """F1 점수를 최대화하는 분류 임계값을 탐색한다."""
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
    """시간순 교차검증(Expanding Window)을 수행한다."""
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

        model = build_model()
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

    metric_keys = ["threshold", "auc", "pr_auc", "precision", "recall", "f1", "hit1", "hit3"]
    macro = {k: np.mean(cv[k]) for k in metric_keys}

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
# 7. 메인 실행 (Main)
# ================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("H3N2 Clade Prediction - Epitope + Glycosylation Features (#5)")
    print("=" * 65)

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
    # [2] Epitope & Glycosylation 통계
    # ================================================================
    print("\n" + "=" * 65)
    print("[2] Epitope & Glycosylation Feature Statistics")
    print("=" * 65)

    for label, raw in [("TEST", test_raw), ("VAL", val_raw)]:
        print(f"\n  [{label}]")
        print(f"    epitope_mut  : mean={raw['epitope_mut'].mean():.2f}, "
              f"median={raw['epitope_mut'].median():.1f}, "
              f"min={raw['epitope_mut'].min()}, max={raw['epitope_mut'].max()}")
        print(f"    glyco_antigen: mean={raw['glyco_antigenic'].mean():.2f}, "
              f"median={raw['glyco_antigenic'].median():.1f}, "
              f"min={raw['glyco_antigenic'].min()}, max={raw['glyco_antigenic'].max()}")

    # 연도별 epitope mutation 추이
    print(f"\n  [연도별 Epitope 돌연변이 평균 (TEST)]")
    yearly_epi = test_raw.groupby("year")["epitope_mut"].mean()
    for y in sorted(yearly_epi.index)[-8:]:
        print(f"    {y}: {yearly_epi[y]:.1f}")

    # ================================================================
    # [3] 피처 엔지니어링
    # ================================================================
    print("\n" + "=" * 65)
    print("[3] Feature Engineering")
    print("=" * 65)

    test_cy  = make_clade_year_table(test_raw)
    val_cy   = make_clade_year_table(val_raw)
    train_df = build_supervised_dataset(test_cy)

    print(f"\nTraining data: {len(train_df)} clade-year rows")
    print(f"y=1 (minority): {train_df['y'].sum()}, y=0 (majority): {(train_df['y']==0).sum()}")

    # y=1 vs y=0 피처 비교 (새 피처 포커스)
    print(f"\n  [New Feature: y=1 vs y=0 비교]")
    for feat in ["epitope_mut_med", "glyco_antigenic_med"]:
        y1_mean = train_df[train_df["y"]==1][feat].mean()
        y0_mean = train_df[train_df["y"]==0][feat].mean()
        diff_pct = ((y1_mean - y0_mean) / y0_mean * 100) if y0_mean != 0 else 0
        print(f"    {feat}: y=1 평균={y1_mean:.2f}, y=0 평균={y0_mean:.2f} "
              f"(차이: {diff_pct:+.1f}%)")

    # ================================================================
    # [4] Model Configuration
    # ================================================================
    print("\n" + "=" * 65)
    print("[4] Model Configuration")
    print("=" * 65)
    print(f"  Algorithm:    ElasticNet Logistic Regression")
    print(f"  l1_ratio:     {MODEL_CONFIG['l1_ratio']}")
    print(f"  C:            {MODEL_CONFIG['C']}")
    print(f"  class_weight: None (가중치 없음, 동일 가중)")
    print(f"  threshold:    최적화 (min_recall >= 0.7)")
    print(f"  Features ({len(FEATURES)}): {FEATURES}")
    print(f"  Epitope positions: {len(ALL_EPITOPE_POS)}개 위치 "
          f"(A:{len(EPITOPE_POSITIONS['A'])}, B:{len(EPITOPE_POSITIONS['B'])}, "
          f"C:{len(EPITOPE_POSITIONS['C'])}, D:{len(EPITOPE_POSITIONS['D'])}, "
          f"E:{len(EPITOPE_POSITIONS['E'])})")
    print(f"  Glyco antigenic positions: {len(GLYCO_ANTIGENIC_POSITIONS)}개 위치")

    # ================================================================
    # [5] 시간순 교차검증 (Expanding Window)
    # ================================================================
    print("\n" + "=" * 65)
    print("[5] Time-Series Cross Validation (Expanding Window)")
    print("=" * 65)

    cv, macro, micro = cross_validate(train_df)

    cv_df = pd.DataFrame(cv)
    print("\n[CV Results by Year]")
    print(cv_df.to_string(index=False))

    # ================================================================
    # [6] 최종 모델 학습 (Final Model)
    # ================================================================
    print("\n" + "=" * 65)
    print("[6] Final Model: Train on ALL data")
    print("=" * 65)

    X_all = train_df[FEATURES].fillna(0)
    y_all = train_df["y"]

    final_model = build_model()
    final_model.fit(X_all, y_all)

    y_proba_all = final_model.predict_proba(X_all)[:, 1]
    opt_threshold, thr_metrics = find_best_threshold(y_all, y_proba_all, min_recall=0.7)
    MODEL_CONFIG["threshold"] = opt_threshold

    print(f"Model trained on {len(X_all)} samples with {len(FEATURES)} features")
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

    # 새 피처의 기여도 강조
    print(f"\n  [New Feature Coefficients]")
    for feat in ["epitope_mut_med", "glyco_antigenic_med"]:
        c = feat_coef[feat_coef["Feature"] == feat]["Coefficient"].values[0]
        rank = feat_coef["Feature"].tolist().index(feat) + 1
        print(f"    {feat}: {c:.4f} (중요도 순위: {rank}/{len(FEATURES)})")

    # ================================================================
    # [7] Performance Summary
    # ================================================================
    print("\n" + "=" * 65)
    print("[7] Performance Summary")
    print("=" * 65)

    print(f"\n[Average (fold별 평균)]")
    print(f"  AUC:          {macro['auc']:.3f}")
    print(f"  PR-AUC:       {macro['pr_auc']:.3f}")
    print(f"  Precision:    {macro['precision']:.3f}")
    print(f"  Recall:       {macro['recall']:.3f}")
    print(f"  F1:           {macro['f1']:.3f}")
    print(f"  Top-1 Hit:    {macro['hit1']:.2f}")
    print(f"  Top-3 Hit:    {macro['hit3']:.2f}")
    print(f"  Threshold:    {macro['threshold']:.3f}")

    # #2 대비 비교 (참고용 - #2 결과값 하드코딩)
    print(f"\n  [vs #2 Baseline (No Weight)]")
    baseline = {"auc": 0.708, "pr_auc": 0.556, "precision": 0.526,
                "recall": 0.909, "f1": 0.603, "hit1": 0.71, "hit3": 0.86}
    for k in ["auc", "pr_auc", "precision", "recall", "f1"]:
        diff = macro[k] - baseline[k]
        arrow = "+" if diff > 0 else ""
        print(f"    {k:>12}: #2={baseline[k]:.3f} → #5={macro[k]:.3f} ({arrow}{diff:.3f})")
    for k in ["hit1", "hit3"]:
        diff = macro[k] - baseline[k]
        arrow = "+" if diff > 0 else ""
        print(f"    {'Top-1 Hit' if k=='hit1' else 'Top-3 Hit':>12}: "
              f"#2={baseline[k]:.0%} → #5={macro[k]:.0%} ({arrow}{diff:.0%})")

    # ================================================================
    # [8] 2025 Prediction
    # ================================================================
    THRESHOLD = opt_threshold

    print("\n" + "=" * 65)
    print(f"[8] 2025 Prediction (2026 CR Candidates) - threshold={THRESHOLD:.3f}")
    print("=" * 65)

    if not val_cy.empty:
        val_pred = val_cy.copy()
        val_pred["probability"]  = final_model.predict_proba(val_pred[FEATURES].fillna(0))[:, 1]
        val_pred["predicted_CR"] = (val_pred["probability"] >= THRESHOLD).astype(int)

        val_pred["dN/dS"] = (val_pred["nonsyn_med"] / val_pred["syn_med"]).round(3)

        print("\n[2025 Clade Analysis - Top 10]")
        result = val_pred.sort_values("probability", ascending=False)[
            ["clade", "n", "freq", "nonsyn_med", "epitope_mut_med", "glyco_antigenic_med",
             "dN/dS", "pam_reversion_med", "probability", "predicted_CR"]
        ].head(10).reset_index(drop=True)
        result.index = [f"#{i+1}" for i in range(len(result))]
        result["predicted_CR"] = result["predicted_CR"].map({1: "<<< CR >>>", 0: ""})
        result.index.name = "rank"
        print(result.to_string())

        n_predicted  = val_pred["predicted_CR"].sum()
        total_clades = len(val_pred)
        print(f"\n[Prediction Summary]")
        print(f"  Total clades in 2025: {total_clades}")
        print(f"  Predicted as 2026 CR: {n_predicted} clades (threshold={THRESHOLD:.3f})")

        print(f"\n[2026 CR Prediction - Top 3 Candidates]")
        print("-" * 55)
        top3 = val_pred.sort_values("probability", ascending=False).head(3)
        for i, (_, row) in enumerate(top3.iterrows(), 1):
            status = "*** PREDICTED CR ***" if row["predicted_CR"] == 1 else ""
            print(f"  #{i}: {row['clade']}")
            print(f"      Probability:    {row['probability']:.3f}")
            print(f"      Frequency:      {row['freq']:.1%} ({row['n']:.0f} samples)")
            print(f"      Epitope Mut:    {row['epitope_mut_med']:.1f}")
            print(f"      Glyco Antigenic:{row['glyco_antigenic_med']:.1f}")
            print(f"      dN/dS:          {row['dN/dS']}")
            print(f"      AA Reversion:   {row['pam_reversion_med']}")
            print(f"      {status}")
            print()
    else:
        print("\n[WARNING] No validation data available.")

    # ================================================================
    # [9] Historical CR Trend (최근 5년)
    # ================================================================
    print("=" * 65)
    print("[9] Historical CR Trend (Recent 5 Years)")
    print("=" * 65)

    cr_history = get_dominant_clade_by_year(test_cy)
    for year in sorted(cr_history.index)[-5:]:
        clade = cr_history[year]
        year_data = test_raw[test_raw["year"] == year]
        total = len(year_data)
        clade_count = len(year_data[year_data["clade"] == clade])
        pct = clade_count / total * 100 if total > 0 else 0
        epi_mean = year_data[year_data["clade"] == clade]["epitope_mut"].mean()
        print(f"  {year}: {clade} ({clade_count}/{total}, {pct:.1f}%, epitope_avg={epi_mean:.1f})")

    # ================================================================
    # [10] Backtesting (2008 ~ 2024)
    # ================================================================
    print("\n" + "=" * 65)
    print("[10] Backtesting (Expanding Window)")
    print("=" * 65)

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

    bt_verified = bt_df[bt_df["hit1"].isin(["O", "X"])]
    n_total = len(bt_verified)
    n_hit1 = (bt_verified["hit1"] == "O").sum()
    n_hit3 = (bt_verified["hit3"] == "O").sum()

    if skipped:
        print(f"\n  * Skipped: {skipped} (학습 데이터 부족)")

    print(f"\n  {'Year':<6} {'Actual CR':<10} {'#1 Pred':<10} {'#2 Pred':<10} {'#3 Pred':<10} {'H@1':>4} {'H@3':>4}")
    print(f"  {'-'*58}")
    for _, r in bt_df.iterrows():
        yr = int(r["year"])
        if yr == 2026:
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

    # vs #2 baseline
    print(f"\n    [vs #2 Baseline]")
    print(f"      #2 Top-1 Hit: 71% → #5 Top-1 Hit: {n_hit1/n_total:.0%}")
    print(f"      #2 Top-3 Hit: 86% → #5 Top-3 Hit: {n_hit3/n_total:.0%}")

    # ================================================================
    # [11] MODEL SUMMARY
    # ================================================================
    print("\n" + "=" * 65)
    print("[11] MODEL SUMMARY")
    print("=" * 65)
    print(f"""
  Algorithm:     ElasticNet Logistic Regression
  Optimization:  No class_weight + Threshold 최적화
  Features:      {len(FEATURES)} features (기존 8 + epitope + glycosylation)
  class_weight:  None (동일 가중)

  New Features:
    - epitope_mut_med:     HA1 Epitope(A-E) 위치 돌연변이 수 중앙값
    - glyco_antigenic_med: 항원 부위 근처 당화 부위 수 중앙값

  Performance (Average):
    - AUC:          {macro['auc']:.3f}
    - PR-AUC:       {macro['pr_auc']:.3f}
    - Precision:    {macro['precision']:.3f}
    - Recall:       {macro['recall']:.3f}
    - F1:           {macro['f1']:.3f}
    - Top-1 Hit:    {macro['hit1']:.0%}
    - Top-3 Hit:    {macro['hit3']:.0%}

  Threshold:       {opt_threshold:.3f} (F1 최대화, min_recall >= 0.7)

  Backtest:
    - Top-1 Hit: {n_hit1}/{n_total} ({n_hit1/n_total:.0%})
    - Top-3 Hit: {n_hit3}/{n_total} ({n_hit3/n_total:.0%})
""")
    print("=" * 65)
    print("Done!")
    print("=" * 65)
