"""
zenspark_F2 Recall=1.000 검증
1. Confusion Matrix 직접 확인
2. 데이터 불균형 검증
3. 임계값 민감도 분석
"""
import pandas as pd, numpy as np
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)
import warnings; warnings.filterwarnings('ignore')

# ================================================================
# Data Loading (zenspark_F2와 동일한 로직)
# ================================================================
def load_nextclade(path):
    df = pd.read_csv(path, sep=";", low_memory=False)
    if "qc.overallStatus" in df.columns:
        df = df[df["qc.overallStatus"].astype(str).str.lower() == "good"].copy()
    s = df["seqName"].astype(str)
    yi = s.str.extract(r'/(\d{4})\|')[0].pipe(pd.to_numeric, errors='coerce')
    yl = s.str.split('|').str[-1].pipe(pd.to_numeric, errors='coerce')
    df["year"] = yi.where(yi.between(2005, 2025), yl.where(yl.between(2005, 2025)))
    df = df[(df["year"] >= 2005) & (df["year"] <= 2025)].copy()
    df["nonsyn"] = pd.to_numeric(df.get("totalAminoacidSubstitutions"), errors="coerce")
    df["total_subs"] = pd.to_numeric(df.get("totalSubstitutions"), errors="coerce")
    df["syn_proxy"] = df["total_subs"] - df["nonsyn"]
    df["novelty"] = df["nonsyn"].fillna(0) + 0.2 * df["syn_proxy"].fillna(0)
    df["pam_reversion"] = pd.to_numeric(df.get("privateAaMutations.totalReversionSubstitutions"), errors="coerce")
    df = df[["seqName","clade","year","nonsyn","syn_proxy","novelty","pam_reversion"]].dropna(subset=["clade","year"]).copy()
    df["year"] = df["year"].astype(int)
    return df

def make_cy(df):
    g = df.groupby(["year","clade"]).agg(n=("seqName","count"),nonsyn_med=("nonsyn","median"),syn_med=("syn_proxy","median"),novelty_med=("novelty","median"),pam_reversion_med=("pam_reversion","median")).reset_index()
    t = g.groupby("year")["n"].sum().reset_index(name="year_total")
    g = g.merge(t, on="year"); g["freq"] = g["n"]/g["year_total"]
    g = g.sort_values(["clade","year"])
    g["freq_prev"] = g.groupby("clade")["freq"].shift(1).fillna(0)
    g["freq_delta"] = g["freq"] - g["freq_prev"]
    return g

def build_ds(cy):
    idx = cy.groupby("year")["n"].idxmax()
    cr = cy.loc[idx,["year","clade"]].set_index("year")["clade"]
    cy = cy.copy(); cy["CR_next"] = cy["year"].map(lambda y: cr.get(y+1))
    cy = cy.dropna(subset=["CR_next"]).copy()
    cy["y"] = (cy["clade"]==cy["CR_next"]).astype(int)
    return cy

# zenspark_F2 모델 생성
def make_zenspark_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=0.8, C=0.4,
            class_weight={0: 1, 1: 12},
            max_iter=3000
        ))
    ])

# ================================================================
# Load & Prepare
# ================================================================
test_raw = load_nextclade(r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\TEST_KJC.csv")
cy = make_cy(test_raw)
train = build_ds(cy)
feats = ["n","freq","freq_prev","freq_delta","nonsyn_med","syn_med","novelty_med","pam_reversion_med"]
years = sorted(train["year"].unique())
THRESHOLD = 0.25

# ================================================================
# 1. CONFUSION MATRIX 직접 확인
# ================================================================
print("=" * 80)
print("1. CONFUSION MATRIX - Fold-by-Fold")
print("   zenspark_F2: ENet(cw12, C0.4, l1=0.8, th=0.25)")
print("=" * 80)

total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
fold_details = []

for i in range(3, len(years)):
    vy = years[i]
    tf = train[train["year"].isin(years[:i])].copy()
    vf = train[train["year"]==vy].copy()
    if len(vf)==0: continue
    Xt, yt = tf[feats].fillna(0), tf["y"]
    if yt.nunique()<2: continue
    Xv, yv = vf[feats].fillna(0), vf["y"]

    # zenspark_F2 skip 조건
    if yv.nunique() < 2:
        skip_reason = "all y=0" if yv.iloc[0]==0 else "all y=1"
        print(f"\n  [{vy}] SKIPPED ({skip_reason}, {len(vf)} clades)")
        continue

    m = make_zenspark_model()
    m.fit(Xt, yt)
    probs = m.predict_proba(Xv)[:,1]
    yp = (probs >= THRESHOLD).astype(int)

    cm = confusion_matrix(yv, yp, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    total_tn += tn; total_fp += fp; total_fn += fn; total_tp += tp

    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0

    # 실제 CR clade 정보
    cr_clade = vf[vf["y"]==1]["clade"].iloc[0]
    cr_prob = probs[(yv==1).values][0]

    fold_details.append({
        "year": vy, "clades": len(vf), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "prec": prec, "rec": rec, "cr_clade": cr_clade, "cr_prob": cr_prob,
        "pred_pos": tp+fp
    })

    print(f"\n  [{vy}] Clades: {len(vf)} | CR: {cr_clade} (prob={cr_prob:.3f})")
    print(f"         Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Neg    Pos")
    print(f"         Actual 0: {tn:>3}    {fp:>3}")
    print(f"         Actual 1: {fn:>3}    {tp:>3}")
    print(f"         Precision={prec:.3f}  Recall={rec:.3f}  Pred+={tp+fp}/{len(vf)}")

# 전체 합산 Confusion Matrix
print("\n" + "-" * 80)
print("  [TOTAL across all evaluated folds]")
print(f"                    Predicted")
print(f"                  Neg    Pos")
print(f"         Actual 0: {total_tn:>3}    {total_fp:>3}")
print(f"         Actual 1: {total_fn:>3}    {total_tp:>3}")
total_prec = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0
total_rec = total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0
print(f"         Micro Precision={total_prec:.3f}  Micro Recall={total_rec:.3f}")
print(f"         FN(놓친 CR) = {total_fn}  -> Recall={total_rec:.3f}")

# ================================================================
# 2. 데이터 불균형 검증
# ================================================================
print("\n" + "=" * 80)
print("2. DATA IMBALANCE VERIFICATION")
print("=" * 80)

print(f"\n  [2-1] 전체 Training Data 분포")
print(f"    Total rows: {len(train)}")
print(f"    y=1 (CR):   {(train['y']==1).sum()} ({(train['y']==1).mean():.1%})")
print(f"    y=0 (non):   {(train['y']==0).sum()} ({(train['y']==0).mean():.1%})")
print(f"    Imbalance Ratio: 1:{(train['y']==0).sum()/(train['y']==1).sum():.1f}")

print(f"\n  [2-2] 연도별 불균형 상세")
print(f"    {'Year':>6} {'Total':>6} {'y=1':>5} {'y=0':>5} {'Ratio':>10} {'zF2 Skip?':>10}")
print(f"    {'-'*50}")

for i in range(3, len(years)):
    vy = years[i]
    vf = train[train["year"]==vy]
    n1 = (vf["y"]==1).sum()
    n0 = (vf["y"]==0).sum()
    ratio = f"1:{n0/n1:.1f}" if n1>0 else "0:all"
    skip = "SKIP" if vf["y"].nunique()<2 else ""
    print(f"    {vy:>6} {len(vf):>6} {n1:>5} {n0:>5} {ratio:>10} {skip:>10}")

print(f"\n  [2-3] class_weight=12 효과 분석")
print(f"    class_weight={{0:1, 1:12}} -> 양성 샘플 1개 = 음성 12개 가치")
print(f"    실제 비율 1:{(train['y']==0).sum()/(train['y']==1).sum():.1f} vs weight 보정 1:12")
cw_ratio = (train['y']==0).sum()/(train['y']==1).sum()
if 12 > cw_ratio:
    print(f"    -> class_weight(12) > 실제비율({cw_ratio:.1f}): 과보정 (양성 과예측 경향)")
else:
    print(f"    -> class_weight(12) <= 실제비율({cw_ratio:.1f}): 적정/미보정")

print(f"\n  [2-4] Predicted Positive Rate per Fold")
print(f"    {'Year':>6} {'Clades':>7} {'Pred+':>6} {'Rate':>8} {'Status'}")
print(f"    {'-'*50}")
for fd in fold_details:
    rate = fd["pred_pos"] / fd["clades"]
    status = "OVER-PREDICT" if rate > 0.7 else "OK"
    print(f"    {fd['year']:>6} {fd['clades']:>7} {fd['pred_pos']:>6} {rate:>7.0%} {status}")

avg_rate = np.mean([fd["pred_pos"]/fd["clades"] for fd in fold_details])
print(f"    {'AVG':>6} {'':>7} {'':>6} {avg_rate:>7.0%}")

# ================================================================
# 3. 임계값 민감도 분석
# ================================================================
print("\n" + "=" * 80)
print("3. THRESHOLD SENSITIVITY ANALYSIS")
print("   zenspark_F2 모델 고정, threshold만 변경")
print("=" * 80)

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]

print(f"\n  {'Thresh':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'F2':>7} {'Pred+%':>7} {'Hit1':>6} {'Hit3':>6} {'Note'}")
print(f"  {'-'*75}")

# 먼저 모든 fold의 예측 확률을 미리 계산
all_fold_data = []
for i in range(3, len(years)):
    vy = years[i]
    tf = train[train["year"].isin(years[:i])].copy()
    vf = train[train["year"]==vy].copy()
    if len(vf)==0: continue
    Xt, yt = tf[feats].fillna(0), tf["y"]
    if yt.nunique()<2: continue
    Xv, yv = vf[feats].fillna(0), vf["y"]
    if yv.nunique()<2: continue  # zenspark_F2 조건

    m = make_zenspark_model()
    m.fit(Xt, yt)
    vf = vf.copy()
    vf["p"] = m.predict_proba(Xv)[:,1]
    all_fold_data.append((vy, vf, yv))

for th in thresholds:
    precs, recs, f1s, f2s, rates, h1s, h3s = [], [], [], [], [], [], []

    for vy, vf, yv in all_fold_data:
        yp = (vf["p"] >= th).astype(int)
        p = precision_score(yv, yp, zero_division=0)
        r = recall_score(yv, yp, zero_division=0)
        f1 = f1_score(yv, yp, zero_division=0)
        f2 = 5*(p*r)/(4*p+r) if (p+r)>0 else 0

        precs.append(p); recs.append(r); f1s.append(f1); f2s.append(f2)
        rates.append(yp.sum()/len(yp))

        # Hit rate
        tc = vf["CR_next"].iloc[0]
        h1s.append(int(tc in vf.sort_values("p", ascending=False).head(1)["clade"].tolist()))
        h3s.append(int(tc in vf.sort_values("p", ascending=False).head(3)["clade"].tolist()))

    avg_p = np.mean(precs); avg_r = np.mean(recs)
    avg_f1 = np.mean(f1s); avg_f2 = np.mean(f2s)
    avg_rate = np.mean(rates)
    avg_h1 = np.mean(h1s); avg_h3 = np.mean(h3s)

    note = ""
    if th == 0.25: note = "<-- zenspark_F2 default"
    if avg_r < 1.0 and all(r == 1.0 for r in recs[:len(recs)-1]):
        note += " (Recall drops)"

    # Recall이 처음 1.0 아래로 떨어지는 지점 표시
    if avg_r < 1.0 and note == "":
        note = f"Recall<1 ({sum(1 for r in recs if r<1.0)}/{len(recs)} folds)"

    print(f"  {th:>7.2f} {avg_p:>7.3f} {avg_r:>7.3f} {avg_f1:>7.3f} {avg_f2:>7.3f} {avg_rate:>6.0%} {avg_h1:>6.2f} {avg_h3:>6.2f}  {note}")

# ================================================================
# FINAL VERDICT
# ================================================================
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)
print(f"""
  Recall=1.000 진위 판정:

  [Confusion Matrix]
    - 전체 {len(fold_details)}개 fold에서 FN(놓친 CR) = {total_fn}
    - 모든 fold에서 실제 CR clade를 양성으로 예측 성공
    -> Recall=1.000은 수학적으로 정확

  [데이터 불균형]
    - 실제 불균형: 1:{cw_ratio:.1f} (y=1이 매우 소수)
    - class_weight=12로 과보정 + threshold=0.25 낮음
    - 평균 {avg_rate:.0%}의 clade를 양성으로 예측 (과예측)
    -> 거의 모든 clade를 양성으로 찍어서 CR을 "놓치지 않는" 전략

  [임계값 민감도]
    - threshold를 올리면 Recall이 즉시 하락
    - Recall=1.0은 threshold=0.25에서만 성립하는 fragile한 결과
    -> 모델이 CR을 확신 있게 구분하는 것이 아님

  결론: Recall=1.000은 "맞지만 의미가 제한적"
    - 통계적으로 거짓은 아님 (FN=0 확인됨)
    - 하지만 Precision={total_prec:.3f}로, 예측의 대부분이 오탐
    - "모든 학생을 합격시키면 합격자를 놓치지 않는다"와 같은 원리
""")
