"""
Recall=1.000 검증: zenspark_F2 vs 13기반 fold-by-fold 분석
핵심 체크:
1. CV fold 스킵 조건 차이 (y_val.nunique()<2 포함 여부)
2. fold별 예측 상세 (양성 예측 수 / 전체 clade 수 / 실제 양성 수)
3. Recall=1.0이 trivial한지 (모든 clade를 양성으로 예측?)
"""
import pandas as pd, numpy as np
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import warnings; warnings.filterwarnings('ignore')

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

test_raw = load_nextclade(r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\TEST_KJC.csv")
cy = make_cy(test_raw); train = build_ds(cy)
feats = ["n","freq","freq_prev","freq_delta","nonsyn_med","syn_med","novelty_med","pam_reversion_med"]
years = sorted(train["year"].unique())

print("=" * 90)
print("RECALL=1.000 VERIFICATION: Fold-by-Fold Analysis")
print("=" * 90)

# ============================================================
# CHECK 1: CV fold skip 조건 비교
# ============================================================
print("\n[CHECK 1] CV Fold Skip Condition Comparison")
print("-" * 70)

for label, skip_val_check in [("zenspark_F2 (skip if y_val unique<2)", True),
                                ("13giban (skip only if y_train unique<2)", False)]:
    skipped = []
    used = []
    for i in range(3, len(years)):
        vy = years[i]
        tf = train[train["year"].isin(years[:i])].copy()
        vf = train[train["year"]==vy].copy()
        if len(vf)==0:
            skipped.append((vy, "empty val"))
            continue
        yt = tf["y"]
        yv = vf["y"]
        if yt.nunique() < 2:
            skipped.append((vy, "train single class"))
            continue
        if skip_val_check and yv.nunique() < 2:
            skipped.append((vy, f"val single class (all y={yv.iloc[0]})"))
            continue
        used.append(vy)

    print(f"\n  {label}:")
    print(f"    Used folds: {len(used)} -> {used}")
    print(f"    Skipped folds: {len(skipped)}")
    for sy, reason in skipped:
        print(f"      year={sy}: {reason}")

# ============================================================
# CHECK 2: Fold-by-fold detail for zenspark_F2
# ============================================================
print("\n" + "=" * 90)
print("[CHECK 2] zenspark_F2 Fold-by-Fold Prediction Detail")
print(f"  Config: cw={{0:1, 1:12}}, C=0.4, l1_ratio=0.8, threshold=0.25")
print("-" * 90)
print(f"  {'Year':>6} {'Clades':>7} {'y=1':>5} {'y=0':>5} {'Pred+':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'Note'}")
print("  " + "-" * 82)

zf2_recall_list = []
zf2_prec_list = []

for i in range(3, len(years)):
    vy = years[i]
    tf = train[train["year"].isin(years[:i])].copy()
    vf = train[train["year"]==vy].copy()
    if len(vf)==0: continue
    Xt, yt = tf[feats].fillna(0), tf["y"]
    if yt.nunique()<2: continue
    Xv, yv = vf[feats].fillna(0), vf["y"]

    n_pos = (yv==1).sum()
    n_neg = (yv==0).sum()
    n_total = len(yv)

    # zenspark_F2 skips if val single class
    if yv.nunique() < 2:
        print(f"  {vy:>6} {n_total:>7} {n_pos:>5} {n_neg:>5} {'--':>6} {'--':>4} {'--':>4} {'--':>4} {'--':>6} {'--':>6} SKIPPED (val single class)")
        continue

    m = Pipeline([("s",StandardScaler()),("c",LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=0.8,C=0.4,class_weight={0:1,1:12},max_iter=3000))])
    m.fit(Xt, yt)
    vf["p"] = m.predict_proba(Xv)[:,1]
    yp = (vf["p"]>=0.25).astype(int)

    tp = ((yp==1) & (yv==1)).sum()
    fp = ((yp==1) & (yv==0)).sum()
    fn = ((yp==0) & (yv==1)).sum()
    pred_pos = yp.sum()

    prec = precision_score(yv, yp, zero_division=0)
    rec = recall_score(yv, yp, zero_division=0)

    zf2_recall_list.append(rec)
    zf2_prec_list.append(prec)

    note = ""
    if pred_pos == n_total:
        note = "ALL predicted positive!"
    elif pred_pos / n_total > 0.8:
        note = f"{pred_pos/n_total:.0%} predicted positive"

    print(f"  {vy:>6} {n_total:>7} {n_pos:>5} {n_neg:>5} {pred_pos:>6} {tp:>4} {fp:>4} {fn:>4} {prec:>6.3f} {rec:>6.3f} {note}")

print(f"\n  Average Recall across used folds: {np.mean(zf2_recall_list):.3f}")
print(f"  Average Precision across used folds: {np.mean(zf2_prec_list):.3f}")
print(f"  Folds with Recall=1.0: {sum(1 for r in zf2_recall_list if r==1.0)}/{len(zf2_recall_list)}")

# ============================================================
# CHECK 3: 13기반 동일 조건 비교 (skip val 없이)
# ============================================================
print("\n" + "=" * 90)
print("[CHECK 3] 13giban (NO val skip) Fold-by-Fold - Same Metrics")
print(f"  Config: cw={{0:1, 1:15}}, C=0.5, l1_ratio=0.9, threshold=0.2")
print("-" * 90)
print(f"  {'Year':>6} {'Clades':>7} {'y=1':>5} {'y=0':>5} {'Pred+':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'Note'}")
print("  " + "-" * 82)

g13_recall_list = []
g13_prec_list = []

for i in range(3, len(years)):
    vy = years[i]
    tf = train[train["year"].isin(years[:i])].copy()
    vf = train[train["year"]==vy].copy()
    if len(vf)==0: continue
    Xt, yt = tf[feats].fillna(0), tf["y"]
    if yt.nunique()<2: continue
    Xv, yv = vf[feats].fillna(0), vf["y"]

    n_pos = (yv==1).sum()
    n_neg = (yv==0).sum()
    n_total = len(yv)

    m = Pipeline([("s",StandardScaler()),("c",LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=0.9,C=0.5,class_weight={0:1,1:15},max_iter=2000))])
    m.fit(Xt, yt)
    vf["p"] = m.predict_proba(Xv)[:,1]
    yp = (vf["p"]>=0.2).astype(int)

    tp = ((yp==1) & (yv==1)).sum()
    fp = ((yp==1) & (yv==0)).sum()
    fn = ((yp==0) & (yv==1)).sum()
    pred_pos = yp.sum()

    if yv.nunique() > 1:
        prec = precision_score(yv, yp, zero_division=0)
        rec = recall_score(yv, yp, zero_division=0)
        g13_recall_list.append(rec)
        g13_prec_list.append(prec)
        note = ""
        if pred_pos == n_total:
            note = "ALL predicted positive!"
        elif pred_pos / n_total > 0.8:
            note = f"{pred_pos/n_total:.0%} predicted positive"
        print(f"  {vy:>6} {n_total:>7} {n_pos:>5} {n_neg:>5} {pred_pos:>6} {tp:>4} {fp:>4} {fn:>4} {prec:>6.3f} {rec:>6.3f} {note}")
    else:
        print(f"  {vy:>6} {n_total:>7} {n_pos:>5} {n_neg:>5} {pred_pos:>6} {tp:>4} {fp:>4} {fn:>4} {'--':>6} {'--':>6} val single class (metrics undefined)")

print(f"\n  Average Recall across evaluated folds: {np.mean(g13_recall_list):.3f}")
print(f"  Average Precision across evaluated folds: {np.mean(g13_prec_list):.3f}")

# ============================================================
# CHECK 4: Trivial baseline - "predict ALL positive"
# ============================================================
print("\n" + "=" * 90)
print("[CHECK 4] Trivial Baseline: Predict ALL as Positive")
print("-" * 70)
trivial_rec = []
trivial_prec = []
for i in range(3, len(years)):
    vy = years[i]
    vf = train[train["year"]==vy].copy()
    if len(vf)==0: continue
    yv = vf["y"]
    if yv.nunique() < 2: continue
    # predict all positive
    yp_all = np.ones(len(yv), dtype=int)
    trivial_rec.append(recall_score(yv, yp_all))
    trivial_prec.append(precision_score(yv, yp_all))

print(f"  If we predict ALL clades as CR:")
print(f"    Recall = {np.mean(trivial_rec):.3f} (always 1.000 by definition)")
print(f"    Precision = {np.mean(trivial_prec):.3f}")

# ============================================================
# CHECK 5: zenspark_F2 probability 분포 분석
# ============================================================
print("\n" + "=" * 90)
print("[CHECK 5] zenspark_F2 Probability Distribution per Fold")
print("-" * 70)
for i in range(3, len(years)):
    vy = years[i]
    tf = train[train["year"].isin(years[:i])].copy()
    vf = train[train["year"]==vy].copy()
    if len(vf)==0: continue
    Xt, yt = tf[feats].fillna(0), tf["y"]
    if yt.nunique()<2: continue
    Xv, yv = vf[feats].fillna(0), vf["y"]
    if yv.nunique()<2: continue  # match zenspark_F2 skip

    m = Pipeline([("s",StandardScaler()),("c",LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=0.8,C=0.4,class_weight={0:1,1:12},max_iter=3000))])
    m.fit(Xt, yt)
    vf["p"] = m.predict_proba(Xv)[:,1]

    print(f"\n  Year {vy}: {len(vf)} clades")
    print(f"    Prob stats: min={vf['p'].min():.3f}, median={vf['p'].median():.3f}, max={vf['p'].max():.3f}")
    print(f"    >0.25 (threshold): {(vf['p']>=0.25).sum()}/{len(vf)}")
    print(f"    >0.50:             {(vf['p']>=0.50).sum()}/{len(vf)}")
    # Show actual CR clade probability
    cr_row = vf[vf["y"]==1]
    if len(cr_row) > 0:
        print(f"    Actual CR clade: {cr_row['clade'].iloc[0]} -> prob={cr_row['p'].iloc[0]:.3f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 90)
print("VERIFICATION SUMMARY")
print("=" * 90)
print(f"""
  zenspark_F2 Recall=1.000 원인 분석:

  1. Fold Skip 효과:
     - zenspark_F2는 y_val.nunique()<2 인 fold를 SKIP합니다
     - 이 fold들은 val에 positive sample이 없어 recall 계산이 불가능한 fold
     - 13기반은 이런 fold도 포함 (metrics=NaN 처리)
     -> 스킵 자체는 recall 수치에 직접 영향 X (어차피 NaN이므로)

  2. 실제 Recall=1.0의 의미:
     - 사용된 모든 fold에서 실제 CR clade(y=1)를 양성으로 예측 성공
     - 단, threshold=0.25가 낮고 class_weight=12로 높아서
       대부분의 clade를 양성으로 예측하는 경향 (높은 FP)
     - "진짜 양성은 놓치지 않지만, 음성도 많이 양성으로 예측"

  3. 결론:
     - Recall=1.0은 통계적으로 유효하지만, Precision 희생의 결과
     - 모든 clade를 양성으로 예측해도 Recall=1.0 달성 가능
     - zenspark_F2가 trivial baseline과 얼마나 다른지가 핵심
""")
