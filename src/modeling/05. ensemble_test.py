"""
Ensemble: ElasticNet + RF 조합 테스트
+ ElasticNet 단독 최적화 (cw=5, C=0.1)
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
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

def make_clade_year_table(df):
    g = df.groupby(["year","clade"]).agg(n=("seqName","count"),nonsyn_med=("nonsyn","median"),syn_med=("syn_proxy","median"),novelty_med=("novelty","median"),pam_reversion_med=("pam_reversion","median")).reset_index()
    t = g.groupby("year")["n"].sum().reset_index(name="year_total")
    g = g.merge(t, on="year")
    g["freq"] = g["n"]/g["year_total"]
    g = g.sort_values(["clade","year"])
    g["freq_prev"] = g.groupby("clade")["freq"].shift(1).fillna(0)
    g["freq_delta"] = g["freq"] - g["freq_prev"]
    return g

def build_ds(cy):
    idx = cy.groupby("year")["n"].idxmax()
    cr = cy.loc[idx,["year","clade"]].set_index("year")["clade"]
    cy = cy.copy()
    cy["CR_next"] = cy["year"].map(lambda y: cr.get(y+1))
    cy = cy.dropna(subset=["CR_next"]).copy()
    cy["y"] = (cy["clade"]==cy["CR_next"]).astype(int)
    return cy

def topk_hit(d, col, k=3):
    tc = d["CR_next"].iloc[0]
    return int(tc in d.sort_values(col, ascending=False).head(k)["clade"].tolist())

test_raw = load_nextclade(r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\TEST_KJC.csv")
cy = make_clade_year_table(test_raw)
train = build_ds(cy)
feats = ["n","freq","freq_prev","freq_delta","nonsyn_med","syn_med","novelty_med","pam_reversion_med"]

print("=" * 70)
print("ENSEMBLE & OPTIMIZED MODEL COMPARISON")
print("=" * 70)

# Models to test
configs = {
    "Current (cw15,C0.5,th0.2)": {
        "model": lambda: Pipeline([("s",StandardScaler()),("c",LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=0.9,C=0.5,class_weight={0:1,1:15},max_iter=2000))]),
        "th": 0.2
    },
    "ElasticNet (cw5,C0.1,th0.3)": {
        "model": lambda: Pipeline([("s",StandardScaler()),("c",LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=0.9,C=0.1,class_weight={0:1,1:5},max_iter=2000))]),
        "th": 0.3
    },
    "ElasticNet (cw10,C0.5,th0.5)": {
        "model": lambda: Pipeline([("s",StandardScaler()),("c",LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=0.9,C=0.5,class_weight={0:1,1:10},max_iter=2000))]),
        "th": 0.5
    },
    "RF(100,cw5,d5,th0.2)": {
        "model": lambda: RandomForestClassifier(n_estimators=100,class_weight={0:1,1:5},max_depth=5,random_state=42),
        "th": 0.2
    },
    "RF(100,cw10,d4,th0.3)": {
        "model": lambda: RandomForestClassifier(n_estimators=100,class_weight={0:1,1:10},max_depth=4,random_state=42),
        "th": 0.3
    },
}

# Also test ensemble
ensemble_configs = {
    "Ensemble(EN+RF) avg, th0.3": {"th": 0.3, "weight": [0.5, 0.5]},
    "Ensemble(EN+RF) avg, th0.4": {"th": 0.4, "weight": [0.5, 0.5]},
    "Ensemble(EN*0.3+RF*0.7) th0.3": {"th": 0.3, "weight": [0.3, 0.7]},
}

years = sorted(train["year"].unique())
all_results = []

# Single models
for name, cfg in configs.items():
    res = defaultdict(list)
    for i in range(3, len(years)):
        vy = years[i]
        tf = train[train["year"].isin(years[:i])].copy()
        vf = train[train["year"]==vy].copy()
        if len(vf)==0: continue
        Xt, yt = tf[feats].fillna(0), tf["y"]
        if yt.nunique()<2: continue
        Xv, yv = vf[feats].fillna(0), vf["y"]
        m = cfg["model"]()
        m.fit(Xt, yt)
        vf["p"] = m.predict_proba(Xv)[:,1]
        if yv.nunique()>1:
            yp = (vf["p"]>=cfg["th"]).astype(int)
            res["auc"].append(roc_auc_score(yv, vf["p"]))
            res["pr_auc"].append(average_precision_score(yv, vf["p"]))
            res["prec"].append(precision_score(yv,yp,zero_division=0))
            res["rec"].append(recall_score(yv,yp,zero_division=0))
            res["f1"].append(f1_score(yv,yp,zero_division=0))
        res["h1"].append(topk_hit(vf,"p",1))
        res["h3"].append(topk_hit(vf,"p",3))
    def sm(l): v=[x for x in l if not np.isnan(x)]; return np.mean(v) if v else np.nan
    all_results.append({"Name": name, "AUC": sm(res["auc"]), "PR-AUC": sm(res["pr_auc"]),
        "Prec": sm(res["prec"]), "Rec": sm(res["rec"]), "F1": sm(res["f1"]),
        "Hit1": np.mean(res["h1"]), "Hit3": np.mean(res["h3"])})

# Ensemble models
for ens_name, ens_cfg in ensemble_configs.items():
    res = defaultdict(list)
    for i in range(3, len(years)):
        vy = years[i]
        tf = train[train["year"].isin(years[:i])].copy()
        vf = train[train["year"]==vy].copy()
        if len(vf)==0: continue
        Xt, yt = tf[feats].fillna(0), tf["y"]
        if yt.nunique()<2: continue
        Xv, yv = vf[feats].fillna(0), vf["y"]

        # ElasticNet
        en = Pipeline([("s",StandardScaler()),("c",LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=0.9,C=0.1,class_weight={0:1,1:5},max_iter=2000))])
        en.fit(Xt, yt)
        p_en = en.predict_proba(Xv)[:,1]

        # RF
        rf = RandomForestClassifier(n_estimators=100,class_weight={0:1,1:5},max_depth=5,random_state=42)
        rf.fit(Xt, yt)
        p_rf = rf.predict_proba(Xv)[:,1]

        # Weighted average
        w = ens_cfg["weight"]
        vf["p"] = w[0]*p_en + w[1]*p_rf

        if yv.nunique()>1:
            yp = (vf["p"]>=ens_cfg["th"]).astype(int)
            res["auc"].append(roc_auc_score(yv, vf["p"]))
            res["pr_auc"].append(average_precision_score(yv, vf["p"]))
            res["prec"].append(precision_score(yv,yp,zero_division=0))
            res["rec"].append(recall_score(yv,yp,zero_division=0))
            res["f1"].append(f1_score(yv,yp,zero_division=0))
        res["h1"].append(topk_hit(vf,"p",1))
        res["h3"].append(topk_hit(vf,"p",3))
    def sm(l): v=[x for x in l if not np.isnan(x)]; return np.mean(v) if v else np.nan
    all_results.append({"Name": ens_name, "AUC": sm(res["auc"]), "PR-AUC": sm(res["pr_auc"]),
        "Prec": sm(res["prec"]), "Rec": sm(res["rec"]), "F1": sm(res["f1"]),
        "Hit1": np.mean(res["h1"]), "Hit3": np.mean(res["h3"])})

# Print results
rdf = pd.DataFrame(all_results)
rdf["composite"] = rdf["PR-AUC"].fillna(0)*0.3 + rdf["Hit3"].fillna(0)*0.3 + rdf["Rec"].fillna(0)*0.2 + rdf["Prec"].fillna(0)*0.2
rdf = rdf.sort_values("composite", ascending=False)

print(f"\n{'Name':<35} {'AUC':>6} {'PR-AUC':>7} {'Prec':>6} {'Rec':>6} {'F1':>6} {'H1':>5} {'H3':>5}")
print("-" * 85)
for _, row in rdf.iterrows():
    print(f"{row['Name']:<35} {row['AUC']:>6.3f} {row['PR-AUC']:>7.3f} {row['Prec']:>6.3f} {row['Rec']:>6.3f} {row['F1']:>6.3f} {row['Hit1']:>5.2f} {row['Hit3']:>5.2f}")
