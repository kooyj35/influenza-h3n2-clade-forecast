import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_SCRIPT = "##33. zenspark_noweight_bootstrap_feature7_082_repro.py"
OUT_DIR = Path("model_comparison_output/m33_explainability")
RANDOM_STATE = 42

# Prefer Garamond across all plots (falls back if unavailable on system)
plt.rcParams["font.family"] = ["Garamond", "Times New Roman", "DejaVu Serif"]


def load_m33_namespace():
    script = Path(TARGET_SCRIPT)
    if not script.exists():
        raise SystemExit(f"Missing script: {TARGET_SCRIPT}")
    return runpy.run_path(str(script))


def fit_final_model(ns):
    load_nextclade = ns["load_nextclade"]
    make_clade_year_table = ns["make_clade_year_table"]
    build_supervised_dataset = ns["build_supervised_dataset"]
    search_best_params = ns["search_best_params"]
    build_model = ns["build_model"]
    make_sample_weight = ns["make_sample_weight"]

    TEST_PATH = ns["TEST_PATH"]
    VAL_PATH = ns["VAL_PATH"]
    FEATURES = ns["FEATURES"]
    MIN_TRAIN_YEARS = ns["MIN_TRAIN_YEARS"]

    test_raw = load_nextclade(TEST_PATH)
    val_raw = load_nextclade(VAL_PATH)
    if test_raw.empty or val_raw.empty:
        raise SystemExit("Data loading failed.")

    test_cy = make_clade_year_table(test_raw)
    val_cy = make_clade_year_table(val_raw)
    train_df = build_supervised_dataset(test_cy)

    _, best = search_best_params(train_df, test_cy=test_cy, val_cy=val_cy, min_train_years=MIN_TRAIN_YEARS)

    model = build_model(best.l1_ratio, best.c_value)
    X = train_df[FEATURES].fillna(0)
    y = train_df["y"]
    if best.use_sample_weight:
        model.fit(X, y, clf__sample_weight=make_sample_weight(train_df["n"]))
    else:
        model.fit(X, y)

    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    Xs = scaler.transform(X)
    coef = clf.coef_[0].astype(float)

    # SHAP-like linear contributions around expected value
    # logit(x) = intercept + sum(z_j * coef_j)
    expected_value = float(np.mean(Xs @ coef + clf.intercept_[0]))
    centered = Xs - np.mean(Xs, axis=0, keepdims=True)
    contrib = centered * coef[None, :]

    return {
        "features": FEATURES,
        "coef": coef,
        "X": X,
        "Xs": Xs,
        "contrib": contrib,
        "expected_value": expected_value,
    }


def styled_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")


def plot_final_coef(payload):
    features = payload["features"]
    coef = payload["coef"]

    order = np.argsort(np.abs(coef))[::-1]
    f = [features[i] for i in order]
    c = coef[order]
    colors = ["#e02020" if v >= 0 else "#1f77b4" for v in c]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    y = np.arange(len(f))
    ax.barh(y, c, color=colors, height=0.8)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(f, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_title("Final Model Coefficients", fontsize=18, fontweight="bold", pad=10)
    ax.set_xlabel("Coefficient (log-odds scale)", fontsize=14, fontweight="bold", labelpad=14)
    ax.tick_params(axis="x", labelsize=13)
    for t in ax.get_xticklabels():
        t.set_fontweight("bold")
    styled_axes(ax)
    fig.tight_layout()
    out = OUT_DIR / "m33_final_model_coefficients.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_shap_beeswarm_like(payload):
    rng = np.random.default_rng(RANDOM_STATE)
    features = payload["features"]
    X = payload["X"]
    contrib = payload["contrib"]

    mean_abs = np.mean(np.abs(contrib), axis=0)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    all_x = []
    all_y = []
    all_c = []
    y_labels = []
    n_feat = len(order)

    for rank, j in enumerate(order):
        xvals = contrib[:, j]
        vals = X.iloc[:, j].to_numpy(dtype=float)

        # robust normalize feature value for color
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
        denom = (hi - lo) if hi > lo else 1.0
        cvals = np.clip((vals - lo) / denom, 0, 1)

        y0 = n_feat - 1 - rank
        jitter = rng.normal(0.0, 0.08, size=len(xvals))
        yvals = y0 + jitter

        all_x.append(xvals)
        all_y.append(yvals)
        all_c.append(cvals)
        y_labels.append(features[j])

    Xp = np.concatenate(all_x)
    Yp = np.concatenate(all_y)
    Cp = np.concatenate(all_c)

    sc = ax.scatter(Xp, Yp, c=Cp, cmap="cool", s=18, alpha=0.95, edgecolors="none")
    ax.axvline(0, color="gray", linewidth=1.2)
    ax.set_yticks(np.arange(n_feat))
    ax.set_yticklabels(list(reversed(y_labels)), fontsize=12, fontweight="bold")
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=17, fontweight="bold", labelpad=16)
    ax.set_title("M33 SHAP-like Summary", fontsize=18, fontweight="bold", pad=22)
    ax.tick_params(axis="x", labelsize=12)
    for t in ax.get_xticklabels():
        t.set_fontweight("bold")
    ax.grid(axis="y", linestyle=(0, (1, 4)), alpha=0.35)
    styled_axes(ax)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Feature value", fontsize=17, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "m33_shap_like_beeswarm.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_mean_abs_contrib_green(payload):
    features = payload["features"]
    contrib = payload["contrib"]
    imp = np.mean(np.abs(contrib), axis=0)

    order = np.argsort(imp)[::-1]
    f = [features[i] for i in order]
    v = imp[order]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    y = np.arange(len(f))
    ax.barh(y, v, color="#2ca02c", height=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(f, fontsize=12)
    ax.invert_yaxis()
    ax.set_title("Feature Importance by Mean |Contribution| (SHAP-like)", fontsize=18, pad=10)
    ax.set_xlabel("Mean absolute contribution (log-odds)", fontsize=14)
    styled_axes(ax)
    fig.tight_layout()
    out = OUT_DIR / "m33_feature_importance_mean_abs_contribution.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_mean_abs_shap_blue(payload):
    features = payload["features"]
    contrib = payload["contrib"]
    imp = np.mean(np.abs(contrib), axis=0)

    order = np.argsort(imp)[::-1]
    f = [features[i] for i in order]
    v = imp[order]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    y = np.arange(len(f))
    ax.barh(y, v, color="#1e88e5", height=0.8)
    ax.axvline(0, color="#999999", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(f, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=18)
    styled_axes(ax)
    fig.tight_layout()
    out = OUT_DIR / "m33_mean_abs_shap_bar.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ns = load_m33_namespace()
    payload = fit_final_model(ns)

    saved = [
        plot_final_coef(payload),
        plot_shap_beeswarm_like(payload),
        plot_mean_abs_contrib_green(payload),
        plot_mean_abs_shap_blue(payload),
    ]

    print("[Saved files]")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
