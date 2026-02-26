import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_SCRIPT = "##33. zenspark_noweight_bootstrap_feature7_082_repro.py"
OUT_DIR = Path("model_comparison_output")

# Garamond first
plt.rcParams["font.family"] = ["Garamond", "Times New Roman", "DejaVu Serif"]


def load_ns():
    p = Path(TARGET_SCRIPT)
    if not p.exists():
        raise SystemExit(f"Missing script: {TARGET_SCRIPT}")
    return runpy.run_path(str(p))


def build_cv_predictions(ns):
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

    years = sorted(train_df["year"].unique())
    rows = []

    for i in range(MIN_TRAIN_YEARS, len(years)):
        val_year = years[i]
        train_years = years[:i]

        train_fold = train_df[train_df["year"].isin(train_years)]
        val_fold = train_df[train_df["year"] == val_year].copy()

        if train_fold.empty or val_fold.empty:
            continue
        if train_fold["y"].nunique() < 2 or val_fold["y"].nunique() < 2:
            continue

        model = build_model(best.l1_ratio, best.c_value)
        Xtr = train_fold[FEATURES].fillna(0)
        ytr = train_fold["y"]
        Xva = val_fold[FEATURES].fillna(0)

        if best.use_sample_weight:
            model.fit(Xtr, ytr, clf__sample_weight=make_sample_weight(train_fold["n"]))
        else:
            model.fit(Xtr, ytr)

        val_fold["proba"] = model.predict_proba(Xva)[:, 1]
        val_fold["n_clades_year"] = len(val_fold)
        rows.append(val_fold[["year", "clade", "CR_next", "y", "proba", "n_clades_year"]])

    if not rows:
        raise SystemExit("No valid CV folds for prediction collection.")

    pred_df = pd.concat(rows, ignore_index=True)
    return pred_df


def plot_decision_curve(pred_df: pd.DataFrame, out_png: Path):
    y = pred_df["y"].to_numpy(dtype=int)
    p = pred_df["proba"].to_numpy(dtype=float)
    n = len(y)

    thresholds = np.arange(0.05, 0.91, 0.01)
    prevalence = y.mean()

    m33_nb = []
    all_nb = []
    none_nb = []

    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))

        w = t / (1 - t)
        nb_model = (tp / n) - (fp / n) * w
        nb_all = prevalence - (1 - prevalence) * w

        m33_nb.append(nb_model)
        all_nb.append(nb_all)
        none_nb.append(0.0)

    m33_nb_arr = np.array(m33_nb, dtype=float)
    all_nb_arr = np.array(all_nb, dtype=float)
    useful_mask = (m33_nb_arr > all_nb_arr) & (m33_nb_arr > 0)
    useful_thresholds = thresholds[useful_mask]
    best_idx = int(np.argmax(m33_nb_arr))
    best_thr = float(thresholds[best_idx])
    best_nb = float(m33_nb_arr[best_idx])
    thr_ref = 0.223
    ref_idx = int(np.argmin(np.abs(thresholds - thr_ref)))
    ref_nb = float(m33_nb_arr[ref_idx])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, m33_nb, color="#1f77b4", linewidth=2.5, label="M33")
    ax.plot(thresholds, all_nb, color="#d62728", linestyle="--", linewidth=1.8, label="Treat All")
    ax.plot(thresholds, none_nb, color="#555555", linestyle=":", linewidth=1.8, label="Treat None")

    ax.set_title("Decision Curve Analysis (Expanding CV OOF)", fontsize=16, fontweight="bold", pad=10)
    ax.set_xlabel("Threshold Probability", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Net Benefit", fontsize=13, fontweight="bold", labelpad=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=11)
    ax.tick_params(axis="both", labelsize=11)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

    if len(useful_thresholds) > 0:
        useful_txt = f"{useful_thresholds.min():.2f} ~ {useful_thresholds.max():.2f}"
    else:
        useful_txt = "N/A"
    note = (
        f"Useful threshold range: {useful_txt}\n"
        f"Best NB: {best_nb:.3f} @ t={best_thr:.2f}\n"
        f"NB @ t=0.223: {ref_nb:.3f}"
    )
    ax.text(
        0.98,
        0.03,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="black",
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_topk_tradeoff(pred_df: pd.DataFrame, out_png: Path):
    yearly = []
    for year, g in pred_df.groupby("year"):
        g2 = g.sort_values("proba", ascending=False).reset_index(drop=True)
        true_clade = g2["CR_next"].iloc[0]
        ordered = g2["clade"].tolist()
        ncl = len(ordered)
        yearly.append((year, ordered, true_clade, ncl))

    max_k = max(1, min(8, min(v[3] for v in yearly)))
    ks = np.arange(1, max_k + 1)
    coverages = []
    alert_vols = []

    for k in ks:
        hit_list = []
        vol_list = []
        for _, ordered, true_clade, ncl in yearly:
            hit = 1 if true_clade in ordered[:k] else 0
            hit_list.append(hit)
            vol_list.append(k / ncl)
        coverages.append(np.mean(hit_list))
        alert_vols.append(np.mean(vol_list))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alert_vols, coverages, marker="o", color="#2ca02c", linewidth=2.5, markersize=7, label="M33")

    # random baseline: expected hit ~= alert volume
    xline = np.linspace(0, max(alert_vols) * 1.05, 100)
    ax.plot(xline, xline, linestyle="--", color="#999999", linewidth=1.8, label="Random baseline")

    for x, y, k in zip(alert_vols, coverages, ks):
        ax.text(x + 0.005, y + 0.01, f"k={k}", fontsize=10, fontweight="bold", color="#1d6f1d")

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_title("Top-k Coverage vs Alert Volume (Expanding CV)", fontsize=16, fontweight="bold", pad=10)
    ax.set_xlabel("Average Alert Volume (k / #clades per year)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Top-k Coverage (Hit Rate)", fontsize=13, fontweight="bold", labelpad=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=11)
    ax.tick_params(axis="both", labelsize=11)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

    summary_lines = []
    for k_target in [1, 2, 3]:
        if k_target in ks:
            idx = int(np.where(ks == k_target)[0][0])
            summary_lines.append(
                f"k={k_target}: coverage {coverages[idx]*100:.1f}%, alert {alert_vols[idx]*100:.1f}%"
            )
    if summary_lines:
        ax.text(
            0.98,
            0.03,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main():
    ns = load_ns()
    pred_df = build_cv_predictions(ns)

    out1 = OUT_DIR / "m33_decision_curve_cv.png"
    out2 = OUT_DIR / "m33_topk_coverage_vs_alert_volume_cv.png"

    plot_decision_curve(pred_df, out1)
    plot_topk_tradeoff(pred_df, out2)

    print("[Saved]", out1)
    print("[Saved]", out2)


if __name__ == "__main__":
    main()
