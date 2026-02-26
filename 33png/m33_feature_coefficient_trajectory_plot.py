import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Prefer Garamond across the figure (fallbacks if unavailable)
plt.rcParams["font.family"] = ["Garamond", "Times New Roman", "DejaVu Serif"]


TARGET_SCRIPT = "##33. zenspark_noweight_bootstrap_feature7_082_repro.py"
OUT_PNG = Path("model_comparison_output/m33_cv_feature_coefficient_trajectory.png")


def stability_badge(vals: np.ndarray, eps: float = 1e-2):
    abs_vals = np.abs(vals)
    nz_rate = float(np.mean(abs_vals > eps))
    nonzero_vals = vals[abs_vals > eps]

    if len(nonzero_vals) == 0:
        return f"Sparse (SC N/A | NZ {nz_rate*100:.0f}%)", "#7f8c8d"

    if np.mean(nonzero_vals) >= 0:
        sign_consistency = float(np.mean(nonzero_vals >= 0))
    else:
        sign_consistency = float(np.mean(nonzero_vals < 0))

    if sign_consistency >= 0.8 and nz_rate < 0.35:
        return f"Sparse but Stable (SC {sign_consistency*100:.0f}% | NZ {nz_rate*100:.0f}%)", "#35c46d"
    if sign_consistency >= 0.8:
        return f"Stable (SC {sign_consistency*100:.0f}% | NZ {nz_rate*100:.0f}%)", "#35c46d"
    if sign_consistency >= 0.6:
        return f"Moderate (SC {sign_consistency*100:.0f}% | NZ {nz_rate*100:.0f}%)", "#f39c3d"
    return f"Variable (SC {sign_consistency*100:.0f}% | NZ {nz_rate*100:.0f}%)", "#e74c3c"


def main():
    script = Path(TARGET_SCRIPT)
    if not script.exists():
        raise SystemExit(f"Missing script: {TARGET_SCRIPT}")

    ns = runpy.run_path(str(script))

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

    # 1) Load data and select best hyperparameters under the same pipeline
    test_raw = load_nextclade(TEST_PATH)
    val_raw = load_nextclade(VAL_PATH)
    if test_raw.empty or val_raw.empty:
        raise SystemExit("Data loading failed.")

    test_cy = make_clade_year_table(test_raw)
    val_cy = make_clade_year_table(val_raw)
    train_df = build_supervised_dataset(test_cy)
    _, best = search_best_params(train_df, test_cy=test_cy, val_cy=val_cy, min_train_years=MIN_TRAIN_YEARS)

    # 2) Re-run expanding folds and capture per-fold coefficients
    years = sorted(train_df["year"].unique())
    rows = []
    for i in range(MIN_TRAIN_YEARS, len(years)):
        val_year = years[i]
        train_years = years[:i]
        train_fold = train_df[train_df["year"].isin(train_years)]
        val_fold = train_df[train_df["year"] == val_year]
        if train_fold.empty or val_fold.empty:
            continue
        if train_fold["y"].nunique() < 2 or val_fold["y"].nunique() < 2:
            continue

        model = build_model(best.l1_ratio, best.c_value)
        if best.use_sample_weight:
            model.fit(
                train_fold[FEATURES].fillna(0),
                train_fold["y"],
                clf__sample_weight=make_sample_weight(train_fold["n"]),
            )
        else:
            model.fit(train_fold[FEATURES].fillna(0), train_fold["y"])

        coef = model.named_steps["clf"].coef_[0]
        row = {"val_year": int(val_year)}
        for f, c in zip(FEATURES, coef):
            row[f] = float(c)
        rows.append(row)

    coef_df = pd.DataFrame(rows)
    if coef_df.empty:
        raise SystemExit("No valid folds for coefficient trajectory.")

    fold_labels = [f"'{str(y)[-2:]}" for y in coef_df["val_year"].tolist()]
    x = np.arange(len(coef_df))

    # 3) Plot (2x4 layout, last panel reserved/blank for style balance)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("CV Fold-wise Feature Coefficient Trajectory (M33)", fontsize=17, fontweight="bold", y=0.98)
    fig.subplots_adjust(hspace=0.33)

    features_for_panels = FEATURES + ["_EMPTY_"]
    for ax, feat in zip(axes.flat, features_for_panels):
        if feat == "_EMPTY_":
            ax.axis("off")
            continue

        vals = coef_df[feat].to_numpy(dtype=float)
        mean_v = float(np.mean(vals))
        badge_txt, badge_color = stability_badge(vals)

        pos_color = "#35c46d"
        neg_color = "#e74c3c"
        line_color = pos_color if mean_v >= 0 else neg_color
        bg_color = "#d9efe1" if mean_v >= 0 else "#f6dddd"

        ymin = min(np.min(vals), mean_v, 0.0)
        ymax = max(np.max(vals), mean_v, 0.0)
        pad = (ymax - ymin) * 0.12 + 0.02

        ax.axhspan(ymin - pad * 0.4, ymax + pad * 0.4, color=bg_color, alpha=0.6, zorder=0)
        ax.axhline(0, color="gray", linewidth=1)
        ax.plot(x, vals, color=line_color, linewidth=2, marker="o", markersize=4, markeredgecolor="white")
        ax.axhline(mean_v, color=line_color, linestyle="--", alpha=0.55)

        ax.set_title(feat, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels, fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(ymin - pad, ymax + pad)

        ax.text(
            0.01,
            0.05,
            f"mean={mean_v:+.3f}",
            transform=ax.transAxes,
            fontsize=9,
            color=line_color,
            fontweight="bold",
        )
        ax.text(
            0.98,
            0.95,
            badge_txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="black",
            fontweight="bold",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)
    print(f"[Saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
