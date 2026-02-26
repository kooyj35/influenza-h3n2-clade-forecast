import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IN_CSV = Path("model_comparison_output/model_metrics_table_6models.csv")
OUT_PNG = Path("model_comparison_output/plot_style_heatmap_bar_4models.png")
KEEP = ["M04", "M06", "M21", "M33"]


def short_label(name: str) -> str:
    m = re.match(r"(M\d+)", str(name))
    return m.group(1) if m else str(name)


def pct_to_ratio(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce") / 100.0


def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Missing CSV: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    df["m"] = df["model"].map(short_label)
    df = df[df["m"].isin(KEEP)].copy()
    df["ord"] = df["m"].map({k: i for i, k in enumerate(KEEP)})
    df = df.sort_values("ord").reset_index(drop=True)

    # Left panel (heatmap): model x metric
    heat_metrics = [
        ("auc", pd.to_numeric(df["auc"], errors="coerce")),
        ("pr_auc", pd.to_numeric(df["pr_auc"], errors="coerce")),
        ("precision", pd.to_numeric(df["precision"], errors="coerce")),
        ("recall", pd.to_numeric(df["recall"], errors="coerce")),
        ("f1", pd.to_numeric(df["f1"], errors="coerce")),
        ("top1_cv", pct_to_ratio(df["top1_hit_pct"])),
        ("top3_cv", pct_to_ratio(df["top3_hit_pct"])),
        ("bt_top1", pct_to_ratio(df["backtest_top1_pct"])),
        ("bt_top3", pct_to_ratio(df["backtest_top3_pct"])),
    ]
    heat_cols = [k for k, _ in heat_metrics]
    heat = np.vstack([v.to_numpy(dtype=float) for _, v in heat_metrics]).T
    heat = np.nan_to_num(heat, nan=0.0)

    # Right panel (bar): M33 - M21 per metric
    row_m33 = df[df["m"] == "M33"].iloc[0]
    row_m21 = df[df["m"] == "M21"].iloc[0]
    bar_items = [
        ("auc", float(row_m33["auc"]) - float(row_m21["auc"])),
        ("pr_auc", float(row_m33["pr_auc"]) - float(row_m21["pr_auc"])),
        ("precision", float(row_m33["precision"]) - float(row_m21["precision"])),
        ("recall", float(row_m33["recall"]) - float(row_m21["recall"])),
        ("f1", float(row_m33["f1"]) - float(row_m21["f1"])),
        ("top1_cv", float(row_m33["top1_hit_pct"] - row_m21["top1_hit_pct"]) / 100.0),
        ("top3_cv", float(row_m33["top3_hit_pct"] - row_m21["top3_hit_pct"]) / 100.0),
        ("bt_top1", float(row_m33["backtest_top1_pct"] - row_m21["backtest_top1_pct"]) / 100.0),
        ("bt_top3", float(row_m33["backtest_top3_pct"] - row_m21["backtest_top3_pct"]) / 100.0),
    ]
    bar_items = sorted(bar_items, key=lambda x: abs(x[1]), reverse=True)
    bar_labels = [k for k, _ in bar_items]
    bar_vals = np.array([v for _, v in bar_items], dtype=float)
    bar_colors = ["#35c46d" if v >= 0 else "#e74c3c" for v in bar_vals]

    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.28)
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    fig.suptitle("H3N2 Model Performance - Heatmap & Delta View", fontsize=20, fontweight="bold", y=0.98)

    im = ax_hm.imshow(heat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax_hm.set_title("Model x Metric Heatmap", fontsize=14, fontweight="bold", pad=12)
    ax_hm.set_xticks(np.arange(len(heat_cols)))
    ax_hm.set_xticklabels(heat_cols, rotation=30, ha="right")
    ax_hm.set_yticks(np.arange(len(df)))
    ax_hm.set_yticklabels(df["m"].tolist(), fontsize=12, fontweight="bold")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i, j]
            ax_hm.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color="black", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.035, pad=0.02)
    cbar.set_label("Metric value (0-1)")

    y = np.arange(len(bar_vals))
    ax_bar.barh(y, bar_vals, color=bar_colors, height=0.62)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(bar_labels, fontsize=11, fontweight="bold")
    ax_bar.invert_yaxis()
    ax_bar.axvline(0, color="gray", linewidth=1)
    ax_bar.set_title("M33 vs M21 Delta by Metric", fontsize=14, fontweight="bold", pad=12)
    ax_bar.set_xlabel("Delta (M33 - M21)")
    for yi, v in enumerate(bar_vals):
        x = v + (0.002 if v >= 0 else -0.002)
        ha = "left" if v >= 0 else "right"
        ax_bar.text(x, yi, f"{v:+.3f}", va="center", ha=ha, fontsize=10, fontweight="bold")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.25)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.text(
        0.5,
        0.02,
        "Key insight: M33 outperforms M21 on AUC/PR-AUC and Top-3 ranking while keeping recall similar.",
        ha="center",
        fontsize=11,
        color="#1f4e79",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f2fb", edgecolor="#4a90d9", linewidth=1.5),
    )

    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)
    print(f"[Saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
