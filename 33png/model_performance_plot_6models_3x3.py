import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Prefer Garamond across the figure (fallbacks if unavailable)
plt.rcParams["font.family"] = ["Garamond", "Times New Roman", "DejaVu Serif"]


IN_CSV = Path("model_comparison_output/model_metrics_table_6models.csv")
OUT_PNG = Path("model_comparison_output/plot_3x3_metrics_6models.png")


def short_label(model_name: str) -> str:
    m = re.match(r"(M\d+)", str(model_name))
    return m.group(1) if m else str(model_name)


def to_ratio(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals / 100.0


def add_value_labels(ax, bars, values):
    for b, v in zip(bars, values):
        if np.isnan(v):
            ax.text(
                b.get_x() + b.get_width() / 2,
                0.02,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=8,
                color="dimgray",
            )
        else:
            ax.text(
                b.get_x() + b.get_width() / 2,
                min(v + 0.015, 0.98),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )


def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Missing CSV: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    labels = [short_label(x) for x in df["model"]]
    x = np.arange(len(df))
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"][: len(df)]

    # 9 panels (all as ratio scale 0~1)
    metrics = [
        ("AUC", pd.to_numeric(df["auc"], errors="coerce")),
        ("PR-AUC", pd.to_numeric(df["pr_auc"], errors="coerce")),
        ("Precision", pd.to_numeric(df["precision"], errors="coerce")),
        ("Recall", pd.to_numeric(df["recall"], errors="coerce")),
        ("F1", pd.to_numeric(df["f1"], errors="coerce")),
        ("Top-1 Hit (CV)", to_ratio(df["top1_hit_pct"])),
        ("Top-3 Hit (CV)", to_ratio(df["top3_hit_pct"])),
        ("Backtest Top-1", to_ratio(df["backtest_top1_pct"])),
        ("Backtest Top-3", to_ratio(df["backtest_top3_pct"])),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle("H3N2 Clade Prediction - Model Performance Comparison", fontsize=16, fontweight="bold", y=0.98)

    for ax, (title, vals) in zip(axes.flat, metrics):
        vals_arr = vals.to_numpy(dtype=float)
        plot_vals = np.where(np.isnan(vals_arr), 0.0, vals_arr)
        bars = ax.bar(x, plot_vals, color=colors, width=0.62)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_value_labels(ax, bars, vals_arr)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)
    print(f"[Saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
