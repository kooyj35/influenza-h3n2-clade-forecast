import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Prefer Garamond across the figure (fallbacks if unavailable)
plt.rcParams["font.family"] = ["Garamond", "Times New Roman", "DejaVu Serif"]


IN_CSV = Path("model_comparison_output/model_metrics_table_6models.csv")
OUT_PNG = Path("model_comparison_output/plot_3x3_metrics_4models.png")
KEEP_MODELS = ["M04", "M06", "M21", "M33"]
BAR_COLORS = ["#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]  # orange, green, blue, purple


def short_label(model_name: str) -> str:
    m = re.match(r"(M\d+)", str(model_name))
    return m.group(1) if m else str(model_name)


def to_ratio(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") / 100.0


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
            y_text = v + 0.015
            if v >= 0.95:
                y_text = v + 0.03
            ax.text(
                b.get_x() + b.get_width() / 2,
                y_text,
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
    df["mshort"] = df["model"].astype(str).map(short_label)
    df = df[df["mshort"].isin(KEEP_MODELS)].copy()
    df["order"] = df["mshort"].map({m: i for i, m in enumerate(KEEP_MODELS)})
    df = df.sort_values("order").reset_index(drop=True)

    labels = df["mshort"].tolist()
    # Slightly wider spacing between categories
    x = np.arange(len(df)) * 1.08

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

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle("H3N2 Clade Prediction - Model Performance Comparison (M04/M06/M21/M33)", fontsize=15, fontweight="bold", y=0.98)

    for ax, (title, vals) in zip(axes.flat, metrics):
        vals_arr = vals.to_numpy(dtype=float)
        plot_vals = np.where(np.isnan(vals_arr), 0.0, vals_arr)
        # Slightly thinner bars
        bars = ax.bar(x, plot_vals, color=BAR_COLORS[: len(df)], width=0.52)
        ax.set_title(title, fontsize=11, fontweight="bold")
        # Add headroom so bars/labels at 1.000 do not touch the top
        ax.set_ylim(0, 1.12)
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
