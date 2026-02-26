import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IN_CSV = Path("model_comparison_output/model_metrics_table_6models.csv")
OUT_DIR = Path("model_comparison_output")


def short_label(model_name: str) -> str:
    m = re.match(r"(M\d+)", str(model_name))
    return m.group(1) if m else str(model_name)


def plot_core_metrics(df: pd.DataFrame, out_path: Path):
    metrics = ["auc", "pr_auc", "precision", "recall", "f1"]
    labels = [short_label(x) for x in df["model"]]
    x = np.arange(len(df))
    width = 0.14

    plt.figure(figsize=(12, 6))
    for i, m in enumerate(metrics):
        vals = df[m].astype(float).values
        plt.bar(x + (i - 2) * width, vals, width=width, label=m.upper())
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Core Metrics Comparison (Mxx)")
    plt.legend(ncol=5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_hit_backtest(df: pd.DataFrame, out_path: Path):
    metrics = ["top1_hit_pct", "top3_hit_pct", "backtest_top1_pct", "backtest_top3_pct"]
    names = ["Top-1 CV", "Top-3 CV", "Backtest Top-1", "Backtest Top-3"]
    labels = [short_label(x) for x in df["model"]]
    x = np.arange(len(df))
    width = 0.18

    plt.figure(figsize=(12, 6))
    for i, (m, name) in enumerate(zip(metrics, names)):
        vals = df[m].astype(float).values
        plt.bar(x + (i - 1.5) * width, vals, width=width, label=name)
    plt.xticks(x, labels)
    plt.ylim(0, 100)
    plt.ylabel("Percent (%)")
    plt.title("Hit/Backtest Comparison (Mxx)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    if not IN_CSV.exists():
        raise SystemExit(f"CSV not found: {IN_CSV}")
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(IN_CSV)
    plot_core_metrics(df, OUT_DIR / "plot_core_metrics_6models.png")
    plot_hit_backtest(df, OUT_DIR / "plot_hit_backtest_6models.png")

    print("[Saved] model_comparison_output/plot_core_metrics_6models.png")
    print("[Saved] model_comparison_output/plot_hit_backtest_6models.png")
    print("[Label Rule] X-axis labels are Mxx only (e.g., M04, M06, M21, ...)")


if __name__ == "__main__":
    main()
