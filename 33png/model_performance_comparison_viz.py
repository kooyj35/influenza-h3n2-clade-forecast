import os
import re
import sys
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


MODELS = [
    ("M21_balanced", "##21. zenspark_balanced_202512_final.py"),
    ("M28_noweight", "##28. zenspark_noweight_final.py"),
    ("M31_weighted_bootstrap", "##31. zenspark_noweight_weighted_bootstrap.py"),
    ("M33_feature7", "##33. zenspark_noweight_bootstrap_feature7개.py"),
]

OUT_DIR = Path("model_comparison_output")
LOG_DIR = OUT_DIR / "logs"
USE_EXISTING_LOGS = True


def _last_float(text: str, pattern: str):
    vals = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not vals:
        return np.nan
    return float(vals[-1])


def _to_pct(v: float):
    if pd.isna(v):
        return np.nan
    return v * 100.0 if v <= 1.0 else v


def parse_metrics(stdout: str) -> dict:
    metrics = {}

    metrics["auc"] = _last_float(stdout, r"^\s*AUC:\s*([0-9]*\.?[0-9]+)\s*$")
    metrics["pr_auc"] = _last_float(stdout, r"^\s*PR-AUC:\s*([0-9]*\.?[0-9]+)\s*$")
    metrics["precision"] = _last_float(stdout, r"^\s*Precision:\s*([0-9]*\.?[0-9]+)\s*$")
    metrics["recall"] = _last_float(stdout, r"^\s*Recall:\s*([0-9]*\.?[0-9]+)\s*$")
    metrics["f1"] = _last_float(stdout, r"^\s*F1:\s*([0-9]*\.?[0-9]+)\s*$")

    # Fold-level Top-k metrics (typically 0~1; convert to % later)
    metrics["top1_hit"] = _last_float(stdout, r"^\s*Top-1 Hit:\s*([0-9]*\.?[0-9]+)\s*$")
    metrics["top3_hit"] = _last_float(stdout, r"^\s*Top-3 Hit:\s*([0-9]*\.?[0-9]+)\s*$")

    # Backtest combined line: "Backtest: Top-1 a/b (x%), Top-3 c/d (y%)"
    bt_combo = re.findall(
        r"Backtest:\s*Top-1\s+\d+/\d+\s+\(([0-9]*\.?[0-9]+)%\),\s*Top-3\s+\d+/\d+\s+\(([0-9]*\.?[0-9]+)%\)",
        stdout,
        flags=re.IGNORECASE,
    )
    if bt_combo:
        metrics["backtest_top1_pct"] = float(bt_combo[-1][0])
        metrics["backtest_top3_pct"] = float(bt_combo[-1][1])
    else:
        bt1 = re.findall(
            r"(?:Backtest Top-1|Top-1 Hit)\s*:\s*\d+/\d+\s+\(([0-9]*\.?[0-9]+)%\)",
            stdout,
            flags=re.IGNORECASE,
        )
        bt3 = re.findall(
            r"(?:Backtest Top-3|Top-3 Hit)\s*:\s*\d+/\d+\s+\(([0-9]*\.?[0-9]+)%\)",
            stdout,
            flags=re.IGNORECASE,
        )
        metrics["backtest_top1_pct"] = float(bt1[-1]) if bt1 else np.nan
        metrics["backtest_top3_pct"] = float(bt3[-1]) if bt3 else np.nan

    # Weighted backtest default (available in newer models)
    w1 = re.findall(
        r"(?:Weighted|가중).*Top-1.*?:\s*([0-9]*\.?[0-9]+)%",
        stdout,
        flags=re.IGNORECASE,
    )
    w3 = re.findall(
        r"(?:Weighted|가중).*Top-3.*?:\s*([0-9]*\.?[0-9]+)%",
        stdout,
        flags=re.IGNORECASE,
    )
    metrics["weighted_backtest_top1_pct"] = float(w1[-1]) if w1 else np.nan
    metrics["weighted_backtest_top3_pct"] = float(w3[-1]) if w3 else np.nan

    return metrics


def run_model(script_path: Path) -> str:
    cmd = [sys.executable, str(script_path)]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout + "\n" + proc.stderr


def plot_core_metrics(df: pd.DataFrame, out_path: Path):
    metrics = ["auc", "pr_auc", "precision", "recall", "f1"]
    x = np.arange(len(df))
    width = 0.14

    plt.figure(figsize=(12, 6))
    for i, m in enumerate(metrics):
        plt.bar(x + (i - 2) * width, df[m].values, width=width, label=m.upper())
    plt.xticks(x, df["model"], rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Core Classification Metrics by Model")
    plt.legend(ncol=5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_hit_metrics(df: pd.DataFrame, out_path: Path):
    metrics = ["top1_hit_pct", "top3_hit_pct", "backtest_top1_pct", "backtest_top3_pct"]
    labels = ["Top-1 Hit", "Top-3 Hit", "Backtest Top-1", "Backtest Top-3"]
    x = np.arange(len(df))
    width = 0.18

    plt.figure(figsize=(12, 6))
    for i, (m, label) in enumerate(zip(metrics, labels)):
        plt.bar(x + (i - 1.5) * width, df[m].values, width=width, label=label)
    plt.xticks(x, df["model"], rotation=15)
    plt.ylim(0, 100)
    plt.ylabel("Percent (%)")
    plt.title("Ranking/Backtest Metrics by Model")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_weighted_backtest(df: pd.DataFrame, out_path: Path):
    sub = df[["model", "weighted_backtest_top1_pct", "weighted_backtest_top3_pct"]].dropna(how="all", subset=["weighted_backtest_top1_pct", "weighted_backtest_top3_pct"])
    if sub.empty:
        return

    x = np.arange(len(sub))
    width = 0.3
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, sub["weighted_backtest_top1_pct"].values, width=width, label="Weighted Top-1")
    plt.bar(x + width / 2, sub["weighted_backtest_top3_pct"].values, width=width, label="Weighted Top-3")
    plt.xticks(x, sub["model"], rotation=15)
    plt.ylim(0, 100)
    plt.ylabel("Percent (%)")
    plt.title("Weighted Backtest (Models That Report It)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    rows = []
    for model_name, script_name in MODELS:
        script_path = Path(script_name)
        if not script_path.exists():
            print(f"[Warn] Missing script: {script_name}")
            continue

        print(f"[Run] {model_name}: {script_name}")
        log_path = LOG_DIR / f"{model_name}.log"
        if USE_EXISTING_LOGS and log_path.exists():
            print(f"[Reuse] {log_path}")
            text = log_path.read_text(encoding="utf-8", errors="replace")
        else:
            text = run_model(script_path)
            log_path.write_text(text, encoding="utf-8")

        metrics = parse_metrics(text)
        metrics["model"] = model_name
        metrics["script"] = script_name
        rows.append(metrics)

    if not rows:
        raise SystemExit("No model outputs collected.")

    df = pd.DataFrame(rows)
    df["top1_hit_pct"] = df["top1_hit"].map(_to_pct)
    df["top3_hit_pct"] = df["top3_hit"].map(_to_pct)

    csv_path = OUT_DIR / "model_metrics_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {csv_path}")

    plot_core_metrics(df, OUT_DIR / "core_metrics_comparison.png")
    plot_hit_metrics(df, OUT_DIR / "hit_and_backtest_comparison.png")
    plot_weighted_backtest(df, OUT_DIR / "weighted_backtest_comparison.png")
    print(f"[Saved] {OUT_DIR / 'core_metrics_comparison.png'}")
    print(f"[Saved] {OUT_DIR / 'hit_and_backtest_comparison.png'}")
    print(f"[Saved] {OUT_DIR / 'weighted_backtest_comparison.png'}")

    cols = [
        "model",
        "auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "top1_hit_pct",
        "top3_hit_pct",
        "backtest_top1_pct",
        "backtest_top3_pct",
        "weighted_backtest_top1_pct",
        "weighted_backtest_top3_pct",
    ]
    print("\n[Summary]")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
