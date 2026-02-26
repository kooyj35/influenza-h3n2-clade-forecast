import re
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = [
    ("M21_balanced", "##21. zenspark_balanced_202512_final.py"),
    ("M28_noweight", "##28. zenspark_noweight_final.py"),
    ("M29_nestedcv_topk", "##29. zenspark_noweight_nestedcv_topk.py"),
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
    m = {}
    m["auc"] = _last_float(stdout, r"^\s*AUC:\s*([0-9]*\.?[0-9]+)\s*$")
    m["pr_auc"] = _last_float(stdout, r"^\s*PR-AUC:\s*([0-9]*\.?[0-9]+)\s*$")
    m["precision"] = _last_float(stdout, r"^\s*Precision:\s*([0-9]*\.?[0-9]+)\s*$")
    m["recall"] = _last_float(stdout, r"^\s*Recall:\s*([0-9]*\.?[0-9]+)\s*$")
    m["f1"] = _last_float(stdout, r"^\s*F1:\s*([0-9]*\.?[0-9]+)\s*$")
    m["top1_hit"] = _last_float(stdout, r"^\s*Top-1 Hit:\s*([0-9]*\.?[0-9]+)\s*$")
    m["top3_hit"] = _last_float(stdout, r"^\s*Top-3 Hit:\s*([0-9]*\.?[0-9]+)\s*$")

    bt_combo = re.findall(
        r"Backtest:\s*Top-1\s+\d+/\d+\s+\(([0-9]*\.?[0-9]+)%\),\s*Top-3\s+\d+/\d+\s+\(([0-9]*\.?[0-9]+)%\)",
        stdout,
        flags=re.IGNORECASE,
    )
    if bt_combo:
        m["backtest_top1_pct"] = float(bt_combo[-1][0])
        m["backtest_top3_pct"] = float(bt_combo[-1][1])
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
        m["backtest_top1_pct"] = float(bt1[-1]) if bt1 else np.nan
        m["backtest_top3_pct"] = float(bt3[-1]) if bt3 else np.nan

    w1 = re.findall(r"(?:Weighted|가중).*Top-1.*?:\s*([0-9]*\.?[0-9]+)%", stdout, flags=re.IGNORECASE)
    w3 = re.findall(r"(?:Weighted|가중).*Top-3.*?:\s*([0-9]*\.?[0-9]+)%", stdout, flags=re.IGNORECASE)
    m["weighted_backtest_top1_pct"] = float(w1[-1]) if w1 else np.nan
    m["weighted_backtest_top3_pct"] = float(w3[-1]) if w3 else np.nan
    return m


def run_model(script_path: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout + "\n" + proc.stderr


def main():
    OUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    rows = []
    for model_name, script_name in MODELS:
        script_path = Path(script_name)
        if not script_path.exists():
            print(f"[Warn] Missing script: {script_name}")
            continue

        log_path = LOG_DIR / f"{model_name}.log"
        if USE_EXISTING_LOGS and log_path.exists():
            print(f"[Reuse] {model_name}")
            text = log_path.read_text(encoding="utf-8", errors="replace")
        else:
            print(f"[Run]   {model_name}")
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

    csv_path = OUT_DIR / "model_metrics_table_5models.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] {csv_path}")

    show_cols = [
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
    print("\n[Comparison Table]")
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
