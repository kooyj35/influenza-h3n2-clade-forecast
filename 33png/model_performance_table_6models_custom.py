import re
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = [
    ("M04_base13", "04. 13기반.py"),
    ("M06_f2", "06. zenspark_F2.py"),
    ("M21_balanced", "##21. zenspark_balanced_202512_final.py"),
    ("M28_noweight", "##28. zenspark_noweight_final.py"),
    ("M29_nestedcv_topk", "##29. zenspark_noweight_nestedcv_topk.py"),
    ("M33_feature7_082", "##33. zenspark_noweight_bootstrap_feature7_082_repro.py"),
]

OUT_DIR = Path("model_comparison_output")
LOG_DIR = OUT_DIR / "logs"
USE_EXISTING_LOGS = True


def _to_pct(v: float):
    if pd.isna(v):
        return np.nan
    return v * 100.0 if v <= 1.0 else v


def _extract_last_float(text: str, pattern: str):
    vals = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not vals:
        return np.nan
    return float(vals[-1])


def _extract_metric(label: str, text: str):
    # Matches lines like:
    # "AUC: 0.760", "Top-3 Hit: 0.82", "Top-3 Hit: 81.8%", "Top-3 Hit: 13/14 (92.9%)"
    pat = rf"^\s*{re.escape(label)}\s*:\s*([0-9]*\.?[0-9]+)\s*%?\s*$"
    v = _extract_last_float(text, pat)
    if not pd.isna(v):
        return v

    # fallback for lines with count + percent in parentheses
    pat2 = rf"{re.escape(label)}\s*:\s*\d+/\d+\s*\(([0-9]*\.?[0-9]+)%\)"
    return _extract_last_float(text, pat2)


def parse_metrics(stdout: str) -> dict:
    m = {}
    m["auc"] = _extract_metric("AUC", stdout)
    m["pr_auc"] = _extract_metric("PR-AUC", stdout)
    m["precision"] = _extract_metric("Precision", stdout)
    m["recall"] = _extract_metric("Recall", stdout)
    m["f1"] = _extract_metric("F1", stdout)
    # Prefer explicit CV lines when available
    m["top1_hit_raw"] = _extract_last_float(
        stdout, r"Top-1 Hit\s*\(CV\)\s*:\s*([0-9]*\.?[0-9]+)"
    )
    m["top3_hit_raw"] = _extract_last_float(
        stdout, r"Top-3 Hit\s*\(CV\)\s*:\s*([0-9]*\.?[0-9]+)"
    )
    if pd.isna(m["top1_hit_raw"]):
        m["top1_hit_raw"] = _extract_metric("Top-1 Hit", stdout)
    if pd.isna(m["top3_hit_raw"]):
        m["top3_hit_raw"] = _extract_metric("Top-3 Hit", stdout)

    m["threshold"] = _extract_last_float(stdout, r"Threshold\s*:\s*([0-9]*\.?[0-9]+)")
    if pd.isna(m["threshold"]):
        m["threshold"] = _extract_last_float(stdout, r"Recommended threshold.*?:\s*([0-9]*\.?[0-9]+)")

    bt_combo = re.findall(
        r"Backtest:\s*Top-1\s+\d+/\d+\s+\(([0-9]*\.?[0-9]+)%\),\s*Top-3\s+\d+/\d+\s+\(([0-9]*\.?[0-9]+)%\)",
        stdout,
        flags=re.IGNORECASE,
    )
    if bt_combo:
        m["backtest_top1_pct"] = float(bt_combo[-1][0])
        m["backtest_top3_pct"] = float(bt_combo[-1][1])
    else:
        m["backtest_top1_pct"] = _extract_last_float(
            stdout, r"Backtest Top-1\s*:\s*\d+/\d+\s*\(([0-9]*\.?[0-9]+)%\)"
        )
        m["backtest_top3_pct"] = _extract_last_float(
            stdout, r"Backtest Top-3\s*:\s*\d+/\d+\s*\(([0-9]*\.?[0-9]+)%\)"
        )
        if pd.isna(m["backtest_top1_pct"]):
            m["backtest_top1_pct"] = _extract_last_float(
                stdout, r"Top-1 Hit\s*:\s*\d+/\d+\s*\(([0-9]*\.?[0-9]+)%\)"
            )
        if pd.isna(m["backtest_top3_pct"]):
            m["backtest_top3_pct"] = _extract_last_float(
                stdout, r"Top-3 Hit\s*:\s*\d+/\d+\s*\(([0-9]*\.?[0-9]+)%\)"
            )

    m["weighted_backtest_top1_pct"] = _extract_last_float(
        stdout, r"(?:Weighted|가중).*Top-1.*?:\s*([0-9]*\.?[0-9]+)%"
    )
    m["weighted_backtest_top3_pct"] = _extract_last_float(
        stdout, r"(?:Weighted|가중).*Top-3.*?:\s*([0-9]*\.?[0-9]+)%"
    )
    return m


def run_model(script_path: Path) -> str:
    env = dict(**__import__("os").environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
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
            if "Traceback" in text:
                print(f"[Rerun] {model_name} (previous log had error)")
                text = run_model(script_path)
                log_path.write_text(text, encoding="utf-8")
        else:
            print(f"[Run]   {model_name}")
            text = run_model(script_path)
            log_path.write_text(text, encoding="utf-8")

        m = parse_metrics(text)
        m["model"] = model_name
        m["script"] = script_name
        rows.append(m)

    if not rows:
        raise SystemExit("No model outputs collected.")

    df = pd.DataFrame(rows)
    df["top1_hit_pct"] = df["top1_hit_raw"].map(_to_pct)
    df["top3_hit_pct"] = df["top3_hit_raw"].map(_to_pct)

    cols = [
        "model",
        "script",
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
        "threshold",
    ]

    out_csv = OUT_DIR / "model_metrics_table_6models.csv"
    df[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] {out_csv}")
    print("\n[Comparison Table]")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
