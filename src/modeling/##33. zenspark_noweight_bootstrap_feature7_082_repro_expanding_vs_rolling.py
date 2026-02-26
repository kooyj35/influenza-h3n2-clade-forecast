import importlib.util
import os
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_SCRIPT = Path(__file__).with_name("##33. zenspark_noweight_bootstrap_feature7_082_repro.py")
ROLLING_WINDOW_YEARS = 5


def load_base_module():
    spec = importlib.util.spec_from_file_location("m33_base_repro", str(BASE_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_rolling_cv(
    mod,
    train_df: pd.DataFrame,
    l1_ratio: float,
    c_value: float,
    min_recall: float,
    min_train_years: int,
    use_sample_weight: bool,
    window_years: int,
):
    years = sorted(train_df["year"].unique())
    rows = []
    for i in range(min_train_years, len(years)):
        val_year = years[i]
        start_idx = max(0, i - window_years)
        train_years = years[start_idx:i]

        if len(train_years) < min_train_years:
            continue

        train_fold = train_df[train_df["year"].isin(train_years)]
        val_fold = train_df[train_df["year"] == val_year]

        if train_fold.empty or val_fold.empty:
            continue
        if train_fold["y"].nunique() < 2 or val_fold["y"].nunique() < 2:
            continue

        model = mod.build_model(l1_ratio=l1_ratio, c_value=c_value)
        if use_sample_weight:
            model.fit(
                train_fold[mod.FEATURES].fillna(0),
                train_fold["y"],
                clf__sample_weight=mod.make_sample_weight(train_fold["n"]),
            )
        else:
            model.fit(train_fold[mod.FEATURES].fillna(0), train_fold["y"])
        y_proba = model.predict_proba(val_fold[mod.FEATURES].fillna(0))[:, 1]

        vf = val_fold.copy()
        vf["p_pred"] = y_proba
        rows.append(
            {
                "val_year": int(val_year),
                "train_size": int(len(train_fold)),
                "val_size": int(len(val_fold)),
                "auc": float(roc_auc_score(val_fold["y"], y_proba)),
                "pr_auc": float(average_precision_score(val_fold["y"], y_proba)),
                "hit1": int(mod.topk_hit(vf, "p_pred", 1)),
                "hit3": int(mod.topk_hit(vf, "p_pred", 3)),
                "y_true": val_fold["y"].to_numpy(),
                "y_proba": y_proba,
            }
        )

    if not rows:
        return None

    thresholds = []
    precision_vals, recall_vals, f1_vals = [], [], []
    for r in rows:
        threshold, _ = mod.find_best_threshold(r["y_true"], r["y_proba"], min_recall=min_recall)
        y_pred = (r["y_proba"] >= threshold).astype(int)
        p = precision_score(r["y_true"], y_pred, zero_division=0)
        rc = recall_score(r["y_true"], y_pred, zero_division=0)
        f = f1_score(r["y_true"], y_pred, zero_division=0)
        thresholds.append(float(threshold))
        precision_vals.append(float(p))
        recall_vals.append(float(rc))
        f1_vals.append(float(f))
        r["precision"] = float(p)
        r["recall"] = float(rc)
        r["f1"] = float(f)
        r["threshold"] = float(threshold)

    cv_df = pd.DataFrame(rows).drop(columns=["y_true", "y_proba"])
    summary = {
        "use_sample_weight": bool(use_sample_weight),
        "l1_ratio": float(l1_ratio),
        "c_value": float(c_value),
        "min_recall": float(min_recall),
        "threshold": float(np.mean(thresholds)),
        "top1": float(cv_df["hit1"].mean()),
        "top3": float(cv_df["hit3"].mean()),
        "auc": float(cv_df["auc"].mean()),
        "pr_auc": float(cv_df["pr_auc"].mean()),
        "precision": float(np.mean(precision_vals)),
        "recall": float(np.mean(recall_vals)),
        "f1": float(np.mean(f1_vals)),
        "n_folds": int(len(cv_df)),
    }
    return cv_df, summary


def search_best_params_cv(mod, train_df: pd.DataFrame, mode: str):
    results = []
    best_cv_df = None
    best_summary = None
    best_key = None

    for use_sw, l1_ratio, c_value, min_recall in product(
        mod.USE_SAMPLE_WEIGHT_GRID,
        mod.MODEL_GRID["l1_ratio"],
        mod.MODEL_GRID["C"],
        mod.MIN_RECALL_GRID,
    ):
        if mode == "expanding":
            out = mod.run_expanding_cv(
                train_df=train_df,
                l1_ratio=float(l1_ratio),
                c_value=float(c_value),
                min_recall=float(min_recall),
                min_train_years=int(mod.MIN_TRAIN_YEARS),
                use_sample_weight=bool(use_sw),
            )
        elif mode == "rolling":
            out = run_rolling_cv(
                mod=mod,
                train_df=train_df,
                l1_ratio=float(l1_ratio),
                c_value=float(c_value),
                min_recall=float(min_recall),
                min_train_years=int(mod.MIN_TRAIN_YEARS),
                use_sample_weight=bool(use_sw),
                window_years=int(ROLLING_WINDOW_YEARS),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if out is None:
            continue

        cv_df, summary = out
        if hasattr(summary, "__dict__"):
            summary = dict(summary.__dict__)
        else:
            summary = dict(summary)
        key = (
            float(summary["top3"]),
            float(summary["top1"]),
            float(summary["f1"]),
            float(summary["auc"]),
            float(summary["pr_auc"]),
        )
        record = {"mode": mode, "rolling_window_years": int(ROLLING_WINDOW_YEARS), **summary}
        results.append(record)

        if best_key is None or key > best_key:
            best_key = key
            best_cv_df = cv_df.copy()
            best_summary = dict(record)

    if best_summary is None:
        raise RuntimeError(f"No valid CV result for mode={mode}")

    result_df = pd.DataFrame(results).sort_values(
        ["top3", "top1", "f1", "auc", "pr_auc"],
        ascending=False,
        kind="mergesort",
    )
    return result_df, best_cv_df, best_summary


def save_metric_bar(comparison_df: pd.DataFrame, out_png: Path):
    metrics = ["top1", "top3", "f1", "auc", "pr_auc"]
    x = np.arange(len(metrics))
    width = 0.36
    exp = comparison_df[comparison_df["mode"] == "expanding"].iloc[0]
    roll = comparison_df[comparison_df["mode"] == "rolling"].iloc[0]
    exp_vals = [float(exp[m]) for m in metrics]
    roll_vals = [float(roll[m]) for m in metrics]

    plt.figure(figsize=(10.5, 5.2))
    plt.bar(x - width / 2, exp_vals, width=width, label="Expanding", color="#1f77b4")
    plt.bar(x + width / 2, roll_vals, width=width, label="Rolling", color="#ff7f0e")
    plt.ylim(0, 1.0)
    plt.xticks(x, [m.upper() for m in metrics])
    plt.ylabel("Score")
    plt.title(f"Expanding vs Rolling CV (rolling window={ROLLING_WINDOW_YEARS}y)")
    plt.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    mod = load_base_module()
    np.random.seed(int(mod.RANDOM_STATE))

    os.makedirs(mod.PLOT_OUTPUT_DIR, exist_ok=True)
    out_dir = Path(mod.PLOT_OUTPUT_DIR)

    print("[Load] data")
    test_raw = mod.load_nextclade(mod.TEST_PATH)
    test_cy = mod.make_clade_year_table(test_raw)
    train_df = mod.build_supervised_dataset(test_cy)
    print(f"[Info] train_df rows={len(train_df)}")

    print("[Run] expanding CV grid search")
    expanding_all, expanding_best_cv, expanding_best = search_best_params_cv(mod, train_df, mode="expanding")

    print("[Run] rolling CV grid search")
    rolling_all, rolling_best_cv, rolling_best = search_best_params_cv(mod, train_df, mode="rolling")

    comparison_df = pd.DataFrame([expanding_best, rolling_best]).copy()
    comparison_df["top1_hits"] = (comparison_df["top1"] * comparison_df["n_folds"]).round().astype(int)
    comparison_df["top3_hits"] = (comparison_df["top3"] * comparison_df["n_folds"]).round().astype(int)

    expanding_all_path = out_dir / "expanding_cv_grid_results_082_repro.csv"
    rolling_all_path = out_dir / "rolling_cv_grid_results_082_repro.csv"
    expanding_best_fold_path = out_dir / "expanding_best_fold_metrics_082_repro.csv"
    rolling_best_fold_path = out_dir / "rolling_best_fold_metrics_082_repro.csv"
    comparison_path = out_dir / "expanding_vs_rolling_comparison_082_repro.csv"
    plot_path = out_dir / "expanding_vs_rolling_comparison_082_repro.png"

    expanding_all.to_csv(expanding_all_path, index=False, encoding="utf-8-sig")
    rolling_all.to_csv(rolling_all_path, index=False, encoding="utf-8-sig")
    expanding_best_cv.to_csv(expanding_best_fold_path, index=False, encoding="utf-8-sig")
    rolling_best_cv.to_csv(rolling_best_fold_path, index=False, encoding="utf-8-sig")
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    save_metric_bar(comparison_df, plot_path)

    print("[Saved]", expanding_all_path)
    print("[Saved]", rolling_all_path)
    print("[Saved]", expanding_best_fold_path)
    print("[Saved]", rolling_best_fold_path)
    print("[Saved]", comparison_path)
    print("[Saved]", plot_path)

    cols = ["mode", "rolling_window_years", "use_sample_weight", "l1_ratio", "c_value", "min_recall", "top1", "top3", "f1", "auc", "pr_auc", "n_folds", "top1_hits", "top3_hits"]
    print("\n[Best Summary]")
    print(comparison_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
