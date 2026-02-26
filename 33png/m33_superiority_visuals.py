import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_DIR = Path("model_comparison_output/logs")
OUT_DIR = Path("model_comparison_output")

MODEL_LOGS = {
    "M04": "M04_base13.log",
    "M21": "M21_balanced.log",
    "M28": "M28_noweight.log",
    "M29": "M29_nestedcv_topk.log",
    "M33": "M33_feature7_082.log",
}

BOOTSTRAP_ROUNDS = 4000
RNG_SEED = 42

# Garamond preferred
plt.rcParams["font.family"] = ["Garamond", "Times New Roman", "DejaVu Serif"]


def parse_cv_table(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    header_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        if "val_year" in s and "hit1" in s and "hit3" in s:
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame(columns=["val_year", "auc", "hit1", "hit3"])

    header = re.split(r"\s+", lines[header_idx].strip())
    header_map = {h: idx for idx, h in enumerate(header)}
    need = ["val_year", "auc", "hit1", "hit3"]
    if not all(k in header_map for k in need):
        return pd.DataFrame(columns=["val_year", "auc", "hit1", "hit3"])

    rows = []
    for ln in lines[header_idx + 1 :]:
        if not re.match(r"^\s*\d{4}\s+", ln):
            if rows:
                break
            continue
        toks = re.split(r"\s+", ln.strip())
        if len(toks) < len(header):
            continue
        try:
            row = {
                "val_year": int(float(toks[header_map["val_year"]])),
                "auc": float(toks[header_map["auc"]]),
                "hit1": float(toks[header_map["hit1"]]),
                "hit3": float(toks[header_map["hit3"]]),
            }
            rows.append(row)
        except Exception:
            continue

    return pd.DataFrame(rows)


def load_all_cv():
    rows = []
    for m, fname in MODEL_LOGS.items():
        p = LOG_DIR / fname
        if not p.exists():
            continue
        df = parse_cv_table(p)
        if df.empty:
            continue
        df["model"] = m
        rows.append(df)
    if not rows:
        raise SystemExit("No CV fold tables parsed from logs.")
    return pd.concat(rows, ignore_index=True)


def plot_heatmap_hits(df: pd.DataFrame, out_path: Path):
    models = sorted(df["model"].unique(), key=lambda x: int(x[1:]))
    years = sorted(df["val_year"].unique())

    p1 = df.pivot(index="model", columns="val_year", values="hit1").reindex(index=models, columns=years)
    p3 = df.pivot(index="model", columns="val_year", values="hit3").reindex(index=models, columns=years)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model x Year Hit Map (CV)", fontsize=18, fontweight="bold", y=0.98)

    for ax, mat, title in [
        (axes[0], p1, "Top-1 Hit by Year"),
        (axes[1], p3, "Top-3 Hit by Year"),
    ]:
        im = ax.imshow(mat.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels(years, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(models, fontsize=12, fontweight="bold")
        for i in range(len(models)):
            for j in range(len(years)):
                v = mat.values[i, j]
                txt = "-" if pd.isna(v) else str(int(v))
                ax.text(j, i, txt, ha="center", va="center", fontsize=10, fontweight="bold", color="black")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Hit (0/1)", fontsize=12, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_rank_bucket(df: pd.DataFrame, out_path: Path):
    # exact rank is unavailable in all logs; use CV-consistent bucket:
    # hit1=1 -> rank1, hit1=0 & hit3=1 -> rank2-3, hit3=0 -> outside top3
    d = df.copy()
    d["bucket"] = np.where(d["hit1"] >= 0.5, "Rank 1", np.where(d["hit3"] >= 0.5, "Rank 2-3", "Rank >3"))

    g = d.groupby(["model", "bucket"]).size().rename("n").reset_index()
    g["total"] = g.groupby("model")["n"].transform("sum")
    g["pct"] = g["n"] / g["total"]

    models = sorted(d["model"].unique(), key=lambda x: int(x[1:]))
    buckets = ["Rank 1", "Rank 2-3", "Rank >3"]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(len(models), dtype=float)

    for b, c in zip(buckets, colors):
        vals = []
        for m in models:
            v = g[(g["model"] == m) & (g["bucket"] == b)]["pct"]
            vals.append(float(v.iloc[0]) if not v.empty else 0.0)
        ax.bar(models, vals, bottom=bottoms, color=c, width=0.65, label=b)
        bottoms += np.array(vals)

    ax.set_title("True-CR Rank Bucket Distribution (CV)", fontsize=16, fontweight="bold", pad=10)
    ax.set_ylabel("Proportion of Years", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, fontsize=11)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    for t in ax.get_xticklabels():
        t.set_fontweight("bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def bootstrap_ci_mean(arr: np.ndarray, rounds: int = BOOTSTRAP_ROUNDS, seed: int = RNG_SEED):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(rounds, dtype=float)
    n = len(arr)
    for i in range(rounds):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = np.mean(sample)
    return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plot_bootstrap_ci(df: pd.DataFrame, out_path: Path):
    models = sorted(df["model"].unique(), key=lambda x: int(x[1:]))

    rows = []
    for m in models:
        sub = df[df["model"] == m]
        for metric in ["auc", "hit3"]:
            mean_v, lo, hi = bootstrap_ci_mean(sub[metric].to_numpy(dtype=float))
            rows.append({"model": m, "metric": metric.upper(), "mean": mean_v, "lo": lo, "hi": hi})
    cdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    fig.suptitle("Bootstrap 95% CI Across CV Years", fontsize=17, fontweight="bold", y=0.98)

    for ax, metric, color in [(axes[0], "AUC", "#1f77b4"), (axes[1], "HIT3", "#2ca02c")]:
        sub = cdf[cdf["metric"] == metric].copy()
        x = np.arange(len(sub))
        y = sub["mean"].to_numpy(dtype=float)
        yerr = np.vstack([y - sub["lo"].to_numpy(dtype=float), sub["hi"].to_numpy(dtype=float) - y])

        ax.errorbar(x, y, yerr=yerr, fmt="o", color=color, ecolor=color, elinewidth=2, capsize=5, markersize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"].tolist(), fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.02)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.03, f"{yi:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    df = load_all_cv()

    # keep models with CV-by-year available
    keep = sorted(df["model"].unique(), key=lambda x: int(x[1:]))
    print("[CV models]", ", ".join(keep))

    heatmap_png = OUT_DIR / "m33_superiority_heatmap_cv_hits.png"
    rank_png = OUT_DIR / "m33_superiority_rank_bucket_cv.png"
    ci_png = OUT_DIR / "m33_superiority_bootstrap_ci_cv.png"

    plot_heatmap_hits(df, heatmap_png)
    plot_rank_bucket(df, rank_png)
    plot_bootstrap_ci(df, ci_png)

    print("[Saved]", heatmap_png)
    print("[Saved]", rank_png)
    print("[Saved]", ci_png)


if __name__ == "__main__":
    main()
