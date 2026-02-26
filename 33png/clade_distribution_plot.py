from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TEST_PATH = Path("01. TEST_KJC.csv")
VAL_PATH = Path("02. VAL_KJC.csv")
OUT_DIR = Path("model_comparison_output")

# Prefer Garamond with fallbacks
plt.rcParams["font.family"] = ["Garamond", "Times New Roman", "DejaVu Serif"]


def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", low_memory=False)
    if "qc.overallStatus" in df.columns:
        df = df[df["qc.overallStatus"].astype(str).str.lower() == "good"].copy()
    df = df.dropna(subset=["clade"]).copy()
    df["clade"] = df["clade"].astype(str).str.strip()
    df = df[df["clade"] != ""]
    return df


def plot_clade_distribution(series: pd.Series, title: str, out_path: Path, top_n: int = 12):
    counts = series.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(12, 7))
    y = counts.index.tolist()[::-1]
    x = counts.values[::-1]

    ax.barh(y, x, color="#1f77b4", edgecolor="white", linewidth=1.0)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=10)
    ax.set_xlabel("Sample Count", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Clade", fontsize=14, fontweight="bold", labelpad=10)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    test_df = load_clean(TEST_PATH)
    val_df = load_clean(VAL_PATH)
    comb = pd.concat([test_df, val_df], ignore_index=True)

    plot_clade_distribution(
        comb["clade"],
        "Clade Distribution (TEST + VAL)",
        OUT_DIR / "clade_distribution_combined_top12.png",
        top_n=12,
    )

    plot_clade_distribution(
        test_df["clade"],
        "Clade Distribution (TEST)",
        OUT_DIR / "clade_distribution_test_top12.png",
        top_n=12,
    )

    print("[Saved]", OUT_DIR / "clade_distribution_combined_top12.png")
    print("[Saved]", OUT_DIR / "clade_distribution_test_top12.png")


if __name__ == "__main__":
    main()
