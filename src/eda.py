from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import load_dataset, FEATURES

def main():
    df = load_dataset()
    output = Path("reports/figures")
    output.mkdir(parents=True, exist_ok=True)

    # 1. Feature Distributions
    df[FEATURES].hist(figsize=(14, 9), bins=20, edgecolor="black")
    plt.suptitle("SmartCrop AI - Feature Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(output / "feature_distributions.png", dpi=160)
    plt.close()

    # 2. Outlier Inspection Boxplots
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df[FEATURES])
    plt.title("SmartCrop AI - Feature Outlier Inspection", fontsize=14)
    plt.tight_layout()
    plt.savefig(output / "feature_boxplots.png", dpi=160)
    plt.close()

    # 3. Correlation Heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[FEATURES].corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("SmartCrop AI - Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig(output / "correlation_heatmap.png", dpi=160)
    plt.close()

    # 4. Crop-Wise Average Analysis
    means = df.groupby("label")[FEATURES].mean()
    means.round(2).to_csv(output / "crop_wise_averages.csv")

    # Min-max normalize per feature column to display balanced color intensities while annotating actual means
    norm_means = (means - means.min()) / (means.max() - means.min())
    plt.figure(figsize=(12, 11))
    sns.heatmap(
        norm_means,
        annot=means.round(1),
        fmt="",
        cmap="YlGnBu",
        cbar_kws={"label": "Normalized Requirement (0 = Lowest, 1 = Highest among crops)"},
        linewidths=0.5,
    )
    plt.title("SmartCrop AI - Crop-Wise Average Requirements (Annotated with Actual Means)", fontsize=13, pad=12)
    plt.xlabel("Features", fontsize=11)
    plt.ylabel("Crop Category", fontsize=11)
    plt.tight_layout()
    plt.savefig(output / "crop_wise_averages.png", dpi=160)
    plt.close()

    print(f"EDA figures and tabular summaries saved to {output.resolve()}")

if __name__ == "__main__":
    main()
