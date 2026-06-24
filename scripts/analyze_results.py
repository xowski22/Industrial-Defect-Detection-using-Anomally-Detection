import json
import pandas as pd
from pathlib import Path
from scipy import stats

TEXTURE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}

def load_results(path="experiments/results_summary.json") -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)

def hypothesis_1(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(index="category", columns="model", values="image_AUROC")
    pivot["diff"] = pivot["patchcore"] - pivot["autoencoder"]

    print("PatchCore vs Autoencoder (image AUROC)")
    print(pivot.round(4))
    print(f"\nPatchCore wygrywa w {int((pivot['diff'] > 0).sum())}/{len(pivot)} kategoriach")
    print(f"Średnie AUROC -> AE: {pivot['autoencoder'].mean():.4f}, "
          f"PatchCore: {pivot['patchcore'].mean():.4f}")

    # Test parowany (Wilcoxon) - sprawdza czy różnica jest systematyczna
    stat, p = stats.wilcoxon(pivot["patchcore"], pivot["autoencoder"])
    print(f"Wilcoxon signed-rank: stat={stat:.3f}, p={p:.4f}")
    return pivot

def hypothesis_2(pivot: pd.DataFrame) -> pd.DataFrame:
    pivot = pivot.copy()
    pivot["group"] = ["texture" if c in TEXTURE_CATEGORIES else "object" for c in pivot.index]

    grouped = pivot.groupby("group")["diff"].agg(["mean", "std", "count"])
    print("\nPrzewaga PatchCore — tekstury vs obiekty")
    print(grouped.round(4))

    texture_diffs = pivot[pivot["group"] == "texture"]["diff"]
    object_diffs = pivot[pivot["group"] == "object"]["diff"]
    # Mann-Whitney U - przy N=5 vs N=10 nie oczekuj p<0.05, ale podaj wynik
    stat, p = stats.mannwhitneyu(object_diffs, texture_diffs, alternative="greater")
    print(f"Mann-Whitney U (obiekty > tekstury): stat={stat:.3f}, p={p:.4f}")
    return grouped

if __name__ == "__main__":
    df = load_results()
    pivot = hypothesis_1(df)
    hypothesis_2(pivot)
    pivot.to_csv("experiments/h1_h2_pivot.csv")