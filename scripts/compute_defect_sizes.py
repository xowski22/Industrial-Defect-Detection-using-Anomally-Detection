import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.data.dataset import MVTecDataset, get_default_transforms

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

def main(root_dir="data/raw/mvtec_anomaly_detection", img_size=256):
    img_transform, mask_transform = get_default_transforms(img_size)
    records = []

    for category in MVTEC_CATEGORIES:
        ds = MVTecDataset(
            root_dir=root_dir, category=category, split="test",
            transform=img_transform, mask_transform=mask_transform,
        )
        for i in range(len(ds)):
            sample = ds[i]
            if sample["defect_type"] == "good":
                continue
            records.append({
                "category": category,
                "defect_type": sample["defect_type"],
                "area_fraction": float(sample["mask"].mean()),
            })
        print(f"{category}: processed {len(ds)} test samples")

    df = pd.DataFrame(records)
    summary = (
        df.groupby(["category", "defect_type"])["area_fraction"]
        .mean()
        .reset_index()
    )

    out_path = Path("experiments") / "defect_sizes.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(summary)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()