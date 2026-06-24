import sys
import yaml
import argparse
import json
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
from typing import List

import wandb

sys.path.append(str(Path(__file__).parent.parent))

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from src.training.engine import build_anomalib_engine
from scripts.train_ae import train_ae
from src.evaluation.metrics import compute_auroc, compute_pixel_auroc



def run_patchcore(config: dict, category: str, exp_dir: Path) -> dict:
    datamodule = MVTecAD(
        root=Path(config['data']['root_dir']),
        category=category,
        train_batch_size=config['training']['batch_size'],
        eval_batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )

    model = Patchcore(
        backbone=config['model']['backbone'],
        layers=["layer2", "layer3"],
        num_neighbors=9
    )

    engine = build_anomalib_engine(exp_dir, "patchcore", category)
    engine.fit(model=model, datamodule=datamodule)

    test_results = engine.test(model=model, datamodule=datamodule)

    results = {
        "model": "patchcore",
        "category": category,
        "image_AUROC": test_results[0].get("image_AUROC", 0.0),
        "pixel_AUROC": test_results[0].get("pixel_AUROC", 0.0),
    }

    wandb.finish()
    return results

def run_patchcore_per_defect(config: dict, category: str, exp_dir: Path) -> dict:
    """
    Trains PatchCore and returns per-defect-type AUROC breakdown,
    analogous to eval_model_per_defect for the autoencoder.
    """
    datamodule = MVTecAD(
        root=Path(config['data']['root_dir']),
        category=category,
        train_batch_size=config['training']['batch_size'],
        eval_batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )

    model = Patchcore(
        backbone=config['model']['backbone'],
        layers=["layer2", "layer3"],
        num_neighbors=9
    )

    engine = build_anomalib_engine(exp_dir, "patchcore", category)
    engine.fit(model=model, datamodule=datamodule)

    predictions = engine.predict(model=model, datamodule=datamodule)

    # Zbierz wszystko do jednego DataFrame
    records = []
    for batch in predictions:
        scores = batch.pred_score.cpu().numpy()
        labels = batch.gt_label.cpu().numpy()
        paths = batch.image_path
        gt_masks = batch.gt_mask.cpu().numpy()       # [B, H, W]
        anomaly_maps = batch.anomaly_map.cpu().numpy()  # [B, H, W]

        for i, path in enumerate(paths):
            # Struktura MVTec: .../test/<defect_type>/<file>.png
            defect_type = Path(path).parent.name
            records.append({
                "defect_type": defect_type,
                "score": float(scores[i]),
                "label": int(labels[i]),
                "mask": gt_masks[i],
                "anomaly_map": anomaly_maps[i],
            })

    df = pd.DataFrame(records)

    overall = {
        "model": "patchcore",
        "category": category,
        "image_AUROC": compute_auroc(df["label"].values, df["score"].values),
        "pixel_AUROC": compute_pixel_auroc(
            np.stack(df["mask"].values), np.stack(df["anomaly_map"].values)
        ),
    }

    per_defect = {}
    good_rows = df[df["defect_type"] == "good"]

    for defect_type in sorted(set(df["defect_type"]) - {"good"}):
        defect_rows = df[df["defect_type"] == defect_type]
        sub = pd.concat([defect_rows, good_rows])

        per_defect[defect_type] = {
            "n_samples": int(len(defect_rows)),
            "image_AUROC": compute_auroc(sub["label"].values, sub["score"].values),
            "pixel_AUROC": compute_pixel_auroc(
                np.stack(sub["mask"].values), np.stack(sub["anomaly_map"].values)
            ),
            "mean_defect_area": float(np.stack(defect_rows["mask"].values).mean()),
        }

    with open(exp_dir / "per_defect_results.json", "w") as f:
        json.dump(per_defect, f, indent=2)

    wandb.finish()
    return overall, per_defect

def run_patchcore_full(config: dict, category: str, exp_dir: Path) -> dict:
    overall, per_defect = run_patchcore_per_defect(config, category, exp_dir)
    overall["status"] = "ok"
    return overall

def run_autoencoder(config: dict, category: str, exp_dir: Path) -> dict:
    cfg = dict(config)
    cfg["data"] = dict(config["data"])
    cfg["model"] = dict(config.get("model", {}))
    cfg["data"]["category"] = category
    cfg["model"]["name"] = "autoencoder"

    results = train_ae(cfg, category, exp_dir)
    results["model"] = "autoencoder"
    results["category"] = category

    return results

def run_vae(config: dict, category: str, exp_dir: Path) -> dict:
    cfg = dict(config)
    cfg["data"] = dict(config["data"])
    cfg["model"] = dict(config.get("model", {}))
    cfg["data"]["category"] = category
    cfg["model"]["name"] = "vae"

    results = train_ae(cfg, category, exp_dir)
    results["model"] = "vae"
    results["category"] = category

    return results

def run_efficientad(config: dict, category: str, exp_dir: Path) -> dict:
    try:
        from anomalib.models import EfficientAd
    except ImportError:
        from anomalib.models import EfficientAD as EfficientAd

    configured_batch_size = int(config['training'].get('batch_size', 1))
    train_batch_size = int(config.get('efficientad', {}).get('train_batch_size', 1))
    eval_batch_size = int(config.get('efficientad', {}).get('eval_batch_size', configured_batch_size))

    if train_batch_size != 1:
        print(
            f"[efficientad] Overriding train_batch_size={train_batch_size} to 1, "
            "because EfficientAD requires train_batch_size=1."
        )
        train_batch_size = 1

    datamodule = MVTecAD(
        root=Path(config['data']['root_dir']),
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=config['training']['num_workers']
    )

    model = EfficientAd()

    engine = build_anomalib_engine(exp_dir, "efficientad", category)
    engine.fit(model=model, datamodule=datamodule)
    test_results = engine.test(model=model, datamodule=datamodule)

    results = {
        "model": "efficientad",
        "category": category,
        "image_AUROC": test_results[0].get("image_AUROC", 0.0),
        "pixel_AUROC": test_results[0].get("pixel_AUROC", 0.0),
    }

    wandb.finish()
    return results

MODEL_RUNNERS = {
    "autoencoder": run_autoencoder,
    "patchcore": run_patchcore_full,
    "vae": run_vae,
    "efficientad": run_efficientad,
}

def log_comparison_table(all_results: List[dict], project: str):
    """
    Logs a W&B table comparing the results of different models and categories.
    """
    wandb.init(project=project, name="comparison_summary", job_type="eval")
    
    table_data = wandb.Table(columns=["model", "category", "image_AUROC", "pixel_AUROC"])
    
    for res in all_results:
        table_data.add_data(
            res.get("model", "?"),
            res.get("category", "?"),
            f"{res.get('image_AUROC', 0.0):.4f}",
            f"{res.get('pixel_AUROC', 0.0):.4f}",
        )

    wandb.log({"comparison/all_models": table_data})

    df = pd.DataFrame(all_results)
    if not df.empty:
        for metric in ["image_AUROC", "pixel_AUROC"]:
            if metric in df.columns:
                pivot = df.pivot(index="category", columns="model", values=metric)
                wandb.log({f"comparison/{metric}_table": wandb.Table(dataframe=pivot.reset_index())})
    
    wandb.finish()

CATEGORIES = ["bottle", "carpet", "transistor", "pill"]
 
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

def main():
    parser = argparse.ArgumentParser(description="Run experiments on MVTec AD dataset")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--models", nargs="+", default=["autoencoder", "patchcore"], choices=MODEL_RUNNERS.keys())
    parser.add_argument("--categories", nargs="+", default=CATEGORIES, choices=MVTEC_CATEGORIES)
    parser.add_argument("--project", default="anomaly-detection-mvtec")
    parser.add_argument("--all_categories", action="store_true", help="Run on all MVTec categories")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    categories = MVTEC_CATEGORIES if args.all_categories else args.categories

    print(f"Running experiments for models: {args.models} on categories: {categories}")
    print(f"Project: {args.project}")
    print(f"Total runs: {len(args.models) * len(categories)}")

    all_results = []

    for model_name in args.models:
        runner = MODEL_RUNNERS[model_name]
        for category in categories:
            print(f"Running {model_name} on category '{category}'...")
            exp_dir = Path("experiments") / model_name / category / datetime.now().strftime('%Y%m%d_%H%M%S')
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                results = runner(config, category, exp_dir)
                results["status"] = "ok"
            except Exception as e:
                print(f"Error running {model_name} on category '{category}': {e}")
                results = {"model": model_name, "category": category, "image_AUROC": 0.0, "pixel_AUROC": 0.0, "status": f"error: {str(e)}"}
                wandb.finish()

            all_results.append(results)
            
            summary_path = Path("experiments") / "results_summary.json"
            with open(summary_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    if len(all_results) > 1:
        log_comparison_table(all_results, project=args.project)

    for r in all_results:
        print(f"{r['model']} - {r['category']}: Image AUROC={r.get('image_AUROC', 0.0):.4f}, Pixel AUROC={r.get('pixel_AUROC', 0.0):.4f}")

if __name__ == "__main__":
    main()