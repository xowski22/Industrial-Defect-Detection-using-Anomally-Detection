import sys
import os
import yaml
import argparse
from pathlib import Path
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from anomalib.data import Folder, MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.metrics import AUROC, AUPR

sys.path.append(str(Path(__file__).parent.parent))
from src.data.dataset import MVTecDataset, get_default_transforms

parser = argparse.ArgumentParser(description="Train anomaly detection model")
parser.add_argument('--config', type=str, default="configs/baseline.yaml", help='Path to the config file')


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def setup_experiment_dir(config: dict) -> Path:
    base_dir = Path(config['output']['experiment_dir'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / 'config.yaml', 'w') as file:
        yaml.dump(config, file)

    return exp_dir

def train_patchcore(config: dict, exp_dir: Path):
    datamodule = MVTecAD(
        root=config['data']['root_dir'],
        category=config['data']['category'],
        train_batch_size=config['training']['batch_size'],
        eval_batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
    )
    datamodule.setup()

    model = Patchcore(
        backbone=config['model']['backbone'],
        layers=["layer2", "layer3"],
        num_neighbors=9,
    )

    engine = Engine(
        max_epochs=1,
        devices=1,
        accelerator="auto",
        default_root_dir=str(exp_dir),
        logger=True,
    )

    print(f"Training Pathcore on {config['data']['category']} category...")
    engine.fit(model=model, datamodule=datamodule)

    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'patchcore_final.ckpt'
    engine.trainer.save_checkpoint(str(checkpoint_path))
    print(f"Model checkpoint saved at {checkpoint_path}")

    print("Evaluating the model...")
    test_results = engine.test(model=model, datamodule=datamodule)
    
    results = {
        "category": config['data']['category'],
        "model": config['model']['name'],
        "image_AUROC": test_results[0]['image_AUROC'],
        "pixel_AUROC": test_results[0]['pixel_AUROC'],
        "timestamp": datetime.now().isoformat()
    }

    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results for {config['data']['category']}")
    print(f"\tImage AUROC: {results['image_AUROC']:.3f}")
    print(f"\tPixel AUROC: {results['pixel_AUROC']:.3f}")

    return results

if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    exp_dir = setup_experiment_dir(config)

    if config['model']['name'] == 'patchcore':
        results = train_patchcore(config, exp_dir)
    else:
        raise ValueError(f"Model {config['model']['name']} not implemented.")