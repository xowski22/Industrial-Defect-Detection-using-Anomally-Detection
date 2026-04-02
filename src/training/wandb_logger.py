import wandb
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Use a non-interactive backend for plotting

from pathlib import Path
from sklearn.metrics import roc_curve, auc
from typing import Dict, Any, List
from anomalib.loggers import AnomalibWandbLogger
from anomalib.engine import Engine


def init_run(
    model_name: str,
    category: str,
    config: Dict[str, Any],
    project: str = "anomaly-detection-mvtec"
) -> wandb.sdk.wandb_run.Run:
    """
    Initializes a Weights & Biases run with the given configuration.

    Args:
        model_name (str): The name of the model being trained.
        category (str): The category or type of the model.
        config (Dict[str, Any]): A dictionary containing the training configuration and hyperparameters.
        project (str): The name of the W&B project to log to.

    Returns:
        wandb.sdk.wandb_run.Run: The initialized W&B run object.
    """
    run = wandb.init(
        project=project,
        name=f"{model_name}_{category}",
        group=category,
        tags=[model_name, category],
        config={
            "model": model_name,
            "category": category,
            **config,
        },
        reinit=True
    )
    return run

# AE helpels

def log_ae_epoch(epoch: int, loss: float, lr: float):
    """Call once per trainig epoch to log the training loss and learning rate."""
    wandb.log({"train/loss": loss, "train/lr": lr, "epoch": epoch})

def log_ae_eval(results: Dict[str, float], category: str):
    """
    Final evaluation results after training is done.
    
    results keys: image_AUROC, pixel_AUROC, f1_score, optimal_threshold
    """

    metrics = {
        f"eval/{category}/image_auroc": results.get("image_AUROC", 0.0),
        f"eval/{category}/pixel_auroc": results.get("pixel_AUROC", 0.0),
        f"eval/{category}/f1": results.get("f1_score", 0.0),
    }
    wandb.log(metrics)
    
    print(f"Logged evaluation results for category '{category}': {metrics}")

def log_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    category: str,
):
    """
    Logs ROC curve as a W&B plot for the given true labels and predicted scores.
    """
    fpr, tpr = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f'ROC-{category}')
    ax.legend()
    ax.grid(alpha=0.3)

    wandb.log({f"roc_curve/{category}": wandb.Image(fig)})
    plt.close(fig)

def log_anomaly_vis(
    images: torch.Tensor,
    reconstructed: torch.Tensor,
    anomaly_maps: torch.Tensor,
    labels: List[int],
    defect_types: List[str],
    category: str,
    max_samples: int = 8
):
    """Logs a grid of original images, reconstructions, and anomaly maps to W&B."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    n = min(len(images), max_samples)
    wandb_images = []

    for i in range(n):
        img = (images[i].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        recon = (reconstructed[i].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        anomaly_map = anomaly_maps[i].cpu().numpy()

        norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        heatmap = plt.cm.jet(norm_map)[:, :, :3]  # Get RGB from colormap
        overlay = 0.5 * img + 0.5 * heatmap

        panel = np.concatenate([img, recon, heatmap, overlay], axis=1)
        label = f"{defect_types[i]} (label={labels[i]})"
        wandb_images.append(wandb.Image(panel, caption=label))

    wandb.log({f"visualizations/{category}": wandb_images})
    print(f"Logged {len(wandb_images)} anomaly visualization panels for category '{category}'")

def init_anomalib_engine(exp_dir: Path, model_name: str, category: str, project: str="anomaly-detection-mvtec"):
    """
    Build an Anomalib Engine with a W&B logger
    Logs anomaly maps, heatmaps, and evaluation images

    Usage:

    """

    logger = AnomalibWandbLogger(
        project=project,
        name=f"{model_name}_{category}",
        group=category,
        tags=[model_name, category],
        log_model=False # Set to True to log model checkpoints
    )

    engine = Engine(
        max_epochs=1,
        devices=1,
        accelerator="auto",
        default_root_dir=str(exp_dir),
        logger=logger
    )

    return engine