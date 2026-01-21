import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import List, Optional
from pathlib import Path
import csv

def denormalize_img(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Image tensor [C, H, W] normalized with ImageNet stats
    
    Returns:
        Numpy array [H, W, C] in [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    denorm = (tensor * std + mean).clamp(0, 1)
    return denorm.permute(1, 2, 0).cpu().numpy()

def apply_colormap(anomaly_map: np.ndarray, vmin: Optional[float] = None, 
                   vmax: Optional[float] = None) -> np.ndarray:
    """
    Apply colormap to anomaly map for better visualization.
    
    Args:
        anomaly_map: 2D array [H, W]
        vmin, vmax: Value range for normalization
    
    Returns:
        RGB image [H, W, 3]
    """

    if vmin is None:
        vmin = anomaly_map.min()
    if vmax is None:
        vmax = anomaly_map.max()
    
    normalized = (anomaly_map - vmin) / (vmax - vmin + 1e-8)
    normalized = np.clip(normalized, 0, 1)

    cmap = plt.cm.jet
    colored = cmap(normalized)[:, :, :3]  # Drop alpha channel

    return colored

def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay heatmap on the original image.
    
    Args:
        image: Original image [H, W, 3] in [0, 1]
        heatmap: Heatmap image [H, W, 3] in [0, 1]
        alpha: Transparency factor for heatmap overlay
    
    Returns:
        Overlayed image [H, W, 3]
    """
    return (1 - alpha) * image + alpha * heatmap

def visualize_results(images: torch.Tensor,
                      reconstructed: torch.Tensor,
                      anomaly_maps: torch.Tensor,
                      labels: List[int],
                      defect_types: List[str],
                      masks: Optional[torch.Tensor] = None,
                      save_path: Optional[Path] = None,
                      max_samples: int = 8):
    """
    Visualize original images, reconstructions and anomaly maps.
    Args:
        images: Original images tensor [B, C, H, W]
        reconstructed: Reconstructed images tensor [B, C, H, W]
        anomaly_maps: Anomaly maps tensor [B, H, W]
        labels: Ground truth labels
        defect_types: Defect type strings
        masks: Ground truth masks tensor [B, 1, H, W] (optional)
        save_path: Path to save visualizations
        max_samples: Maximum number of samples to visualize
    """

    n_samples = min(len(images), max_samples)
    n_cols = 5 if masks is not None else 4

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]  # Ensure 2D array for consistency
    
    for i in range(n_samples):
        img = denormalize_img(images[i])
        recon = denormalize_img(reconstructed[i])

        anomaly_map = anomaly_maps[i].cpu().numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original\n{defect_types[i]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(recon)
        axes[i, 1].set_title("Reconstruction")
        axes[i, 1].axis('off')

        error = np.abs(img - recon).mean(axis=2)
        im = axes[i, 2].imshow(error, cmap='hot', vmin=0, vmax=0.3)
        axes[i,2].set_title("Reconstruction Error")
        axes[i,2].axis('off')
        plt.colorbar(im, ax=axes[i,2], fraction=0.046)

        heatmap = apply_colormap(anomaly_map)
        overlay = overlay_heatmap(img, heatmap, alpha=0.5)

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f"Anomaly Score\nLabel {labels[i]}")
        axes[i, 3].axis('off')

        if masks is not None and i < len(masks):
            mask = masks[i]
            if mask is not None and not torch.all(mask == 0):
                gt_mask = mask.squeeze().cpu().numpy()
                axes[i, 4].imshow(gt_mask, cmap='gray')
                axes[i, 4].set_title('Ground Truth')
            else:
                axes[i, 4].text(0.5, 0.5, 'No defect', ha='center', va='center')
            axes[i, 4].axis('off')
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: Optional[Path] = None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()

def create_summary(results: dict, category: str, save_path: Optional[Path] = None):
    """Text summary of results."""
    summary = f"""
Evaluation results: {category.upper()}

Image-level Metrics:
\tAUROC: {results['image_AUROC']:.4f}
\tF1 Score: {results['f1_score']:.4f}
\tOptimal Threshold: {results['optimal_threshold']:.4f}

Pixel-level Metrics:
\tAUROC: {results.get('pixel_AUROC', 'N/A')}
"""
    
    print(summary)
    
    if save_path:
        csv_path = save_path.with_suffix('.csv')
        file_exists = csv_path.exists()

        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Category', 'Image_AUROC', 'F1_Score', 'Optimal_Threshold', 'Pixel_AUROC'])
            writer.writerow([
                category,
                f"{results['image_AUROC']:.4f}",
                f"{results['f1_score']:.4f}",
                f"{results['optimal_threshold']:.4f}",
                f"{results.get('pixel_AUROC', 'N/A')}"
            ])
        print(f"Results summary saved to {csv_path}")
    return summary