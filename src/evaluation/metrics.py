import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple, Dict

def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (AUROC).

    Args:
        y_true (np.ndarray): Ground truth labels
        y_scores (np.ndarray): Anomaly scores (higher means more anomalous).

    Returns:
        AUROC value as a float.
    """
    if len(np.unique(y_true)) < 2:
        return float('nan')  # Undefined AUROC if only one class present

    return roc_auc_score(y_true, y_scores)

def compute_pixel_auroc(masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    """
    Compute pixel-level AUROC.
    
    Args:
        masks: Ground truth masks [N, H, W] (0=normal pixel, 1=anomaly pixel)
        anomaly_maps: Predicted anomaly maps [N, H, W] (higher = more anomalous)
    
    Returns:
        Pixel-level AUROC
    """

    masks_flat = (masks.flatten() > 0.5).astype(int)
    scores_flat = anomaly_maps.flatten()

    valid_mask = ~np.isnan(masks_flat)

    if valid_mask.sum() == 0 or len(np.unique(masks_flat[valid_mask])) < 2:
        return float('nan')  # No valid pixels to evaluate
    
    return roc_auc_score(masks_flat[valid_mask], scores_flat[valid_mask])

def compute_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    y_pred = (y_scores >= optimal_threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return optimal_threshold, f1

def compute_anomaly_score(original: torch.Tensor, reconstructed: torch.Tensor, method: str = "mse") -> torch.Tensor:
    """
    Commputer anomaly score from reconstruction error.

    Args:
        original: Original images [B, C, H, W]
        reconstructed: Reconstructed images [B, C, H, W]
        method: 'mse', 'mae', or 'ssim'
    
    Returns:
        Anomaly scores [B] (image-level) or [B, H, W] (pixel-level)
    """
    if method == 'mse':
        error = (original - reconstructed) ** 2
        # Average over channels
        error = error.mean(dim=1)  # [B, H, W]
        
    elif method == 'mae':
        error = torch.abs(original - reconstructed)
        error = error.mean(dim=1)
        
    elif method == 'ssim':
        # Placeholder - można dodać SSIM z torchmetrics
        raise NotImplementedError("SSIM not implemented yet")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return error  # [B, H, W]

def eval_model(model, dataloader, device, method="mse") -> Dict[str, float]:
    """
    Evaluation pipeline

    Returns:
        Dictionary with 'image_AUROC', 'pixel_AUROC', optimal thresholds and F1
    """

    model.eval()

    all_labels = []
    all_scores = []
    all_masks = []
    all_anomaly_maps = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            masks = batch['mask'].cpu().numpy()

            reconstructed = model(images)
            error_maps = compute_anomaly_score(images, reconstructed, method=method)  # [B, H, W]

            image_scores = error_maps.view(error_maps.size(0), -1).max(dim=1)[0]  # [B]

            all_labels.extend(labels)
            all_scores.extend(image_scores.cpu().numpy())

            for i, mask in enumerate(masks):
                if labels[i] == 1:
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.squeeze().cpu().numpy()
                    else:
                        mask_np = np.array(mask).squeeze()
                    
                    if mask_np.sum() > 0:
                        all_masks.append(mask_np)
                        all_anomaly_maps.append(error_maps[i].cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    results = {}
    results['image_AUROC'] = compute_auroc(all_labels, all_scores)

    if len(np.unique(all_labels)) >= 2:
        results['optimal_threshold'], results['f1_score'] = compute_optimal_threshold(
            all_labels, all_scores
        )
    else:
        results['optimal_threshold'] = 0.5
        results['f1_score'] = 0.0

    if len(all_masks) > 0:
        all_masks = np.array(all_masks)
        all_anomaly_maps = np.array(all_anomaly_maps)
        results['pixel_AUROC'] = compute_pixel_auroc(all_masks, all_anomaly_maps)
    else:
        results['pixel_AUROC'] = float('nan')
    
    return results