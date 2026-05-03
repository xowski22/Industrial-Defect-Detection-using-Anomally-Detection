import torch
import numpy as np
import torch.nn.functional as F
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

def _ssim_error_map(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """
    Compute per-pixel anomaly map from SSIM as (1 - SSIM) / 2.

    Args:
        original: Tensor [B, C, H, W]
        reconstructed: Tensor [B, C, H, W]
        window_size: Local averaging window for SSIM.

    Returns:
        Tensor [B, H, W] with higher values indicating stronger anomalies.
    """
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    padding = window_size // 2

    mu_x = F.avg_pool2d(original, kernel_size=window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(reconstructed, kernel_size=window_size, stride=1, padding=padding)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(original * original, window_size, stride=1, padding=padding) - mu_x2
    sigma_y2 = F.avg_pool2d(reconstructed * reconstructed, window_size, stride=1, padding=padding) - mu_y2
    sigma_xy = F.avg_pool2d(original * reconstructed, window_size, stride=1, padding=padding) - mu_xy

    ssim_num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    ssim_den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = ssim_num / (ssim_den + 1e-8)
    ssim_map = torch.clamp(ssim_map, -1.0, 1.0)

    anomaly_map = (1.0 - ssim_map) / 2.0
    return anomaly_map.mean(dim=1)

def _gaussian_kernel2d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d

def apply_gaussian_smoothing(error_map: torch.Tensor, sigma: float = 0.0) -> torch.Tensor:
    """
    Apply Gaussian smoothing to anomaly maps.

    Args:
        error_map: Tensor [B, H, W]
        sigma: Standard deviation of Gaussian kernel.

    Returns:
        Smoothed anomaly maps [B, H, W].
    """
    if sigma <= 0:
        return error_map

    kernel = _gaussian_kernel2d(sigma=sigma, device=error_map.device, dtype=error_map.dtype)
    kernel_size = kernel.shape[0]
    padding = kernel_size // 2

    x = error_map.unsqueeze(1)
    x = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    x = F.conv2d(x, kernel.unsqueeze(0).unsqueeze(0))
    return x.squeeze(1)

def compute_anomaly_score(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    method: str = "mse",
    gaussian_sigma: float = 0.0,
) -> torch.Tensor:
    """
    Commputer anomaly score from reconstruction error.

    Args:
        original: Original images [B, C, H, W]
        reconstructed: Reconstructed images [B, C, H, W]
        method: 'mse', 'mae', or 'ssim'
        gaussian_sigma: Sigma for optional Gaussian smoothing of anomaly maps.
    
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
        error = _ssim_error_map(original, reconstructed)
    
    else:
        raise ValueError(f"Unknown method: {method}")

    error = apply_gaussian_smoothing(error, sigma=gaussian_sigma)
    
    return error  # [B, H, W]

def eval_model(model, dataloader, device, method="mse", gaussian_sigma: float = 0.0) -> Dict[str, float]:
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

            model_output = model(images)
            if isinstance(model_output, (tuple, list)):
                reconstructed = model_output[0]
            else:
                reconstructed = model_output

            error_maps = compute_anomaly_score(
                images,
                reconstructed,
                method=method,
                gaussian_sigma=gaussian_sigma,
            )  # [B, H, W]

            image_scores = error_maps.view(error_maps.size(0), -1).max(dim=1)[0]  # [B]

            all_labels.extend(labels)
            all_scores.extend(image_scores.cpu().numpy())

            for i, mask in enumerate(masks):
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.squeeze().cpu().numpy()
                else:
                    mask_np = np.array(mask).squeeze()

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