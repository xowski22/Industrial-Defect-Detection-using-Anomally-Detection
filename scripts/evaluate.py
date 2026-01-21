import sys
import yaml
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import json
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from src.data.dataset import MVTecDataset, get_default_transforms
from src.utils.visualization import plot_roc_curve, create_summary, visualize_results
from src.evaluation.metrics import eval_model, compute_anomaly_score
from src.models.ae import ConvAutoencoder

def evaluate(config_path: str, checkpoint_path: Path, output_dir: Path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    img_transform, mask_transforms = get_default_transforms(config['data']['img_size'])

    test_dataset = MVTecDataset(
        root_dir=Path(config['data']['root_dir']),
        category=config['data']['category'],
        split='test',
        transform=img_transform,
        mask_transform=mask_transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    print(f"Test dataset size: {len(test_dataset)} images")
    print(f"Stats: {test_dataset.get_stats()}")

    model = ConvAutoencoder(latent_dim=128).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded model from {checkpoint_path}")

    print("Evaluating model...")
    results = eval_model(model, test_loader, device, method='mse')

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    create_summary(results, config['data']['category'], save_path=output_dir / "summary.csv")

    model.eval()
    anomaly_samples = []
    normal_samples = []

    for batch in test_loader:
        for i in range(len(batch['label'])):
            if batch['label'][i] == 1 and len(anomaly_samples) < 4:
                anomaly_samples.append({
                    'image': batch['image'][i:i+1],
                    'label': batch['label'][i],
                    'mask': batch['mask'][i:i+1],
                    'defect_type': batch['defect_type'][i]
                })
            elif batch['label'][i] == 0 and len(normal_samples) < 2:
                normal_samples.append({
                    'image': batch['image'][i:i+1],
                    'label': batch['label'][i],
                    'mask': batch['mask'][i:i+1] if batch['mask'][i] is not None else None,
                    'defect_type': batch['defect_type'][i]
                })
        
        if len(anomaly_samples) >= 4 and len(normal_samples) >= 2:
            break
    
    viz_samples = anomaly_samples + normal_samples

    if viz_samples:
        images = torch.cat([s['image'] for s in viz_samples], dim=0).to(device)
        labels = [s['label'] for s in viz_samples]
        defect_types = [s['defect_type'] for s in viz_samples]
        masks = [s['mask'] for s in viz_samples]

        with torch.no_grad():
            reconstructed = model(images)
            anomaly_maps = compute_anomaly_score(images, reconstructed, method='mse')

            visualize_results(
                images, reconstructed, anomaly_maps,
                labels, defect_types, masks,
                save_path=output_dir / "predictions.png",
                max_samples = len(viz_samples)
            )

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)

            reconstructed = model(images)
            anomaly_maps = compute_anomaly_score(images, reconstructed, method='mse')

            image_scores = anomaly_maps.view(anomaly_maps.size(0), -1).max(dim=1)[0]

            all_labels.extend(batch['label'].cpu().numpy())
            all_scores.extend(image_scores.cpu().numpy())

    plot_roc_curve(
        np.array(all_labels),
        np.array(all_scores),
        save_path=output_dir / "roc_curve.png"
    )

    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection model")
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='experiments/evaluation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint, args.output)