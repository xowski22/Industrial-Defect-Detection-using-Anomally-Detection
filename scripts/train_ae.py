import sys
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import argparse
from datetime import datetime
import numpy as np
import wandb

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import MVTecDataset, get_default_transforms
from src.models.ae import ConvAutoencoder
from src.training.wandb_logger import log_ae_epoch, log_ae_eval, log_roc_curve, log_anomaly_vis, init_run
from src.evaluation.metrics import eval_model, compute_anomaly_score

def train_ae(config: dict, category: str, exp_dir: Path):

    init_run("autoencoder", category, config=config)
    
    img_transform, mask_transforms = get_default_transforms(config['data']['img_size'])

    train_dataset = MVTecDataset(
        root_dir=Path(config['data']['root_dir']),
        category=config['data']['category'],
        split='train',
        transform=img_transform,
        mask_transform=mask_transforms
    )

    test_dataset = MVTecDataset(
        root_dir=Path(config['data']['root_dir']),
        category=config['data']['category'],
        split='test',
        transform=img_transform,
        mask_transform=mask_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvAutoencoder(latent_dim=128).to(device)
    num_epochs = config['training']['num_epochs']

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = torch.nn.MSELoss()

    wandb.watch(model, log="all", log_freq=50)

    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"{category} Epoch {epoch+1}/{num_epochs}"):
            images = batch['image'].to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        log_ae_epoch(epoch, avg_loss, scheduler.get_last_lr()[0])
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_dir/  f"ae_epoch_{epoch+1}.pth")
    
    print(f"{category} Evaluating...")
    results = eval_model(model, test_loader, device, method="mse")
    log_ae_eval(results, category)

    model.eval()
    all_labels, all_scores = [], []
    sample_imgs, sample_recons, sample_maps, sample_labels, sample_types = [], [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            recon = model(images)
            amaps = compute_anomaly_score(images, recon, method="mse")
            scores = amaps.view(amaps.size(0), -1).max(dim=1)[0]

            all_labels.extend(batch['label'].cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

            if len(sample_imgs) < 8:
                sample_imgs.append(images.cpu())
                sample_recons.append(recon.cpu())
                sample_maps.append(amaps.cpu())
                sample_labels.extend(batch['label'].tolist())
                sample_types.extend(batch['defect_type'])

    log_roc_curve(np.array(all_labels), np.array(all_scores), category)

    vis_imgs = torch.cat(sample_imgs)[:8]
    vis_recons = torch.cat(sample_recons)[:8]
    vis_maps = torch.cat(sample_maps)[:8]

    log_anomaly_vis(vis_imgs, vis_recons, vis_maps, sample_labels[:8], sample_types[:8], category)

    wandb.finish()
    print(f"Finished training and evaluation for category '{category}'. Image AUROC: {results['image_AUROC']:.4f}, Pixel AUROC: {results['pixel_AUROC']:.4f}, F1 Score: {results['f1_score']:.4f}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Autoencoder on MVTec AD dataset")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--category", default=None, help="Override category from config (if specified)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.category:
        config['data']['category'] = args.category
    
    category = config['data']['category']
    exp_dir = Path("experiments") / "ae" / category / datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_ae(config, category, exp_dir)