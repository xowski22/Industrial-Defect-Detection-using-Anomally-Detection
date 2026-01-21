import sys
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))
from src.data.dataset import MVTecDataset, get_default_transforms
from src.models.ae import ConvAutoencoder

def train_ae(config: dict, exp_dir: Path):
    img_transform, mask_transforms = get_default_transforms(config['data']['img_size'])

    train_dataset = MVTecDataset(
        root_dir=Path(config['data']['root_dir']),
        category=config['data']['category'],
        split='train',
        transform=img_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvAutoencoder(latent_dim=128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = torch.nn.MSELoss()

    num_epochs = config['training']['num_epochs']

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in bar:
            images = batch['image'].to(device)

            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            bar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = exp_dir / "checkpoints" / f"ae_epoch_{epoch+1}.pth"
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
    
    final_path = exp_dir / "checkpoints" / "final_model.pth"
    torch.save(model.state_dict(), final_path)

    return model

config = yaml.safe_load(open("configs/baseline.yaml"))
exp_dir = Path("experiments/test_run")
exp_dir.mkdir(parents=True, exist_ok=True)

train_ae(config, exp_dir)