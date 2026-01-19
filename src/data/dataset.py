import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter

class MVTecDataset(Dataset):
    """
    MVTec Anomaly Detection Dataset Loader.
    This class handles loading images and their corresponding labels from the MVTec dataset.

    Structure expected:
    data/raw/mvtec_anomaly_detection/
        ├── bottle/
        │   ├── train/good/*.png
        │   ├── test/good/*.png
        │   └── test/broken_large/*.png
        └── cable/...
    """

    def __init__(self, 
                 root_dir: str, category: str, split: str = "train", 
                 transform: Optional[transforms.Compose] = None, mask_transform: Optional[transforms.Compose] = None
                ):
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform

        self.image_paths = []
        self.labels = []
        self.mask_paths = []
        self.defect_types = []

        self._load_dataset()
    
    def _load_dataset(self):
        category_path = self.root_dir / self.category / self.split

        if self.split == "train":
            # Train set only contains 'good' images, with no defects
            good_dir = category_path / "good"
            for img_path in sorted(good_dir.glob("*.png")):
                self.image_paths.append(img_path)
                self.labels.append(0)  # 0 for normal
                self.mask_paths.append(None)
                self.defect_types.append("good")

        else:
            for defect_dir in sorted(category_path.iterdir()):
                if not defect_dir.is_dir():
                    continue
                
                defect_type = defect_dir.name
                is_good = (defect_type == "good")

                mask_dir = self.root_dir / self.category / "ground_truth" / defect_type

                for img_path in sorted(defect_dir.glob("*.png")):
                    self.image_paths.append(img_path)
                    self.labels.append(0 if is_good else 1)  # 0 for normal, 1 for defect
                    self.defect_types.append(defect_type)

                    if is_good:
                        self.mask_paths.append(None)
                    else:
                        mask_name = img_path.stem + "_mask.png"
                        mask_path = mask_dir / mask_name
                        self.mask_paths.append(mask_path if mask_path.exists() else None)
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        """
        Dict contains keys 'image', 'label', 'mask' (if available), 'defect_type'
        """
        img = Image.open(self.image_paths[index]).convert("RGB")

        mask = None
        if self.mask_paths[index] is not None:
            mask = Image.open(self.mask_paths[index]).convert("L")
        
        if self.transform:
            img = self.transform(img)
        
        if mask and self.mask_transform:
            mask = self.mask_transform(mask)
        
        return {
            "image": img,
            "label": self.labels[index],
            "mask": mask,
            "defect_type": self.defect_types[index],
            "image_path": str(self.image_paths[index])
        }
    
    def get_stats(self) -> dict:
        return {
            "total_images": len(self),
            "normal": sum(1 for label in self.labels if label == 0),
            "anomaly": sum(1 for label in self.labels if label == 1),
            "defect_types": dict(Counter(self.defect_types))
        }
    
def get_default_transforms(image_size: int = 256) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    default transforms for mvtec dataset
    
    :param image_size: The size to which images and masks will be resized
    :type image_size: int
    :return: A tuple containing the image and mask transforms
    :rtype: Tuple[Compose, Compose]
    """
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    return image_transform, mask_transform