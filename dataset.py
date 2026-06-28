# dataset.py
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image

class SimplePedestrianDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        # Sort files to ensure images and masks match up perfectly
        self.imgs = sorted(os.listdir(self.root_dir / "PNGImages"))
        self.masks = sorted(os.listdir(self.root_dir / "PedMasks"))

    def __getitem__(self, idx):
        # 1. Load the image and its mask
        img_path = self.root_dir / "PNGImages" / self.imgs[idx]
        mask_path = self.root_dir / "PedMasks" / self.masks[idx]
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        # 2. Find unique object IDs (0 is always the background, so skip it)
        obj_ids = np.unique(mask)[1:]
        
        boxes = []
        for obj_id in obj_ids:
            # Find coordinates where this specific pedestrian exists
            pos = np.where(mask == obj_id)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # 3. Converting everything into PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Class 1 represents pedestrians. Create a '1' label for each box.
        labels = torch.ones((len(obj_ids),), dtype=torch.int64) 
        
        target = {
            "boxes": boxes,
            "labels": labels
        }

        # Converting image to standard PyTorch Tensor format [Channels, Height, Width]
        img_tensor = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

        return img_tensor, target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    """Combines images and targets into a tuple instead of a stacked matrix."""
    return tuple(zip(*batch))