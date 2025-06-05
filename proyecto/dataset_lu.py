import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config import IMAGE_SIZE

# ──────────────── Transformaciones ────────────────
class ToTensor:
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        image = image.astype('float32') / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)
        mask = mask.transpose(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32)
        return {"image": image, "mask": mask}

# Exportar transformación compuesta
import torchvision.transforms as transforms
transform = transforms.Compose([ToTensor()])

# ──────────────── Dataset personalizado ────────────────
class NopalDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = [f"{os.path.splitext(f)[0]}_laser.png" for f in self.image_files]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        # Imagen: escalar y replicar a RGB
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        image = np.stack([image] * 3, axis=-1)

        # Máscara: RGB normalizado
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = mask.astype('float32') / 255.0

        sample = {"image": image, "mask": mask}
        if self.transform:
            sample = self.transform(sample)
        return sample
