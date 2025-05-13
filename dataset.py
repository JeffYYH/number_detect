import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os
import numpy as np

# Custom dataset class
class HandwritingDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.images = []
        self.labels = []
        self.transform = transform


        for label in range(10):
            label_dir = os.path.join(dataset_dir, str(label))
            if not os.path.exists(label_dir):
                continue
            for filename in os.listdir(label_dir):
                if filename.endswith('.png'):
                    img_path = os.path.join(label_dir, filename)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        if self.transform:
            img = self.transform(img)
        return img, label
    

# Data transformations with augmentation for training
train_transform = transforms.Compose([
    transforms.RandomRotation(15),  # Rotate by up to 15 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Translate and scale
    transforms.Normalize((0.5,), (0.5,))
])

# Transform for validation (no augmentation)
val_transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])