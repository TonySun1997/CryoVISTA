import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
import config
from PIL import Image

class AugmentedCryoEMDataset(Dataset):
    """CryoEM dataset with data augmentation to reduce overfitting"""
    
    def __init__(self, img_dir, transform=None, augment=True):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.img_dir)
    
    def apply_augmentation(self, image, mask):
        """Apply data augmentation"""
        if not self.augment:
            return image, mask
        
        img_np = image.squeeze(0).numpy()  # [H, W]
        mask_np = mask.squeeze(0).numpy()  # [H, W]
        
        if random.random() > 0.5:
            img_np = np.fliplr(img_np).copy()
            mask_np = np.fliplr(mask_np).copy()
        
        if random.random() > 0.5:
            img_np = np.flipud(img_np).copy()
            mask_np = np.flipud(mask_np).copy()
        
        if random.random() > 0.5:
            k = random.randint(1, 3)
            img_np = np.rot90(img_np, k).copy()
            mask_np = np.rot90(mask_np, k).copy()
        
        if random.random() > 0.3:
            brightness_factor = random.uniform(0.8, 1.2)
            img_np = np.clip(img_np * brightness_factor, 0, 1)
        
        if random.random() > 0.3:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = img_np.mean()
            img_np = np.clip((img_np - mean) * contrast_factor + mean, 0, 1)
        
        if random.random() > 0.5:
            noise_std = random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_std, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 1)
        
        image = torch.from_numpy(img_np).unsqueeze(0).float()
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()
        
        return image, mask

    def __getitem__(self, idx):
        image_path = self.img_dir[idx]
        mask_path = image_path[:-4] + '_mask.jpg'
        mask_path = mask_path.replace('images', 'masks')
        
        image = np.array(Image.open(image_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))
        
        image = np.array(Image.fromarray(image).resize((config.input_image_width, config.input_image_height)))
        mask = np.array(Image.fromarray(mask).resize((config.input_image_width, config.input_image_height)))
        
        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        image = image/255.0
        
        mask = (mask > 127.5).float()
        
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        return (image, mask)
