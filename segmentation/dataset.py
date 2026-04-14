import torch
from torch.utils.data import Dataset
import numpy as np

class Synthetic3DDataset(Dataset):
    """Generate synthetic 3D volumes with spherical tumors."""
    def __init__(self, num_samples=100, size=(64,64,64), noise_std=0.05):
        self.num_samples = num_samples
        self.size = size
        self.noise_std = noise_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        vol = np.zeros(self.size, dtype=np.float32)
        # Random tumor radius between 5 and 15
        radius = np.random.randint(5, 15)
        # Random center inside volume
        center = tuple(np.random.randint(radius, s - radius) for s in self.size)
        # Draw sphere
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                for z in range(self.size[2]):
                    d2 = (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2
                    if d2 <= radius**2:
                        vol[x,y,z] = 1.0
        # Add noise
        noisy_vol = vol + np.random.normal(0, self.noise_std, self.size)
        noisy_vol = np.clip(noisy_vol, 0, 1)
        # Convert to tensors: (1, D, H, W)
        img_tensor = torch.from_numpy(noisy_vol).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(vol).float().unsqueeze(0)
        return img_tensor, mask_tensor
