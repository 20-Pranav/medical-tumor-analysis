import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path.cwd()))
from models.unet3d import UNet3D
from losses.dice_loss import DiceLoss

class Synthetic3DDataset(Dataset):
    """Generate synthetic 3D volumes with spherical tumors (no real data required)."""
    def __init__(self, num_samples=50, size=(64,64,64)):
        self.num_samples = num_samples
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create empty volume
        vol = np.zeros(self.size, dtype=np.float32)
        # Random tumor radius between 5 and 15
        radius = np.random.randint(5, 15)
        # Random center inside volume (avoid edges)
        center = tuple(np.random.randint(radius, s - radius) for s in self.size)
        # Draw sphere
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                for z in range(self.size[2]):
                    d2 = (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2
                    if d2 <= radius**2:
                        vol[x,y,z] = 1.0
        # Add some noise to the image (not to mask)
        noisy_vol = vol + np.random.normal(0, 0.1, self.size)
        noisy_vol = np.clip(noisy_vol, 0, 1)
        # Convert to tensors: (1, D, H, W)
        img_tensor = torch.from_numpy(noisy_vol).float().unsqueeze(0)  # (1,D,H,W)
        mask_tensor = torch.from_numpy(vol).float().unsqueeze(0)       # (1,D,H,W)
        return img_tensor, mask_tensor

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Use synthetic data (no real BraTS needed)
    dataset = Synthetic3DDataset(num_samples=100, size=(64,64,64))
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceLoss()
    
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress = tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for img, mask in progress:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(img)       # pred shape: (B,1,D,H,W)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'models/unet3d_synthetic.pth')
    print("✅ 3D U‑Net trained on synthetic data. Model saved to models/unet3d_synthetic.pth")

if __name__ == '__main__':
    train()
