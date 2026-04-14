import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from models.unet3d import UNet3D
from losses.dice_loss import DiceLoss
from segmentation.dataset import Synthetic3DDataset
from tqdm import tqdm

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset with lower noise (0.05 instead of 0.1)
    dataset = Synthetic3DDataset(num_samples=200, size=(64,64,64), noise_std=0.05)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceLoss()
    
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress = tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for img, mask in progress:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    # Save model
    Path('models').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'models/unet3d_synthetic.pth')
    print("✅ Model saved to models/unet3d_synthetic.pth")

if __name__ == '__main__':
    train()
