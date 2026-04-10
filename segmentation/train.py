import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from models.tumor_maskrcnn import create_model
from segmentation.dataset import TumorSliceDataset
from tqdm import tqdm

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc='Training')
    for images, targets in progress:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        progress.set_postfix({'loss': losses.item()})
    
    return total_loss / len(loader)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    dataset = TumorSliceDataset('data/raw_mri/patient001_old.nii.gz')
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    model = create_model(num_classes=2)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    num_epochs = 5
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {loss:.4f}")
    
    torch.save(model.state_dict(), 'models/tumor_maskrcnn.pth')
    print("✅ Model saved to models/tumor_maskrcnn.pth")

if __name__ == "__main__":
    main()
