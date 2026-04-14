import torch
import nibabel as nib
import numpy as np
from models.unet3d import UNet3D

def dice_score(pred_mask, true_mask):
    """Calculate Dice similarity coefficient."""
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask)
    if union == 0:
        return 1.0
    return 2.0 * intersection / union

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet3D(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load('models/unet3d_synthetic.pth', map_location=device))
model.eval()

print("Evaluating model on test volumes...")
print("-" * 40)

for fname in ['old_large.nii.gz', 'new_small.nii.gz']:
    vol = nib.load(fname).get_fdata().astype(np.float32)
    true_mask = (vol > 0.5).astype(np.uint8)
    
    vol_norm = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    vol_tensor = torch.from_numpy(vol_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(vol_tensor).squeeze().cpu().detach().numpy()
    pred_mask = (pred > 0.5).astype(np.uint8)
    
    dice = dice_score(pred_mask, true_mask)
    print(f"{fname}: Dice Score = {dice:.4f}")
