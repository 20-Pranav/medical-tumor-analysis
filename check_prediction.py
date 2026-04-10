import torch
import nibabel as nib
import numpy as np
from models.unet3d import UNet3D

device = 'cpu'
model = UNet3D(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load('models/unet3d_synthetic.pth', map_location=device))
model.eval()

for fname in ['old_large.nii.gz', 'new_small.nii.gz']:
    vol = nib.load(fname).get_fdata().astype(np.float32)
    vol_norm = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    vol_tensor = torch.from_numpy(vol_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(vol_tensor).squeeze().cpu().detach().numpy()   # <-- added .detach()
    tumor_voxels = np.sum(pred > 0.5)
    print(f"{fname}: predicted tumor voxels = {tumor_voxels}")
