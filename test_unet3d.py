import torch
import numpy as np
import matplotlib.pyplot as plt
from models.unet3d import UNet3D

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet3D(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load('models/unet3d_synthetic.pth', map_location=device))
model.eval()

# Create a test volume (tumor radius 12)
vol = np.zeros((64,64,64), dtype=np.float32)
radius = 12
cx, cy, cz = 32, 32, 32
for x in range(64):
    for y in range(64):
        for z in range(64):
            if (x-cx)**2 + (y-cy)**2 + (z-cz)**2 <= radius**2:
                vol[x,y,z] = 1.0
# Add noise
vol_noisy = vol + np.random.normal(0, 0.1, (64,64,64))
vol_noisy = np.clip(vol_noisy, 0, 1)

# Run inference
input_tensor = torch.from_numpy(vol_noisy).float().unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad():
    pred = model(input_tensor).squeeze().cpu().numpy()
pred_mask = (pred > 0.5).astype(np.uint8)

print(f"True tumor voxels: {np.sum(vol)}")
print(f"Predicted tumor voxels: {np.sum(pred_mask)}")

# Visualise middle slice
mid = 32
fig, axes = plt.subplots(1,3, figsize=(12,4))
axes[0].imshow(vol_noisy[:,:,mid], cmap='gray')
axes[0].set_title('Input (noisy)')
axes[1].imshow(vol[:,:,mid], cmap='gray')
axes[1].set_title('Ground truth')
axes[2].imshow(pred_mask[:,:,mid], cmap='gray')
axes[2].set_title('Prediction')
plt.show()
