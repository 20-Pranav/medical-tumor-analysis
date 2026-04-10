import nibabel as nib
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def make_volume(radius, filename):
    vol = np.zeros((64,64,64), dtype=np.float32)
    center = 32
    for x in range(64):
        for y in range(64):
            for z in range(64):
                if (x-center)**2 + (y-center)**2 + (z-center)**2 <= radius**2:
                    vol[x,y,z] = 1.0
    # Add same noise pattern (because seed is fixed)
    noise = np.random.normal(0, 0.1, (64,64,64))
    vol = vol + noise
    vol = np.clip(vol, 0, 1)
    nib.save(nib.Nifti1Image(vol, np.eye(4)), filename)
    print(f"Created {filename} with tumor radius {radius}")

make_volume(radius=12, filename="old_large.nii.gz")
make_volume(radius=8,  filename="new_small.nii.gz")
