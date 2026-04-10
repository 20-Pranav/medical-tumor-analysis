import numpy as np
import nibabel as nib
from pathlib import Path

def generate_tumor_volume(size=(128,128,128), tumor_radius=12, center=None):
    """Generate a 3D volume with a spherical tumor."""
    vol = np.zeros(size, dtype=np.float32)
    if center is None:
        center = (size[0]//2, size[1]//2, size[2]//2)
    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                d2 = (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2
                if d2 <= tumor_radius**2:
                    vol[x,y,z] = 1.0  # tumor
    return vol

def main():
    # Create output directory
    Path('data/raw_mri').mkdir(parents=True, exist_ok=True)
    
    # Generate two volumes: old (larger tumor) and new (smaller – regression)
    old_vol = generate_tumor_volume(tumor_radius=15)   # 15 mm radius
    new_vol = generate_tumor_volume(tumor_radius=10)   # 10 mm radius
    
    # Save as NIfTI files (assume voxel spacing = 1 mm isotropic)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(old_vol, affine), 'data/raw_mri/patient001_old.nii.gz')
    nib.save(nib.Nifti1Image(new_vol, affine), 'data/raw_mri/patient001_new.nii.gz')
    
    print("✅ Synthetic MRI volumes created:")
    print("   - data/raw_mri/patient001_old.nii.gz (tumor radius 15 mm)")
    print("   - data/raw_mri/patient001_new.nii.gz (tumor radius 10 mm)")

if __name__ == "__main__":
    main()
