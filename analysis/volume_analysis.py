import numpy as np
import nibabel as nib

def compute_volume_from_mask(mask_nifti_path, voxel_spacing_mm=(1,1,1)):
    """Compute tumor volume in mm³ from a NIfTI mask."""
    img = nib.load(mask_nifti_path)
    data = img.get_fdata()
    voxel_volume = np.prod(voxel_spacing_mm)
    tumor_voxels = np.sum(data > 0.5)
    volume_mm3 = tumor_voxels * voxel_volume
    return volume_mm3

def compare_tumors(old_volume_mm3, new_volume_mm3):
    """Compare two tumor volumes and return change metrics."""
    change = new_volume_mm3 - old_volume_mm3
    percent_change = (change / old_volume_mm3) * 100 if old_volume_mm3 > 0 else 0
    status = "Progression" if change > 0 else "Regression" if change < 0 else "Stable"
    return {
        'old_volume_mm3': old_volume_mm3,
        'new_volume_mm3': new_volume_mm3,
        'change_mm3': change,
        'percent_change': percent_change,
        'status': status
    }
