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

def reconstruct_3d_mask_from_slices(model, volume_path, device='cuda'):
    """
    Run inference on each slice of a 3D volume and reconstruct a 3D mask.
    (Simplified: here we assume we already have ground truth masks.)
    For demonstration, we return the ground truth mask.
    """
    # In a real scenario, you'd run the model on each slice and aggregate.
    # For this demo, we just load the original volume as mask.
    vol = nib.load(volume_path).get_fdata()
    mask = (vol > 0.5).astype(np.uint8)
    return mask

def compare_tumors(old_mask_path, new_mask_path, voxel_spacing=(1,1,1)):
    old_vol = compute_volume_from_mask(old_mask_path, voxel_spacing)
    new_vol = compute_volume_from_mask(new_mask_path, voxel_spacing)
    change = new_vol - old_vol
    percent_change = (change / old_vol) * 100 if old_vol > 0 else 0
    status = "Progression" if change > 0 else "Regression" if change < 0 else "Stable"
    return {
        'old_volume_mm3': old_vol,
        'new_volume_mm3': new_vol,
        'change_mm3': change,
        'percent_change': percent_change,
        'status': status
    }

if __name__ == "__main__":
    # Example: use synthetic volumes as masks
    result = compare_tumors('data/raw_mri/patient001_old.nii.gz', 'data/raw_mri/patient001_new.nii.gz')
    print(result)
