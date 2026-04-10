import kagglehub
import nibabel as nib
from pathlib import Path
import shutil
import os

def download_brats_sample():
    # Download latest BraTS dataset (2020)
    path = kagglehub.dataset_download("awsaf49/brats2020-training-data")
    print(f"Dataset downloaded to: {path}")
    
    # Create local data folder
    raw_dir = Path('data/raw_mri')
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy first 5 cases (each case has 4 modalities + segmentation)
    # BrATS files are nii.gz
    # We'll copy only the T1 and segmentation for simplicity
    # Structure: path/.../BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_*/*.nii.gz
    import glob
    all_cases = glob.glob(os.path.join(path, '**/BraTS20_*'), recursive=True)
    for i, case_dir in enumerate(all_cases[:5]):  # take 5 cases
        t1_file = glob.glob(os.path.join(case_dir, '*t1.nii.gz'))
        seg_file = glob.glob(os.path.join(case_dir, '*seg.nii.gz'))
        if t1_file and seg_file:
            shutil.copy(t1_file[0], raw_dir / f'case{i:03d}_t1.nii.gz')
            shutil.copy(seg_file[0], raw_dir / f'case{i:03d}_seg.nii.gz')
            print(f"Copied case {i}")

if __name__ == "__main__":
    download_brats_sample()
