import torch
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from analysis.volume_analysis import compare_tumors
from reporting.generate_report import create_tumor_report

def main():
    print("=== Medical Tumor Analysis Pipeline ===\n")
    
    # Step 1: Compute volumes from existing masks (or run segmentation)
    print("1. Comparing old and new scans...")
    comparison = compare_tumors(
        'data/raw_mri/patient001_old.nii.gz',
        'data/raw_mri/patient001_new.nii.gz'
    )
    
    print(f"   Old volume: {comparison['old_volume_mm3']:.2f} mm³")
    print(f"   New volume: {comparison['new_volume_mm3']:.2f} mm³")
    print(f"   Change: {comparison['change_mm3']:.2f} mm³ ({comparison['percent_change']:.1f}%)")
    print(f"   Status: {comparison['status']}")
    
    # Step 2: Simulate confidence scores (in real use, compute from model)
    confidence = {'mean': 0.92}  # placeholder
    
    # Step 3: Generate report
    print("\n2. Generating PDF report...")
    create_tumor_report(comparison, confidence, 'clinical_report.pdf')
    
    print("\n✅ Pipeline completed. Check clinical_report.pdf")

if __name__ == "__main__":
    main()
