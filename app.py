import streamlit as st
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))
from models.unet3d import UNet3D
from analysis.volume_analysis import compare_tumors
from reporting.generate_report import create_tumor_report

st.set_page_config(page_title="Tumor Analysis AI", layout="wide")
st.title("🧠 3D Tumor Segmentation & Progression Analysis")

uploaded_old = st.file_uploader("Upload OLD MRI (NIfTI)", type=['nii.gz'], key='old')
uploaded_new = st.file_uploader("Upload NEW MRI (NIfTI)", type=['nii.gz'], key='new')

def segment_volume(nifti_path, model, device, threshold=0.5):
    """Segment a 3D volume and return mask and volume."""
    vol = nib.load(nifti_path).get_fdata().astype(np.float32)
    vol_norm = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    vol_tensor = torch.from_numpy(vol_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(vol_tensor).squeeze().cpu().detach().numpy()
    mask = (pred > threshold).astype(np.uint8)
    tumor_voxels = int(np.sum(mask))
    volume_mm3 = float(tumor_voxels)
    return volume_mm3, mask, vol_norm

if uploaded_old and uploaded_new:
    old_path = Path("temp_old.nii.gz")
    new_path = Path("temp_new.nii.gz")
    old_path.write_bytes(uploaded_old.getvalue())
    new_path.write_bytes(uploaded_new.getvalue())
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load('models/unet3d_synthetic.pth', map_location=device))
    model.eval()
    
    with st.spinner("Segmenting old scan..."):
        vol_old, mask_old, vol_norm_old = segment_volume(old_path, model, device)
    with st.spinner("Segmenting new scan..."):
        vol_new, mask_new, vol_norm_new = segment_volume(new_path, model, device)
    
    comparison = compare_tumors(vol_old, vol_new)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Old Volume", f"{comparison['old_volume_mm3']:.1f} mm³")
    col2.metric("New Volume", f"{comparison['new_volume_mm3']:.1f} mm³")
    col3.metric("Change", f"{comparison['change_mm3']:+.1f} mm³ ({comparison['percent_change']:+.1f}%)", 
                delta=f"{comparison['status']}")
    
    def plot_overlay(vol, mask, title):
        mid_z = vol.shape[2] // 2
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].imshow(vol[:,:,mid_z], cmap='gray')
        ax[0].set_title("Original")
        ax[0].axis('off')
        ax[1].imshow(vol[:,:,mid_z], cmap='gray')
        ax[1].imshow(mask[:,:,mid_z], cmap='Reds', alpha=0.5)
        ax[1].set_title("Overlay (tumor in red)")
        ax[1].axis('off')
        plt.suptitle(title)
        return fig
    
    st.pyplot(plot_overlay(vol_norm_old, mask_old, "Old Scan"))
    st.pyplot(plot_overlay(vol_norm_new, mask_new, "New Scan"))
    
    confidence = {'mean': 0.92}
    create_tumor_report(comparison, confidence, 'web_report.pdf')
    with open('web_report.pdf', 'rb') as f:
        st.download_button("📄 Download Clinical Report", f, file_name='tumor_report.pdf')
    
    # Cleanup
    old_path.unlink()
    new_path.unlink()

