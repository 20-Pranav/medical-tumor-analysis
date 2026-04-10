import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib

class TumorSliceDataset(Dataset):
    """Load 3D NIfTI volume and return only slices that contain tumor."""
    def __init__(self, volume_path, mask_path=None, slice_axis=2):
        self.volume = nib.load(volume_path).get_fdata().astype(np.float32)
        if mask_path:
            self.mask = nib.load(mask_path).get_fdata().astype(np.float32)
        else:
            self.mask = (self.volume > 0.5).astype(np.float32)
        self.slice_axis = slice_axis
        self.num_slices = self.volume.shape[slice_axis]
        
        # Precompute indices of slices that contain tumor
        self.valid_indices = []
        for idx in range(self.num_slices):
            if slice_axis == 0:
                slice_mask = self.mask[idx, :, :]
            elif slice_axis == 1:
                slice_mask = self.mask[:, idx, :]
            else:
                slice_mask = self.mask[:, :, idx]
            if slice_mask.sum() > 0:
                self.valid_indices.append(idx)
        
        print(f"Found {len(self.valid_indices)} slices with tumor out of {self.num_slices}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        slice_idx = self.valid_indices[idx]
        
        # Extract slice
        if self.slice_axis == 0:
            img = self.volume[slice_idx, :, :]
            mask = self.mask[slice_idx, :, :]
        elif self.slice_axis == 1:
            img = self.volume[:, slice_idx, :]
            mask = self.mask[:, slice_idx, :]
        else:
            img = self.volume[:, :, slice_idx]
            mask = self.mask[:, :, slice_idx]

        # Normalize image to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # (1, H, W)
        
        # Convert mask to binary and to torch tensor
        binary_mask = (mask > 0.5).astype(np.uint8)
        binary_mask_tensor = torch.from_numpy(binary_mask).byte()
        
        pos = torch.where(binary_mask_tensor)
        if len(pos[0]) == 0:
            # No tumor in this slice (should not happen, but fallback)
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks_target = torch.zeros((0, img.shape[0], img.shape[1]), dtype=torch.uint8)
            area = torch.zeros((0,))
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            ymin, xmin = pos[0].min().item(), pos[1].min().item()
            ymax, xmax = pos[0].max().item(), pos[1].max().item()
            
            # Ensure bounding box has positive height and width
            if ymin == ymax:
                ymax = ymin + 1
            if xmin == xmax:
                xmax = xmin + 1
            
            boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
            labels = torch.ones((1,), dtype=torch.int64)
            masks_target = binary_mask_tensor.unsqueeze(0)  # (1, H, W)
            area = (boxes[0,2] - boxes[0,0]) * (boxes[0,3] - boxes[0,1])
            iscrowd = torch.zeros((1,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks_target,
            'image_id': torch.tensor([idx]),
            'area': area.unsqueeze(0) if area.dim() == 0 else area,
            'iscrowd': iscrowd
        }
        return img_tensor, target
