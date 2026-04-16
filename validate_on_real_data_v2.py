import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))
from models.unet3d import UNet3D

def dice_score(pred_mask, true_mask):
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask)
    if union == 0:
        return 1.0
    return 2.0 * intersection / union

def iou_score(pred_mask, true_mask):
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask) - intersection
    if union == 0:
        return 1.0
    return intersection / union

def find_dataset_files():
    """Find the COVID-19 dataset in possible locations."""
    import os
    
    possible_paths = [
        Path.home() / ".cache" / "kagglehub" / "datasets" / "tawsifurrahman" / "covid19-radiography-database" / "versions" / "5",
        Path.home() / ".cache" / "kagglehub" / "datasets" / "tawsifurrahman" / "covid19-radiography-database" / "versions" / "6",
        Path.home() / "real_data",
        Path.cwd() / "real_data",
        Path("C:/Users/Pranav/real_data"),
    ]
    
    for cache_path in possible_paths:
        if cache_path.exists():
            print(f"Checking: {cache_path}")
            
            # Try different folder structures
            covid_images = list(cache_path.glob("COVID-19/images/*.png")) + \
                          list(cache_path.glob("images/*.png")) + \
                          list(cache_path.glob("*.png"))
            
            covid_masks = list(cache_path.glob("COVID-19/masks/*.png")) + \
                         list(cache_path.glob("masks/*.png")) + \
                         list(cache_path.glob("*_mask.png"))
            
            if covid_images:
                print(f"Found {len(covid_images)} images")
                print(f"Found {len(covid_masks)} masks")
                return covid_images, covid_masks
    
    print("❌ Dataset not found in any location")
    return [], []
def preprocess_image(img_path, target_size=(256, 256)):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_mask(mask_path, target_size=(256, 256)):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    mask = cv2.resize(mask, target_size)
    mask = (mask > 127).astype(np.uint8)
    return mask

def predict_2d(model, image, device):
    input_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_tensor).squeeze().cpu().numpy()
    return pred

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model_path = Path('models/unet3d_synthetic.pth')
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first: python segmentation/train.py")
        return
    
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Find real data
    covid_images, covid_masks = find_dataset_files()
    
    if len(covid_images) == 0:
        print("\n❌ No real data found.")
        return
    
    dice_scores = []
    iou_scores = []
    
    for i, img_path in enumerate(covid_images[:5]):  # Process first 5 images
        # Find corresponding mask
        mask_path = None
        for m in covid_masks:
            if img_path.stem in m.stem or m.stem in img_path.stem:
                mask_path = m
                break
        
        if mask_path is None:
            print(f"⚠️ Mask not found for {img_path.name}")
            continue
        
        print(f"\n📊 Processing {img_path.name}...")
        
        img = preprocess_image(img_path)
        mask_gt = preprocess_mask(mask_path)
        pred = predict_2d(model, img, device)
        mask_pred = (pred > 0.5).astype(np.uint8)
        
        dice = dice_score(mask_pred, mask_gt)
        iou = iou_score(mask_pred, mask_gt)
        dice_scores.append(dice)
        iou_scores.append(iou)
        
        print(f"   Dice Score: {dice:.4f}")
        print(f"   IoU Score: {iou:.4f}")
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Input X-ray')
        axes[0].axis('off')
        
        axes[1].imshow(mask_gt, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(mask_pred, cmap='gray')
        axes[2].set_title(f'Prediction (Dice: {dice:.3f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"validation_result_{i}.png", dpi=150)
        plt.close()
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 VALIDATION SUMMARY")
    print(f"Total samples processed: {len(dice_scores)}")
    if dice_scores:
        print(f"Mean Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
        print(f"Mean IoU Score: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
