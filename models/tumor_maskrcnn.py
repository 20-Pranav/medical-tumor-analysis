import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def create_model(num_classes=2, pretrained=True):
    """
    Create a Mask R-CNN model.
    num_classes: 2 (background and tumor)
    """
    backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained)
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model

if __name__ == "__main__":
    model = create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
