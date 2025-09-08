
# model.py
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class R2Plus1DModel(nn.Module):
    """
    3D CNN for fall vs no-fall classification.
    Input shape: (B, C, T, H, W)
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = R2Plus1D_18_Weights.KINETICS400_V1
            self.backbone = r2plus1d_18(weights=weights)
        else:
            self.backbone = r2plus1d_18(weights=None)
        # Replace final FC to match our classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: (B, C, T, H, W)
        return self.backbone(x)
