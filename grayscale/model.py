# model.py
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class R2Plus1DModel(nn.Module):
    """
    3D CNN for fall vs no-fall classification.
    Input shape: (B, C, T, H, W) where C can be 1 (grayscale) or 3 (RGB).
    """
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        in_channels: int = 1
    ):
        super().__init__()

        # Load backbone (standard 3-channel R(2+1)D)
        if pretrained:
            weights = R2Plus1D_18_Weights.KINETICS400_V1
            self.backbone = r2plus1d_18(weights=weights)
        else:
            self.backbone = r2plus1d_18(weights=None)

        # Optionally adapt the first conv to handle grayscale input
        if in_channels == 1:
            self._convert_first_conv_to_grayscale()

        # Replace final FC to match our classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def _convert_first_conv_to_grayscale(self):
        """
        Modify the first Conv3d in the stem to accept 1 input channel.
        If pretrained, we approximate grayscale weights by averaging RGB.
        """
        # In torchvision's r2plus1d_18, first conv is usually backbone.stem[0]
        first_conv = self.backbone.stem[0]

        # If it's already 1-channel, nothing to do
        if first_conv.in_channels == 1:
            return

        # Old weights: (out_channels, 3, kT, kH, kW)
        old_weight = first_conv.weight.data
        # Average along channel dimension to get grayscale
        new_weight = old_weight.mean(dim=1, keepdim=True)  # -> (out, 1, kT, kH, kW)

        # Create new Conv3d with 1 input channel
        new_conv = nn.Conv3d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None),
        )

        # Assign averaged weights (and bias if present)
        new_conv.weight.data = new_weight
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data

        # Replace in the backbone
        self.backbone.stem[0] = new_conv

    def forward(self, x):
        # x: (B, C, T, H, W) with C = 1 for grayscale
        return self.backbone(x)
