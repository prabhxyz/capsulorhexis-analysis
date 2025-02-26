import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import swin3d_t

class PhaseRecognitionModel(nn.Module):
    """
    A simple wrapper around a 3D Swin Transformer from torchvision, for phase classification.
    """

    def __init__(self, num_phases):
        super().__init__()
        # Load a 3D Swin Tiny model (there is also 'swin3d_s', etc. for bigger versions)
        self.backbone = swin3d_t(weights=None)
        # Replace the final classification head
        in_feats = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_feats, num_phases)

    def forward(self, x):
        # x: (B, T, C, H, W) as expected by Swin3D
        # Need to rearrange to (B, C, T, H, W) for the standard model
        x = x.permute(0, 2, 1, 3, 4)
        logits = self.backbone(x)
        return logits
