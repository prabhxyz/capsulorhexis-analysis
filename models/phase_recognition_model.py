import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class PhaseRecognitionNet(nn.Module):
    """
    Single-frame classification for surgical phase, using MobileNetV3-Large pretrained on ImageNet.
    Loads and infers from a checkpoint: 'phase_recognition.pth'.
    """
    def __init__(self, num_phases=12, use_pretrained=True):
        super().__init__()
        if use_pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
            backbone = mobilenet_v3_large(weights=weights)
        else:
            backbone = mobilenet_v3_large(weights=None)

        # Replace the final classifier to match 'num_phases'
        backbone.classifier[3] = nn.Linear(1280, num_phases)
        self.base = backbone

    def forward(self, x):
        return self.base(x)