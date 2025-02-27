import torch
import torch.nn as nn
import timm

class PhaseRecognitionNet(nn.Module):
    """
    Single-frame classification for surgical phase, using a more advanced
    architecture from timm (e.g. convnext_tiny).
    """
    def __init__(self, num_phases=12, use_pretrained=True):
        super().__init__()
        model_name = "convnext_tiny"
        if use_pretrained:
            self.base = timm.create_model(model_name, pretrained=True, num_classes=num_phases)
        else:
            self.base = timm.create_model(model_name, pretrained=False, num_classes=num_phases)

    def forward(self, x):
        return self.base(x)
