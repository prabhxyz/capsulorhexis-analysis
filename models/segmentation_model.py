import torch
import torch.nn as nn
import torchvision.models.segmentation as segm_models
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights

class LightweightSegModel(nn.Module):
    """
    DeepLabv3 with MobileNetV3-Large backbone, with (num_classes) channels.
    """
    def __init__(self, num_classes=13, use_pretrained=True, aux_loss=True):
        super().__init__()
        if use_pretrained:
            weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            self.model = segm_models.deeplabv3_mobilenet_v3_large(weights=weights,
                                                                  aux_loss=aux_loss)
        else:
            self.model = segm_models.deeplabv3_mobilenet_v3_large(weights=None,
                                                                  aux_loss=aux_loss)
        # Overwrite final head
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)
