import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class AdvancedSegModel(nn.Module):
    """
    Example advanced segmentation model using a timm backbone (tf_efficientnetv2_s)
    and a minimal upsampling head that resembles DeepLab-like behavior.
    """
    def __init__(self, num_classes=13):
        super().__init__()
        encoder_name = "tf_efficientnetv2_s"
        # pretrained encoder, features_only => returns multiple feature maps
        self.encoder = timm.create_model(encoder_name, pretrained=True, features_only=True)

        # We'll just pick the final feature for a minimal "head"
        final_channels = self.encoder.feature_info[-1]["num_chs"]
        self.reduce_conv = nn.Conv2d(final_channels, 256, kernel_size=1)
        self.classifier  = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # x => (B, C, H, W)
        feats = self.encoder(x)  # list of feature maps at different stages
        f = feats[-1]            # last one is typically the deepest layer

        f = self.reduce_conv(f)  # reduce channels
        # upsample back to input size
        f = F.interpolate(f, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        logits = self.classifier(f)

        return {"out": logits}
