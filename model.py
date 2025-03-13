import segmentation_models_pytorch as smp
import torch.nn as nn

def get_segmentation_model(num_classes, encoder_name="efficientnet-b0"):
    """
    Returns a U-Net model from segmentation_models_pytorch with the specified encoder.
    num_classes includes background or not (depending on usage).
    If your dataset's mask is: 0=background, 1=class1, 2=class2..., 
    then num_classes should match the highest label + 1.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,  # e.g. background + N relevant categories
        activation=None
    )
    return model
