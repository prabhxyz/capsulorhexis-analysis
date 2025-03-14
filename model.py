import segmentation_models_pytorch as smp

def get_segmentation_model(model_type, num_classes, encoder_name="efficientnet-b0"):
    if model_type.lower() == 'unet':
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    elif model_type.lower() == 'deeplab':
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: ['unet','deeplab'].")
