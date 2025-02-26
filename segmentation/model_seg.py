import torch
import torchvision
import torch.nn as nn
import torchvision.transforms.functional as F

def get_mask_rcnn_model(num_classes):
    """
    Returns a Mask R-CNN model with a ResNet50-FPN backbone.
    num_classes includes background.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    # now get num of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)
    return model
