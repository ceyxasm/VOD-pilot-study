# Importing the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MaskRCNN(nn.Module):
    def __init__(self, num_classes, num_frames, pretrained=False):
        super(MaskRCNN, self).__init__()

        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)

        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            self.model.roi_heads.box_predictor.cls_score.in_features, num_classes * 4)
        self.model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            self.model.roi_heads.mask_predictor.conv5_mask.in_channels, num_classes, num_frames)

        nn.init.normal_(self.model.roi_heads.box_predictor.cls_score.weight, std=0.01)
        nn.init.normal_(self.model.roi_heads.mask_predictor.conv5_mask.weight, std=0.01)
        nn.init.constant_(self.model.roi_heads.box_predictor.cls_score.bias, 0)
        nn.init.constant_(self.model.roi_heads.mask_predictor.conv5_mask.bias, 0)

        self.model.roi_heads.box_predictor.num_classes = num_classes
        self.model.roi_heads.mask_predictor.num_classes = num_classes

        self.model.roi_heads.mask_predictor.num_frames = num_frames

    def forward(self, images, targets=None):
        return self.model(images, targets)