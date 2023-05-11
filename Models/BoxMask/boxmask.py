import torch.nn as nn
import torch.nn.functional as F

class BoxMask(nn.Module):
    def __init__(self, model, num_classes, num_frames, box_loss_weight=1.0, mask_loss_weight=0.5):
        super(BoxMask, self).__init__()

        self.model = model
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.box_loss_weight = box_loss_weight
        self.mask_loss_weight = mask_loss_weight

        self.box_loss_fn = nn.SmoothL1Loss(reduction='sum')
        self.mask_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

        # Define the box and mask prediction heads
        self.box_head = nn.Linear(model.roi_heads.box_predictor.cls_score.in_features, num_classes * 4)
        self.mask_head = nn.Conv2d(model.roi_heads.mask_predictor.conv5_mask.in_channels, num_classes, kernel_size=1)

        # Initialize the box and mask prediction heads
        nn.init.normal_(self.box_head.weight, std=0.01)
        nn.init.normal_(self.mask_head.weight, std=0.01)
        nn.init.constant_(self.box_head.bias, 0)
        nn.init.constant_(self.mask_head.bias, 0)

    def forward(self, images, targets=None):
        if self.training:
            # Unpack the targets
            boxes = [target["boxes"] for target in targets]
            labels = [target["labels"] for target in targets]
            masks = [target["masks"] for target in targets]

        features = self.model.backbone(images)
        proposals, proposal_losses = self.model.rpn(images, features, targets)

        features = [self.model.roi_heads._shared_roi_transform(feature, proposals) for feature in features]

        # Predict the class scores, box offsets, and mask logits
        box_features = [self.model.roi_heads.box_roi_pool(feature, proposals) for feature in features]
        box_features = self.model.roi_heads.box_head(box_features)
        box_features = self.model.roi_heads.box_predictor.cls_score(box_features)
        box_features = self.box_head(box_features)
        box_features = box_features.view(-1, self.num_classes, 4)

        mask_features = [self.model.roi_heads.mask_roi_pool(feature, proposals) for feature in features]
        mask_features = self.model.roi_heads.mask_head(mask_features)
        mask_features = F.interpolate(mask_features, size=(self.num_frames, ) + images.shape[-2:])
        mask_features = self.mask_head(mask_features)
        mask_features = mask_features.view(-1, self.num_classes, self.num_frames, images.shape[-2], images.shape[-1])

        loss_box = 0
        loss_mask = 0
        if self.training:
            for box_feature, mask_feature, box, label, mask in zip(box_features, mask_features, boxes, labels, masks):
                box_feature = box_feature.view(-1, 4)
                box = box.view(-1, 4)
                label = label.view(-1)
                mask = mask.view(-1)
                loss_box += self.box_loss_fn(box_feature, box, reduction='sum')

                mask_feature = mask_feature.view(-1)
                mask = mask.float()
                loss_mask += self.mask_loss_fn(mask_feature, mask, reduction='sum')

            loss_box /= len(targets)
            loss_mask /= len(targets)

            loss = loss_box * self.box_loss_weight + loss_mask * self.mask_loss_weight
            return loss, {}
        else:
            return box_features, mask_features