import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models.detection as detection
import torchvision.models.detection.rpn as rpn
from boxmask import BoxMask
from dataloader import VideoDataset, DataLoader

data_dir = '../Dataset/vidvrd-dataset'
train_dataset = VideoDataset(data_dir, 'train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_dataset = VideoDataset(data_dir, 'test')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

model = detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=31, pretrained_backbone=True, trainable_backbone_layers=5, rpn_anchor_generator=rpn.AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)), box_detections_per_img=128, box_score_thresh=0.5, box_nms_thresh=0.5, box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5, box_batch_size_per_image=128, box_positive_fraction=0.25, bbox_reg_weights=None)
model.roi_heads.box_predictor = rpn.FastRCNNPredictor(1024, 31)
model.roi_heads.mask_predictor = rpn.MaskRCNNPredictor(256, 256, 31)
model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for i, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = [{k: v.cuda() for k, v in target.items()} for target in targets]
        optimizer.zero_grad()
        loss = model(images, targets)
        loss['loss_classifier'].backward()
        optimizer.step()
        print('Epoch:', epoch, 'Batch:', i, 'Loss:', loss['loss_classifier'].item())
    torch.save(model.state_dict(), 'model.pth')

model.eval()
with torch.no_grad():
    for i, (images, targets) in enumerate(test_loader):
        images = images.cuda()
        targets = [{k: v.cuda() for k, v in target.items()} for target in targets]
        output = model(images)
        print(output)
        print(targets)
        break

boxmask = BoxMask(model)
boxmask.eval()
with torch.no_grad():
    for i, (images, targets) in enumerate(test_loader):
        images = images.cuda()
        targets = [{k: v.cuda() for k, v in target.items()} for target in targets]
        output = boxmask(images)
        print(output)
        print(targets)
        break