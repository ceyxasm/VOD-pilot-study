import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.video_files = os.listdir(os.path.join(root_dir, 'videos'))
        
    def __len__(self):
        return len(self.video_files)
        
    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, 'videos', self.video_files[idx])
        video_name = os.path.splitext(self.video_files[idx])[0]
        annotations_path = os.path.join(self.root_dir, self.split, f'{video_name}.json')
        
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # preprocess annotations
        boxes = []
        labels = []
        for annotation in annotations:
            box = annotation['box']
            box = [box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h']]
            boxes.append(box)
            labels.append(annotation['label'])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        sample = {'frames': frames, 'boxes': boxes, 'labels': labels}
        return sample