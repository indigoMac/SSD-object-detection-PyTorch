import torchvision
from torchvision.transforms import functional as F
import torch

# Define class mapping
VOC_CLASSES = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}

# Reverse the dictionary to map indices to class names
VOC_CLASSES_REVERSE = {v: k for k, v in VOC_CLASSES.items()}

# def get_transform(train):
#     transforms = []
#     transforms.append(torchvision.transforms.ToTensor())
#     if train:
#         transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
#     return torchvision.transforms.Compose(transforms)

def get_transform(train):
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    if train:
        transforms.extend([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.RandomRotation(30),
            torchvision.transforms.RandomResizedCrop(size=(300, 300), scale=(0.8, 1.0))
        ])
    return torchvision.transforms.Compose(transforms)

class VOCDataset:
    def __init__(self, root, years, image_set, transform):
        self.datasets = []
        for year in years:
            dataset = torchvision.datasets.VOCDetection(root=root, year=year, image_set=image_set, download=True, transform=transform)
            self.datasets.append(dataset)
        self.transform = transform

    def __getitem__(self, idx):
        dataset_idx = 0
        while idx >= len(self.datasets[dataset_idx]):
            idx -= len(self.datasets[dataset_idx])
            dataset_idx += 1
        img, target = self.datasets[dataset_idx][idx]

        # The image is already transformed to a tensor by the dataset transform
        # Extract bounding boxes and labels from target
        boxes = []
        labels = []
        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            xmin, ymin, xmax, ymax = round(float(bbox['xmin'])), round(float(bbox['ymin'])), round(float(bbox['xmax'])), round(float(bbox['ymax']))

            # Ensure box values are valid
            if xmin >= xmax or ymin >= ymax:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(VOC_CLASSES[obj['name']])  # Map class name to class index

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        return img, target

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
