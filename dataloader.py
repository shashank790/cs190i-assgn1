import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
class_to_idx = {cls: i for i, cls in enumerate(VOC_CLASSES)}

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    # self.image_files = sorted(self.image_files)[:100]  # use first 100 images only

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        label_filename = img_filename.replace(".jpg", ".xml")

        # Load image
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Parse annotation
        label_path = os.path.join(self.label_dir, label_filename)
        #boxes = self.parse_voc_xml(label_path, image.shape[1:], self.S, self.C)
        boxes = self.parse_voc_xml(label_path, self.S, self.C)

        return image, boxes

    def parse_voc_xml(self, xml_path, S, C):
        """
        Convert VOC labels to a target tensor of shape (S, S, C + 5B)
        For each cell, output [C class probs + B boxes (x, y, w, h, confidence)]
        """
        target = torch.zeros((S, S, C + 5 * self.B))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        w, h = int(root.find("size").find("width").text), int(root.find("size").find("height").text)

        for obj in root.iter("object"):
            cls_name = obj.find("name").text.lower().strip()
            if cls_name not in class_to_idx:
                continue
            cls_idx = class_to_idx[cls_name]

            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))

            # Normalize box coords (YOLO format)
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h

            # Determine which grid cell this object belongs to
            i = int(x_center * S)
            j = int(y_center * S)

            if target[j, i, C] == 0:  # Only one object per cell allowed
                target[j, i, cls_idx] = 1  # class one-hot
                box_coords = torch.tensor([x_center * S - i, y_center * S - j, box_w, box_h])
                target[j, i, C:C + 5] = torch.cat([box_coords, torch.tensor([1.0])])  # 1 = objectness/confidence

        return target