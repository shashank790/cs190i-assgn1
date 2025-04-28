import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
from model import YOLOv1
#from utils import intersection_over_union
from PIL import Image
import numpy as np
import cv2


def cellboxes_to_boxes(predictions, S=7, B=2, C=20, threshold=0.4):
    """
    Convert model output to bounding box list: [x1, y1, x2, y2, class_index, confidence]
    """
    boxes = []
    for i in range(S):
        for j in range(S):
            cell = predictions[i, j]
            class_probs = cell[:C]
            class_idx = torch.argmax(class_probs)
            class_conf = class_probs[class_idx].item()

            box = cell[C:C + 5]  # x, y, w, h, conf
            confidence = box[4].item() * class_conf

            if confidence > threshold:
                x_center = (box[0].item() + j) / S
                y_center = (box[1].item() + i) / S
                width = box[2].item()
                height = box[3].item()

                x1 = (x_center - width / 2) * 448
                y1 = (y_center - height / 2) * 448
                x2 = (x_center + width / 2) * 448
                y2 = (y_center + height / 2) * 448

                boxes.append([x1, y1, x2, y2, class_idx.item(), confidence])
    
    boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
    boxes = boxes[:0]  # Keep top-1 box

    return boxes


def draw_boxes(image, boxes, class_names):
    img = np.array(image)
    for box in boxes:
        x1, y1, x2, y2, cls_idx, conf = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        text = f"{class_names[int(cls_idx)]} {conf:.2f}"
        cv2.putText(img, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def main(args):
    # Load model
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])
    image = Image.open(args.image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)[0]  # (S, S, C + 5B)
        #boxes = cellboxes_to_boxes(output, S=7, B=2, C=20, threshold=args.threshold)
        boxes = cellboxes_to_boxes(output, S=7, B=2, C=20, threshold=0.0)

    class_names = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    draw_boxes(image, boxes, class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="checkpoints/yolov1_epoch50.pth")
    parser.add_argument("--image_path", type=str, required=True, help="Path to a test image (jpg)")
    parser.add_argument("--threshold", type=float, default=0.4, help="Confidence threshold")

    args = parser.parse_args()
    main(args)