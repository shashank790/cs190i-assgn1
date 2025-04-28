import torch
import torch.nn as nn

def intersection_over_union(boxes_preds, boxes_labels):
    """
    Calculates intersection over union (IOU)
    Boxes: [BATCH_SIZE, S, S, 4]
    Format: x, y, w, h
    """
    box1_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
    box1_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
    box1_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
    box1_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2
    box2_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
    box2_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
    box2_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
    box2_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union = box1_area + box2_area - intersection + 1e-6

    return intersection / union


class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')

        self.S = S
        self.B = B
        self.C = C

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        # predictions: [B, S, S, C + 5B]
        # target: same shape

        N = predictions.shape[0]

        # Split predictions into components
        pred_classes = predictions[..., :self.C]
        pred_boxes = predictions[..., self.C:self.C + 5]
        pred_confidence = predictions[..., self.C + 4:self.C + 5]  # Confidence of box 1

        # Get target components
        target_classes = target[..., :self.C]
        target_boxes = target[..., self.C:self.C + 4]
        target_confidence = target[..., self.C + 4:self.C + 5]

        # Find cells where object exists
        #obj_mask = target_confidence > 0  # Object present
        obj_mask = (target_confidence > 0).squeeze(-1)  # ✅ shape: [B, S, S]

        noobj_mask = target_confidence == 0  # No object

        # ==================== #
        #   1. Localization    #
        # ==================== #
        iou = intersection_over_union(pred_boxes[..., :4], target_boxes[..., :4]).unsqueeze(-1)
        box_loss = self.mse(
            pred_boxes[obj_mask][..., :2], target_boxes[obj_mask][..., :2]
        ) + self.mse(
            torch.sqrt(pred_boxes[obj_mask][..., 2:4].clamp(1e-6)),
            torch.sqrt(target_boxes[obj_mask][..., 2:4].clamp(1e-6))
        )

        # ==================== #
        #   2. Confidence Loss #
        # ==================== #
        obj_loss = self.mse(pred_confidence[obj_mask], iou[obj_mask])
        noobj_loss = self.mse(pred_confidence[noobj_mask], target_confidence[noobj_mask])

        # ==================== #
        #   3. Class Loss      #
        # ==================== #
        class_loss = self.mse(pred_classes[obj_mask], target_classes[obj_mask])

        total_loss = (
            self.lambda_coord * box_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + class_loss
        ) / N

        return total_loss