import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import YOLOv1
from utils import YOLOLoss
from dataloader import YOLODataset 
import argparse
from tqdm import tqdm
import os

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and loader
    train_dataset = YOLODataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        S=args.grid_size,
        B=args.num_boxes,
        C=args.num_classes
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Model and optimizer
    model = YOLOv1(split_size=args.grid_size, num_boxes=args.num_boxes, num_classes=args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = YOLOLoss(S=args.grid_size, B=args.num_boxes, C=args.num_classes)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (images, targets) in enumerate(loop):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}: Total Loss = {total_loss / len(train_loader):.4f}")

        # Optional: Save model
        if args.save_model:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/yolov1_epoch{epoch+1}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data/images", help="Path to training images")
    parser.add_argument("--label_dir", type=str, default="data/labels", help="Path to training labels")
    parser.add_argument("--grid_size", type=int, default=7)
    parser.add_argument("--num_boxes", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_model", action="store_true", help="Save model after each epoch")

    args = parser.parse_args()
    train(args)