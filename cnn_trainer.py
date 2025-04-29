import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

#################################################
# CONFIGURATION SETTINGS
#################################################
# Data paths
DATA_DIR = "cnn_dataset"  # Base directory for dataset

# Model parameters
NUM_CLASSES = 2  # 1 FOD class + background
NUM_EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 0.005

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === TRANSFORMS ===
def get_transform():
    """
    Get the transforms for image preprocessing
    
    Returns:
        Composed transforms
    """
    return T.Compose([
        T.ToTensor()
    ])


# === CUSTOM COCO DATASET WRAPPER ===
class CocoDetectionWrapper(CocoDetection):
    """
    Custom wrapper for COCO dataset to format data for Faster R-CNN
    """
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, annotations = super().__getitem__(idx)

        boxes = []
        labels = []

        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue  # Skip invalid boxes

            # Convert to [xmin, ymin, xmax, ymax]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        # Handle empty annotations properly
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self._transforms is not None:
            img = self._transforms(img)

        return img, target


# === LOAD DATA ===
def load_data():
    """
    Load training and validation datasets
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = CocoDetectionWrapper(
        img_folder=os.path.join(DATA_DIR, "train/images"),
        ann_file=os.path.join(DATA_DIR, "train/annotations/instances_train.json"),
        transforms=get_transform()
    )

    val_dataset = CocoDetectionWrapper(
        img_folder=os.path.join(DATA_DIR, "val/images"),
        ann_file=os.path.join(DATA_DIR, "val/annotations/instances_val.json"),
        transforms=get_transform()
    )

    # Custom collate function to handle batch creation properly
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, val_loader


# === BUILD MODEL ===
def get_model(num_classes):
    """
    Create a Faster R-CNN model with a ResNet-50 backbone
    
    Args:
        num_classes: Number of output classes including background
        
    Returns:
        Configured Faster R-CNN model
    """
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


# === TRAIN FUNCTION ===
def train(model, train_loader):
    """
    Train the Faster R-CNN model
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
    """
    # Move model to the appropriate device
    model.to(DEVICE)
    model.train()

    # Set up optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        epoch_loss = 0.0
        for images, targets in tqdm(train_loader):
            # Move data to the appropriate device
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # Print epoch results
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")
        
        # Update learning rate
        lr_scheduler.step()

    # Save the trained model
    print("\nTraining complete. Saving model...")
    torch.save(model.state_dict(), "fasterrcnn_fod.pth")


def main():
    """
    Main function to run the training pipeline
    """
    print(f"Using device: {DEVICE}")
    
    print("Loading data...")
    train_loader, val_loader = load_data()

    print("Creating model...")
    model = get_model(NUM_CLASSES)

    print("Starting training...")
    train(model, train_loader)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
