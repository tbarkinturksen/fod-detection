import os
import yaml
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import cv2
import glob
from tqdm import tqdm
import albumentations as A
import argparse
from pathlib import Path

#################################################
# CONFIGURATION SETTINGS
#################################################
# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class Config:
    """Configuration class for YOLO training settings"""
    # Data paths
    DATA_YAML = "data/yolo/data.yaml"  # Path to your data.yaml file
    OUTPUT_DIR = "runs/training"  # Output directory for training results
    PRETRAINED_WEIGHTS = "yolov8x.pt"  # Pre-trained weights

    # Model options
    MODEL_SIZE = "x"  # n, s, m, l, x (nano, small, medium, large, xlarge)
    IMAGE_SIZE = 640  # Training image size (higher helps with small objects)

    # Training options
    EPOCHS = 30  # Number of training epochs
    BATCH_SIZE = 4  # Batch size
    PATIENCE = 20  # Early stopping patience

    # Augmentation intensity (0-1)
    AUG_INTENSITY = 0.7  # Higher means more aggressive augmentation

    # Mixed precision training
    MIXED_PRECISION = True  # Use mixed precision for faster training

    # GPU settings
    DEVICE = 0  # GPU device (or 'cpu')
    WORKERS = 4  # Number of workers for data loading

    # Small object detection specifics
    MOSAIC_SCALE = (0.5, 1.5)  # Mosaic scale range - wider for more size variation
    ENABLE_MOSAICS = False  # Enable mosaic augmentation
    ENABLE_MIXUP = False  # Enable mixup augmentation
    ENABLE_COPY_PASTE = True  # Enable copy-paste augmentation for small objects

    # Custom anchors for small objects
    USE_CUSTOM_ANCHORS = True  # Use custom anchors optimized for small objects

    # Validation options
    VAL_INTERVAL = 5  # Validate every N epochs

    # Early stopping
    EARLY_STOPPING = True  # Enable early stopping

    # Focal Loss parameters (helps with class imbalance and small objects)
    FL_GAMMA = 1.5  # Focal Loss gamma (higher focuses more on hard examples)

    # Custom hyperparameters for small object detection
    CUSTOM_HYPS = {
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate factor
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # Weight decay
        'warmup_epochs': 3.0,  # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup momentum
        'box': 7.5,  # Box loss gain (higher for small objects)
        'cls': 0.5,  # Classification loss gain
        'hsv_h': 0.015,  # Image HSV-Hue augmentation
        'hsv_s': 0.7,  # Image HSV-Saturation augmentation
        'hsv_v': 0.4,  # Image HSV-Value augmentation
        'degrees': 0.0,  # Image rotation (+/- deg)
        'translate': 0.1,  # Image translation (+/- fraction)
        'scale': 0.5,  # Image scale (+/- gain)
        'shear': 0.0,  # Image shear (+/- deg)
        'perspective': 0.0,  # Image perspective (+/- fraction)
        'flipud': 0.0,  # Image flip up-down probability
        'fliplr': 0.5,  # Image flip left-right probability
        'mosaic': 1.0 if ENABLE_MOSAICS else 0.0,  # Mosaic probability
        'mixup': 0.1 if ENABLE_MIXUP else 0.0,  # Mixup probability
        'copy_paste': 0.1 if ENABLE_COPY_PASTE else 0.0,  # Copy-paste probability
        'auto_augment': 'randaugment'  # Auto augmentation policy
    }


def prepare_dataset(config):
    """
    Analyze dataset and prepare it for training
    
    Args:
        config: Configuration object
        
    Returns:
        Data dictionary loaded from YAML
    """
    # Load data.yaml
    with open(config.DATA_YAML, 'r') as f:
        data_dict = yaml.safe_load(f)

    # Get paths
    yaml_dir = os.path.dirname(os.path.abspath(config.DATA_YAML))

    # Resolve relative paths
    if not os.path.isabs(data_dict['train']):
        data_dict['train'] = os.path.join(yaml_dir, data_dict['train'])
    if not os.path.isabs(data_dict['val']):
        data_dict['val'] = os.path.join(yaml_dir, data_dict['val'])

    # Analyze dataset to understand object sizes
    analyze_dataset(data_dict, config)

    return data_dict


def analyze_dataset(data_dict, config):
    """
    Analyze the dataset to understand object distributions and sizes
    
    Args:
        data_dict: Data dictionary loaded from YAML
        config: Configuration object
    """
    print("\n📊 Analyzing dataset to optimize for small object detection...")

    # Get train image and label paths
    train_img_dir = data_dict['train']
    label_dir = train_img_dir.replace('images', 'labels')

    # Get image files
    img_files = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir)
                 if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Initialize counters
    total_objects = 0
    class_counts = {i: 0 for i in range(len(data_dict['names']))}
    size_distributions = []

    # Sample random images for analysis (for efficiency)
    sample_size = min(100, len(img_files))
    sampled_files = random.sample(img_files, sample_size)

    for img_file in tqdm(sampled_files, desc="Analyzing images"):
        # Get corresponding label file
        label_file = os.path.join(label_dir, os.path.basename(img_file).replace(
            os.path.splitext(img_file)[1], '.txt'))

        if not os.path.exists(label_file):
            continue

        # Get image dimensions
        img = Image.open(img_file)
        img_width, img_height = img.size

        # Read labels
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class_id, x_center, y_center, width, height (normalized)
                class_id = int(parts[0])
                width = float(parts[3])  # Normalized width
                height = float(parts[4])  # Normalized height

                # Convert to pixels (diagonal size)
                obj_width = width * img_width
                obj_height = height * img_height
                obj_size = (obj_width ** 2 + obj_height ** 2) ** 0.5  # Diagonal

                # Update counters
                class_counts[class_id] += 1
                total_objects += 1
                size_distributions.append((obj_width, obj_height, obj_size, class_id))

    # Calculate statistics
    if total_objects > 0:
        # Sort objects by size
        size_distributions.sort(key=lambda x: x[2])

        # Calculate size percentiles
        small_threshold = size_distributions[int(len(size_distributions) * 0.25)][2]
        medium_threshold = size_distributions[int(len(size_distributions) * 0.75)][2]

        # Count objects by size
        small_objects = sum(1 for _, _, size, _ in size_distributions if size <= small_threshold)
        medium_objects = sum(1 for _, _, size, _ in size_distributions
                             if small_threshold < size <= medium_threshold)
        large_objects = sum(1 for _, _, size, _ in size_distributions if size > medium_threshold)

        # Print stats
        print(f"\n📏 Object Size Analysis (based on {total_objects} objects from {sample_size} images):")
        print(f"  - Small objects:  {small_objects} ({small_objects / total_objects * 100:.1f}%)")
        print(f"  - Medium objects: {medium_objects} ({medium_objects / total_objects * 100:.1f}%)")
        print(f"  - Large objects:  {large_objects} ({large_objects / total_objects * 100:.1f}%)")

        # Print class distribution
        print("\n📋 Class Distribution:")
        for class_id, count in class_counts.items():
            if count > 0:
                print(f"  - {data_dict['names'][class_id]}: {count} ({count / total_objects * 100:.1f}%)")

        # Generate custom anchors if needed and many small objects
        if config.USE_CUSTOM_ANCHORS and small_objects / total_objects > 0.3:
            # Extract width and height ratios
            wh_ratios = [(w / img_width, h / img_height) for w, h, _, _ in size_distributions]
            print("\n⚓ Generating custom anchors optimized for this dataset's object sizes...")

            # Get width and height of each size category
            small_wh = [(w, h) for w, h, s, _ in size_distributions if s <= small_threshold]
            medium_wh = [(w, h) for w, h, s, _ in size_distributions
                         if small_threshold < s <= medium_threshold]
            large_wh = [(w, h) for w, h, s, _ in size_distributions if s > medium_threshold]

            # Calculate means for each group
            if small_wh:
                small_mean_w = sum(w for w, _ in small_wh) / len(small_wh)
                small_mean_h = sum(h for _, h in small_wh) / len(small_wh)
                print(f"  - Small object avg size: {small_mean_w:.1f}x{small_mean_h:.1f} pixels")

            if medium_wh:
                medium_mean_w = sum(w for w, _ in medium_wh) / len(medium_wh)
                medium_mean_h = sum(h for _, h in medium_wh) / len(medium_wh)
                print(f"  - Medium object avg size: {medium_mean_w:.1f}x{medium_mean_h:.1f} pixels")

            if large_wh:
                large_mean_w = sum(w for w, _ in large_wh) / len(large_wh)
                large_mean_h = sum(h for _, h in large_wh) / len(large_wh)
                print(f"  - Large object avg size: {large_mean_w:.1f}x{large_mean_h:.1f} pixels")

    # Calculate optimal image size
    avg_img_width = sum(Image.open(f).width for f in sampled_files) / len(sampled_files)
    avg_img_height = sum(Image.open(f).height for f in sampled_files) / len(sampled_files)

    print(f"\n🖼️ Average image dimensions: {avg_img_width:.1f}x{avg_img_height:.1f}")
    print(f"   Training at {config.IMAGE_SIZE}x{config.IMAGE_SIZE} resolution")

    # Give recommendations
    if total_objects > 0:
        small_object_percentage = small_objects / total_objects * 100
        if small_object_percentage > 50:
            print("\n⚠️ Your dataset has many small objects (>50%). Recommendations:")
            print("   - Consider increasing IMAGE_SIZE to 1280 or higher")
            print("   - Enable mosaic and mixup augmentations")
            print("   - Use a larger model like YOLOv8x or YOLOv8l")
            print("   - Consider longer training with more epochs")


def create_custom_yaml(data_dict, output_path):
    """
    Create a custom YAML file with optimized settings
    
    Args:
        data_dict: Data dictionary with dataset information
        output_path: Path to save the custom YAML file
        
    Returns:
        Path to the created YAML file
    """
    # Save a copy of the original data YAML with absolute paths
    with open(output_path, 'w') as f:
        yaml.dump(data_dict, f, default_flow_style=False)

    print(f"✅ Created custom data YAML at: {output_path}")
    return output_path


def get_optimal_model_settings(config, data_dict):
    """
    Determine optimal model settings based on the dataset analysis
    
    Args:
        config: Configuration object
        data_dict: Data dictionary with dataset information
        
    Returns:
        Dictionary of model arguments
    """
    # Get number of classes
    num_classes = len(data_dict['names'])

    # Create model args dictionary with compatible arguments
    model_args = {
        'data': config.DATA_YAML,
        'epochs': config.EPOCHS,
        'patience': config.PATIENCE,
        'batch': config.BATCH_SIZE,
        'imgsz': config.IMAGE_SIZE,
        'device': config.DEVICE,
        'workers': config.WORKERS,
        'val': True,  # Instead of 'noval'
        'project': config.OUTPUT_DIR,
        'name': f'yolov8_{config.MODEL_SIZE}_small_fod_detector',
        'exist_ok': True,
        'pretrained': True if config.PRETRAINED_WEIGHTS else False,
        'optimizer': 'AdamW',  # Better for small objects
        'amp': config.MIXED_PRECISION,  # Mixed precision training
        'rect': False,  # Don't use rectangular training (better for small objects)
        'cos_lr': True,  # Use cosine learning rate
        'close_mosaic': 10,  # Disable mosaic in last 10 epochs for stability
        'cache': 'ram' if torch.cuda.is_available() else False,  # Cache images in RAM
        'plots': True,  # Generate plots
        'save': True,  # Save checkpoints
        'save_period': 10,  # Save checkpoints every 10 epochs
    }

    # Add hyperparameters directly through cfg
    for k, v in config.CUSTOM_HYPS.items():
        model_args[k] = v

    return model_args


def visualize_annotations(data_dict, num_samples=3):
    """
    Visualize sample annotations to verify dataset
    
    Args:
        data_dict: Data dictionary with dataset information
        num_samples: Number of samples to visualize
    """
    print("\n🔍 Visualizing random samples from the dataset...")

    # Get train image and label paths
    train_img_dir = data_dict['train']
    label_dir = train_img_dir.replace('images', 'labels')

    # Get all image files
    img_files = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir)
                 if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Sample random images
    sampled_files = random.sample(img_files, min(num_samples, len(img_files)))

    # Create a figure
    fig, axs = plt.subplots(1, len(sampled_files), figsize=(15, 5))

    # Ensure axs is always a list
    if len(sampled_files) == 1:
        axs = [axs]

    for i, img_file in enumerate(sampled_files):
        # Get corresponding label file
        label_file = os.path.join(label_dir, os.path.basename(img_file).replace(
            os.path.splitext(img_file)[1], '.txt'))

        # Read image
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # Display image
        axs[i].imshow(img)
        axs[i].set_title(f"Sample {i + 1}")

        # If label file exists, draw bounding boxes
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height

                    # Calculate box coordinates
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)

                    # Draw rectangle
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor='r', facecolor='none')
                    axs[i].add_patch(rect)

                    # Add label
                    class_name = data_dict['names'][class_id]
                    axs[i].text(x1, y1 - 5, class_name, color='r', fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.7))

        # Hide axes
        axs[i].axis('off')

    plt.tight_layout()

    # Save and display figure
    os.makedirs('analysis', exist_ok=True)
    plt.savefig('analysis/sample_annotations.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Saved sample visualizations to 'analysis/sample_annotations.png'")


def create_validation_images(data_dict, config, num_samples=4):
    """
    Create validation visualizations with both training resolution and target resolution
    
    Args:
        data_dict: Data dictionary with dataset information
        config: Configuration object
        num_samples: Number of samples to visualize
    """
    print("\n📸 Creating validation visualizations for resolution comparison...")

    # Get validation image and label paths
    val_img_dir = data_dict['val']
    label_dir = val_img_dir.replace('images', 'labels')

    # Get image files
    img_files = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir)
                 if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Sample random images
    sampled_files = random.sample(img_files, min(num_samples, len(img_files)))

    # Create output directory
    os.makedirs('analysis/resolution_comparison', exist_ok=True)

    # Load a YOLO model without training for visualization
    model = YOLO(f'yolov8{config.MODEL_SIZE}.pt')

    for img_file in sampled_files:
        img_name = os.path.basename(img_file)
        original_img = cv2.imread(img_file)

        # Get corresponding label file
        label_file = os.path.join(label_dir, os.path.basename(img_file).replace(
            os.path.splitext(img_file)[1], '.txt'))

        # Create comparison image
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Original image with ground truth
        axs[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"Original ({original_img.shape[1]}x{original_img.shape[0]})")

        # If label file exists, draw ground truth boxes
        if os.path.exists(label_file):
            height, width = original_img.shape[:2]
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height

                    # Calculate box coordinates
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)

                    # Draw rectangle
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor='g', facecolor='none')
                    axs[0].add_patch(rect)

                    # Add label
                    class_name = data_dict['names'][class_id]
                    axs[0].text(x1, y1 - 5, class_name, color='g', fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.7))

        # 512x512 resolution (typical training size)
        resized_512 = cv2.resize(original_img, (512, 512))
        axs[1].imshow(cv2.cvtColor(resized_512, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"Resized to 512x512")

        # Target resolution
        resized_target = cv2.resize(original_img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        axs[2].imshow(cv2.cvtColor(resized_target, cv2.COLOR_BGR2RGB))
        axs[2].set_title(f"Resized to {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")

        # Hide axes
        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'analysis/resolution_comparison/{img_name}', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✅ Created resolution comparison visualizations in 'analysis/resolution_comparison/'")


def augmentation_preview(data_dict, config):
    """
    Create visualizations of augmentations to be used during training
    
    Args:
        data_dict: Data dictionary with dataset information
        config: Configuration object
    """
    print("\n🔄 Creating augmentation previews...")

    try:
        # Get train image and label paths
        train_path = os.path.join(data_dict['train'])
        label_path = train_path.replace('images', 'labels')

        # Get image files
        img_files = [f for f in glob.glob(os.path.join(train_path, '**', '*.jpg'), recursive=True)]
        img_files += [f for f in glob.glob(os.path.join(train_path, '**', '*.jpeg'), recursive=True)]
        img_files += [f for f in glob.glob(os.path.join(train_path, '**', '*.png'), recursive=True)]

        if not img_files:
            print("⚠️ No images found for augmentation preview. Skipping.")
            return

        # Sample a random image
        img_file = random.choice(img_files)
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # Get corresponding label file
        rel_path = os.path.relpath(img_file, train_path)
        label_file = os.path.join(label_path, os.path.splitext(rel_path)[0] + '.txt')

        # Load labels
        boxes = []
        class_ids = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_ids.append(class_id)

                    # Parse YOLO format (class_id, x_center, y_center, width, height)
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    # Convert to albumentations format [x_min, y_min, x_max, y_max]
                    x_min = max(0, x_center - w / 2)
                    y_min = max(0, y_center - h / 2)
                    x_max = min(1, x_center + w / 2)
                    y_max = min(1, y_center + h / 2)

                    boxes.append([x_min, y_min, x_max, y_max])

        # Define simplified augmentations that are compatible with current Albumentations API
        aug_intensity = config.AUG_INTENSITY
        augmentations = [
            ("Original", None),

            ("Horizontal Flip", A.Compose([
                A.HorizontalFlip(p=1.0)
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))),

            ("Color Jitter", A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.2, p=1.0)
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))),

            ("Blur & Noise", A.Compose([
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=0.5),
                    A.MotionBlur(blur_limit=3, p=0.5),
                ], p=0.5),
                A.GaussNoise(p=0.5)
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))),

            ("Affine Transform", A.Compose([
                A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), p=1.0)
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))),

            ("Dropout", A.Compose([
                A.CoarseDropout(p=1.0)
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))),
        ]

        # Create a figure
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()

        # Process each augmentation
        for i, (aug_name, aug) in enumerate(augmentations):
            img_aug = img.copy()
            bboxes_aug = boxes.copy()
            class_ids_aug = class_ids.copy()

            # Skip if no boxes
            if not bboxes_aug:
                axs[i].imshow(img_aug)
                axs[i].set_title(aug_name)
                axs[i].axis('off')
                continue

            if aug:
                try:
                    # Apply augmentation
                    transformed = aug(image=img_aug, bboxes=bboxes_aug, class_labels=class_ids_aug)
                    img_aug = transformed['image']
                    bboxes_aug = transformed['bboxes']
                    class_ids_aug = transformed['class_labels']
                except Exception as e:
                    print(f"⚠️ Error in {aug_name} augmentation: {str(e)}")
                    # Just display the original image in case of error

            # Display image
            axs[i].imshow(img_aug)
            axs[i].set_title(aug_name)

            # Draw bounding boxes
            for j, bbox in enumerate(bboxes_aug):
                if j >= len(class_ids_aug):
                    continue

                # Get coordinates in pixel space
                x_min, y_min, x_max, y_max = bbox
                x_min = int(x_min * width)
                y_min = int(y_min * height)
                x_max = int(x_max * width)
                y_max = int(y_max * height)

                # Draw rectangle
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='r', facecolor='none')
                axs[i].add_patch(rect)

                # Add label
                class_id = class_ids_aug[j]
                if class_id < len(data_dict['names']):
                    class_name = data_dict['names'][class_id]
                    axs[i].text(x_min, y_min - 5, class_name, color='r', fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.7))

            # Hide axes
            axs[i].axis('off')

        plt.tight_layout()

        # Save figure
        os.makedirs('analysis', exist_ok=True)
        plt.savefig('analysis/augmentation_preview.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("✅ Created augmentation preview at 'analysis/augmentation_preview.png'")

    except Exception as e:
        print(f"⚠️ Error creating augmentation preview: {str(e)}")
        print("Skipping augmentation preview.")


def train_model(model_args, config):
    """
    Train the YOLOv8 model with the specified arguments
    
    Args:
        model_args: Dictionary of model arguments
        config: Configuration object
        
    Returns:
        Trained YOLO model
    """
    print("\n🚀 Starting model training...")

    # Get model path
    model_size = config.MODEL_SIZE
    model_path = f'yolov8{model_size}.pt'

    # Initialize model
    model = YOLO(model_path)

    # Train model
    model.train(**model_args)

    return model


def validate_model(model, data_yaml, image_size=640):
    """
    Validate the trained model on the validation set
    
    Args:
        model: Trained YOLO model
        data_yaml: Path to data YAML file
        image_size: Image size for validation
        
    Returns:
        Validation metrics
    """
    print("\n📊 Validating trained model...")

    # Run validation
    metrics = model.val(data=data_yaml, imgsz=image_size)

    return metrics


def export_model(model, format='onnx'):
    """
    Export the trained model to the specified format
    
    Args:
        model: Trained YOLO model
        format: Export format (e.g., 'onnx', 'tflite', 'torchscript')
    """
    print(f"\n📦 Exporting model to {format.upper()} format...")

    # Export model
    model.export(format=format, imgsz=640)


def main():
    """
    Main function to train YOLOv8 for small FOD detection
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train YOLOv8 for small FOD detection')
    parser.add_argument('--data', type=str, default='data/yolo/data.yaml',
                        help='Path to data.yaml file')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--model-size', type=str, default='x',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size')
    parser.add_argument('--pretrained', type=str, default='yolov8x.pt',
                        help='Pretrained weights path')
    parser.add_argument('--output-dir', type=str, default='runs/training',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (e.g., 0, 0,1,2,3, cpu)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers')

    args = parser.parse_args()

    # Update config with command-line arguments
    config = Config()
    config.DATA_YAML = args.data
    config.IMAGE_SIZE = args.img_size
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.MODEL_SIZE = args.model_size
    config.PRETRAINED_WEIGHTS = args.pretrained
    config.OUTPUT_DIR = args.output_dir
    config.DEVICE = args.device
    config.WORKERS = args.workers

    # Print configuration
    print("\n" + "=" * 50)
    print("   Small FOD Detection Training Pipeline")
    print("=" * 50)
    print(f"Data config: {config.DATA_YAML}")
    print(f"Model: YOLOv8{config.MODEL_SIZE}")
    print(f"Training resolution: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Device: {config.DEVICE}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print("=" * 50 + "\n")

    # Check if CUDA is available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"🔥 Found {num_gpus} GPU(s)")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"  - GPU {i}: {gpu_name}")
    else:
        print("⚠️ No GPUs found, using CPU (training will be slow)")

    # Prepare dataset
    data_dict = prepare_dataset(config)

    # Create custom YAML file
    custom_yaml_path = os.path.join(config.OUTPUT_DIR, 'custom_data.yaml')
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    custom_yaml_path = create_custom_yaml(data_dict, custom_yaml_path)

    # Visualize sample annotations
    visualize_annotations(data_dict)

    # Create validation resolution comparison
    create_validation_images(data_dict, config)

    # Create augmentation preview
    augmentation_preview(data_dict, config)

    # Get optimal model settings
    model_args = get_optimal_model_settings(config, data_dict)
    model_args['data'] = custom_yaml_path  # Use the custom YAML

    # Print training configuration summary
    print("\n📋 Training Configuration Summary:")
    print(f"  - Model: YOLOv8{config.MODEL_SIZE}")
    print(f"  - Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Epochs: {config.EPOCHS}")
    print(f"  - Augmentation intensity: {config.AUG_INTENSITY}")
    print(f"  - Mixed precision: {'Enabled' if config.MIXED_PRECISION else 'Disabled'}")
    print(f"  - Mosaic augmentation: {'Enabled' if config.ENABLE_MOSAICS else 'Disabled'}")
    print(f"  - Mixup augmentation: {'Enabled' if config.ENABLE_MIXUP else 'Disabled'}")
    print(f"  - Copy-paste augmentation: {'Enabled' if config.ENABLE_COPY_PASTE else 'Disabled'}")
    print(f"  - Custom anchors: {'Enabled' if config.USE_CUSTOM_ANCHORS else 'Disabled'}")
    print(f"  - Early stopping: {'Enabled' if config.EARLY_STOPPING else 'Disabled'}")

    # Ask for confirmation before starting training
    if input("\n🚀 Ready to start training. Continue? (y/n): ").lower() != 'y':
        print("Training cancelled.")
        return

    # Train model
    model = train_model(model_args, config)

    # Validate model
    metrics = validate_model(model, custom_yaml_path, config.IMAGE_SIZE)

    # Export model
    export_model(model)

    print("\n✅ Training completed!")
    print(f"  - Best model saved at: {os.path.join(config.OUTPUT_DIR, 'yolov8_' + config.MODEL_SIZE + '_small_fod_detector', 'weights', 'best.pt')}")
    print(f"  - Final mAP50: {metrics.box.map50:.3f}")
    print(f"  - Final mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
