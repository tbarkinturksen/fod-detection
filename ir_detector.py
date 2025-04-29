import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# ========================
# CONFIGURATION SETTINGS
# ========================
# This script is designed for IR (Infrared) FOD (Foreign Object Debris) detection
# using both YOLO and R-CNN models, with an autoencoder to filter false positives.
INPUT_FOLDER = 'input'  # Input folder for source files
OUTPUT_FOLDER = 'output'  # Output folder for processed files
CROPS_FOLDER = os.path.join(OUTPUT_FOLDER, 'detection_crops_ir')  # Folder for detection crops
ACCEPTED_FOLDER = os.path.join(CROPS_FOLDER, 'accepted')  # Accepted detections
REJECTED_FOLDER = os.path.join(CROPS_FOLDER, 'rejected')  # Rejected detections (false positives)
AUTOENCODER_MODEL_PATH = os.path.join(INPUT_FOLDER, 'ir_autoencoder.pth')  # Path to load autoencoder model

FRAME_SKIP = 10  # Process every nth frame
CONF_THRESHOLD_YOLO = 0.05
CONF_THRESHOLD_RCNN = 0.01
IOU_THRESHOLD = 0.45
ANOMALY_THRESHOLD = 0.025  # Threshold for autoencoder reconstruction error
USE_AUTOENCODER = False  # Set to False to disable autoencoder filtering
YOLO_MODEL_PATH = os.path.join(INPUT_FOLDER, 'yolov8m.pt')  # Using YOLO m model as requested
RCNN_MODEL_PATH = os.path.join(INPUT_FOLDER, 'fasterrcnn_sd_ir_fod.pth')  # Custom IR model
NUM_CLASSES = 2
CLASS_NAMES = ["background", "FOD"]

# Autoencoder settings
AUTOENCODER_CROP_SIZE = 128  # Size to resize detection crops for autoencoder

# GPU Acceleration settings (RTX)
USE_AMP = False  # Use Automatic Mixed Precision
USE_HALF = False  # Use half precision (FP16)
CUDA_MEMORY_FRACTION = 0.8  # Use 80% of available GPU memory

# Debug settings
SAVE_ALL_CROPS = True  # Always save both accepted and rejected crops for debugging
DEBUG_MODE = True  # Print more debug information

# ========================
# DEVICE SETUP
# ========================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set up CUDA optimizations if available
if DEVICE == 'cuda':
    # Configure CUDA for optimal performance
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Set memory allocation strategy
    if CUDA_MEMORY_FRACTION < 1.0:
        torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION)

    # Print GPU information
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in GB
    print(f"✅ Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ Half precision (FP16): {'Enabled' if USE_HALF else 'Disabled'}")
    print(f"✅ Automatic Mixed Precision: {'Enabled' if USE_AMP else 'Disabled'}")
else:
    print(f"✅ Using CPU: {torch.get_num_threads()} threads")
    print("⚠️ GPU acceleration not available. Processing will be slower.")


# ========================
# AUTOENCODER MODEL
# ========================
class IRAutoencoder(nn.Module):
    """
    Autoencoder model for IR image anomaly detection.
    Used to filter out false positives in runway markings.
    """

    def __init__(self):
        super(IRAutoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(True)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32 -> 64x64
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64 -> 128x128
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_autoencoder():
    """
    Load the pre-trained autoencoder model.

    Returns:
        model: The loaded autoencoder model or None if disabled or the model file doesn't exist
    """
    # If autoencoder is disabled, return None
    if not USE_AUTOENCODER:
        print(f"ℹ️ Autoencoder processing is disabled. All detections will be accepted.")
        return None

    if not os.path.exists(AUTOENCODER_MODEL_PATH):
        print(f"⚠️ Pre-trained autoencoder not found: {AUTOENCODER_MODEL_PATH}")
        print("Detection will proceed without false positive filtering.")
        return None

    print(f"Loading pre-trained autoencoder from: {AUTOENCODER_MODEL_PATH}")
    try:
        model = IRAutoencoder().to(DEVICE)
        model.load_state_dict(torch.load(AUTOENCODER_MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"✅ Autoencoder loaded successfully")
        print(f"Anomaly threshold set to: {ANOMALY_THRESHOLD}")
        return model
    except Exception as e:
        print(f"❌ ERROR: Failed to load autoencoder: {e}")
        print("Detection will proceed without false positive filtering.")
        return None


def is_anomaly(autoencoder, crop_img):
    """
    Determine if a detection crop is an anomaly (FOD) or normal runway marking.

    Args:
        autoencoder: The trained autoencoder model
        crop_img: The cropped image from detection (numpy array)

    Returns:
        is_fod (bool): True if the crop is an anomaly (FOD), False otherwise
        error (float): The reconstruction error
    """
    if autoencoder is None:
        return True, 0.0  # If no autoencoder or disabled, accept all detections

    # Skip tiny crops that might cause issues
    if crop_img.shape[0] < 10 or crop_img.shape[1] < 10:
        if DEBUG_MODE:
            print(f"Skipping tiny crop of size {crop_img.shape}")
        return True, 0.0

    # Convert to grayscale if needed
    if len(crop_img.shape) == 3 and crop_img.shape[2] == 3:
        crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop_img

    # Resize to expected size
    crop_resized = cv2.resize(crop_gray, (AUTOENCODER_CROP_SIZE, AUTOENCODER_CROP_SIZE))

    # Print min/max of the crop for debugging
    if DEBUG_MODE and np.random.random() < 0.05:  # Only print occasionally
        print(f"Crop min/max: {np.min(crop_resized)}/{np.max(crop_resized)}")

    # Normalize and convert to tensor (ensure proper scaling)
    crop_tensor = torch.from_numpy(crop_resized).float() / 255.0
    crop_tensor = crop_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)  # Add batch and channel dimensions

    # Get reconstruction
    with torch.no_grad():
        reconstruction = autoencoder(crop_tensor)

    # Calculate reconstruction error (MSE)
    error = torch.mean((reconstruction - crop_tensor) ** 2).item()

    # Determine if it's an anomaly based on threshold
    is_fod = error > ANOMALY_THRESHOLD

    # In debug mode, show some random samples
    if DEBUG_MODE and np.random.random() < 0.05:  # Only print occasionally
        print(f"Crop error: {error:.6f}, Threshold: {ANOMALY_THRESHOLD}, Decision: {'FOD' if is_fod else 'Normal'}")

    # For very small errors, always consider normal (avoids false positives)
    if error < ANOMALY_THRESHOLD * 0.5:
        is_fod = False

    # For very large errors, always consider FOD (avoids false negatives)
    if error > ANOMALY_THRESHOLD * 2.0:
        is_fod = True

    return is_fod, error


def save_detection_crop(frame, box, frame_idx, detection_idx, is_accepted, error=None):
    """
    Save detection crop to appropriate folder (accepted or rejected).

    Args:
        frame: The full video frame
        box: Bounding box coordinates [x1, y1, x2, y2]
        frame_idx: Current frame index
        detection_idx: Detection index in the current frame
        is_accepted: Whether the detection was accepted as FOD
        error: Optional reconstruction error value
    """
    # Skip saving crops if we're not using autoencoder and not explicitly saving all crops
    if not USE_AUTOENCODER and not SAVE_ALL_CROPS:
        return

    # Create crop folders if they don't exist
    os.makedirs(ACCEPTED_FOLDER, exist_ok=True)
    os.makedirs(REJECTED_FOLDER, exist_ok=True)

    # Extract crop coordinates
    x1, y1, x2, y2 = map(int, box)

    # Ensure coordinates are within frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    # Extract crop from frame
    crop = frame[y1:y2, x1:x2]

    # Skip if crop is empty
    if crop.size == 0:
        return

    # Determine destination folder
    dest_folder = ACCEPTED_FOLDER if is_accepted else REJECTED_FOLDER

    # Create filename with frame and detection indices
    if error is not None:
        filename = f"frame_{frame_idx}_det_{detection_idx}_error_{error:.6f}.jpg"
    else:
        filename = f"frame_{frame_idx}_det_{detection_idx}.jpg"

    filepath = os.path.join(dest_folder, filename)

    # Save crop
    cv2.imwrite(filepath, crop)


# ========================
# HELPER FUNCTIONS
# ========================
def iou(boxA, boxB):
    """
    Calculate intersection over union between two bounding boxes.

    Args:
        boxA: First bounding box in format [x1, y1, x2, y2]
        boxB: Second bounding box in format [x1, y1, x2, y2]

    Returns:
        float: IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate area of intersection
    inter = max(0, xB - xA) * max(0, yB - yA)

    # Calculate areas of both boxes
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calculate IoU
    return inter / float(areaA + areaB - inter + 1e-6)


def merge_boxes(boxes, scores, iou_threshold):
    """
    Merge overlapping bounding boxes using Non-Maximum Suppression.

    Args:
        boxes: List of bounding boxes in format [x1, y1, x2, y2]
        scores: Confidence scores for each box
        iou_threshold: IoU threshold for considering boxes as overlapping

    Returns:
        merged_boxes: List of merged bounding boxes
        merged_scores: List of scores for merged boxes
    """
    if len(boxes) == 0:
        return [], []

    # Convert to numpy arrays if not already
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Sort boxes by score in descending order
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    scores = scores[indices]

    # Initialize list for keeping merged boxes and scores
    merged_boxes = []
    merged_scores = []

    # Process each box
    while len(boxes) > 0:
        # Take the box with highest score
        current_box = boxes[0]
        current_score = scores[0]

        # Add to merged boxes
        merged_boxes.append(current_box)
        merged_scores.append(current_score)

        # Remove the current box
        boxes = boxes[1:]
        scores = scores[1:]

        if len(boxes) == 0:
            break

        # Calculate IoU with all remaining boxes
        ious = np.array([iou(current_box, box) for box in boxes])

        # Find boxes to merge (those with IoU > threshold)
        merge_indices = np.where(ious > iou_threshold)[0]

        if len(merge_indices) > 0:
            # Merge boxes by taking weighted average based on confidence
            merge_boxes = boxes[merge_indices]
            merge_scores = scores[merge_indices]

            # Calculate weights based on scores
            weights = merge_scores / np.sum(merge_scores)

            # Add current box with its weight
            all_boxes = np.vstack([current_box.reshape(1, -1), merge_boxes])
            all_weights = np.append([current_score], merge_scores)
            all_weights = all_weights / np.sum(all_weights)

            # Calculate weighted average box
            merged_box = np.sum(all_boxes * all_weights.reshape(-1, 1), axis=0)
            merged_boxes[-1] = merged_box  # Replace last added box with merged one

            # Remove merged boxes
            keep_indices = np.ones(len(boxes), dtype=bool)
            keep_indices[merge_indices] = False
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]

    return np.array(merged_boxes), np.array(merged_scores)


def find_fod_in_frame(yolo_boxes, rcnn_boxes, rcnn_scores, rcnn_labels, iou_threshold, conf_threshold_rcnn):
    """
    Combine YOLO and R-CNN detections to find FOD in the frame.
    Using OR logic - detect FOD if EITHER model detects it.

    Args:
        yolo_boxes: YOLO detection boxes
        rcnn_boxes: R-CNN detection boxes
        rcnn_scores: R-CNN detection scores
        rcnn_labels: R-CNN detection labels
        iou_threshold: IoU threshold for determining overlapping detections
        conf_threshold_rcnn: Confidence threshold for R-CNN detections

    Returns:
        bool: True if FOD is detected, False otherwise
    """
    # IMPORTANT CHANGE: Using OR logic for fod_detected
    fod_detected = False

    # Check YOLO detections (if any YOLO detection exists, there is FOD)
    if len(yolo_boxes) > 0:
        fod_detected = True

    # Check R-CNN detections
    for box, score, label in zip(rcnn_boxes, rcnn_scores, rcnn_labels):
        if score >= conf_threshold_rcnn:
            # Any valid RCNN detection is considered FOD
            fod_detected = True
            break

    return fod_detected


# ========================
# MODEL LOADING
# ========================
def load_models():
    """
    Load the detection models.

    Returns:
        yolo: The loaded YOLO model
        rcnn: The loaded R-CNN model
    """
    print(f"\n===== LOADING DETECTION MODELS =====")

    # Load YOLO model
    print(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
    try:
        yolo = YOLO(YOLO_MODEL_PATH).to(DEVICE)
        # Enable half precision for YOLO if configured
        if USE_HALF and DEVICE == 'cuda':
            print("Enabling half precision for YOLO model")
            yolo.model.half()
        print(f"✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to load YOLO model: {e}")
        exit(1)

    # Load R-CNN model
    print(f"Loading R-CNN model from: {RCNN_MODEL_PATH}")
    try:
        rcnn = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=NUM_CLASSES)
        rcnn.load_state_dict(torch.load(RCNN_MODEL_PATH, map_location=DEVICE))
        rcnn.to(DEVICE).eval()
        # Enable half precision for R-CNN if configured
        if USE_HALF and DEVICE == 'cuda':
            print("Enabling half precision for R-CNN model")
            rcnn.half()
        print(f"✅ R-CNN model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to load R-CNN model: {e}")
        exit(1)

    return yolo, rcnn


# ========================
# SIMPLE ROI SELECTION GUI
# ========================
def select_roi(video_path):
    """
    Open the video and let the user select an ROI using OpenCV's built-in selector.

    Args:
        video_path: Path to the input video file

    Returns:
        tuple: Selected ROI as (x, y, width, height) or None if no selection
    """
    print(f"Opening video for ROI selection: {video_path}")

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return None

    # Resize the frame if it's too large
    height, width = frame.shape[:2]
    if width > 1280 or height > 720:
        scale = min(1280 / width, 720 / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
        print(f"Resized frame to {new_width}x{new_height} for ROI selection")

    # Display instructions
    cv2.putText(frame, "Select ROI and press ENTER. Press ESC to cancel.",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Use OpenCV's built-in ROI selector
    roi = cv2.selectROI("Select Area of Interest (ROI)", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # If ROI has no area, return None
    if roi[2] == 0 or roi[3] == 0:
        print("No ROI selected or selection canceled.")
        cap.release()
        return None

    # If we resized the frame, adjust the ROI coordinates back to original scale
    if width > 1280 or height > 720:
        roi = (
            int(roi[0] / scale),
            int(roi[1] / scale),
            int(roi[2] / scale),
            int(roi[3] / scale)
        )

    cap.release()
    return roi


# ========================
# MAIN FUNCTION
# ========================
def main():
    """Main function to process video and create visualizations."""
    print("\n===== IR FOD DETECTION SYSTEM =====")
    print("===== Grayscale Detection with Autoencoder Verification =====\n")
    print(f"Autoencoder processing: {'ENABLED' if USE_AUTOENCODER else 'DISABLED'}")

    # Ensure input and output directories exist
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CROPS_FOLDER, exist_ok=True)

    # Only create accepted/rejected folders if using autoencoder or saving all crops
    if USE_AUTOENCODER or SAVE_ALL_CROPS:
        os.makedirs(ACCEPTED_FOLDER, exist_ok=True)
        os.makedirs(REJECTED_FOLDER, exist_ok=True)

    # Create file dialog to select input video
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    video_path = filedialog.askopenfilename(
        title="Select Input Video",
        filetypes=[("Video files", "*.mp4;*.avi;*.mov"), ("All files", "*.*")]
    )

    if not video_path:
        print("No video selected. Exiting.")
        return

    print(f"Selected video: {video_path}")

    # Check if required model files exist
    required_files = [YOLO_MODEL_PATH, RCNN_MODEL_PATH]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ ERROR: Required model file not found: {file_path}")
            print(f"Please place all model files in the {INPUT_FOLDER} folder.")
            exit(1)

    # Let user select ROI
    roi = select_roi(video_path)
    if roi:
        print(f"Selected ROI: {roi} (x, y, width, height)")
    else:
        print("No ROI selected. Will process the entire frame.")
        roi = None

    # Load detection models
    yolo, rcnn = load_models()

    # Load autoencoder for false positive filtering (only if enabled)
    autoencoder = load_autoencoder() if USE_AUTOENCODER else None

    # ========================
    # VIDEO INITIALIZATION
    # ========================
    print(f"\n===== PROCESSING VIDEO =====")
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video {video_path}")
        exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height} @ {fps}fps, {frame_count} frames")

    # Set up output video paths
    output_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_yolo = os.path.join(OUTPUT_FOLDER, f"{output_basename}_yolo.mp4")
    output_rcnn = os.path.join(OUTPUT_FOLDER, f"{output_basename}_rcnn.mp4")
    output_combined = os.path.join(OUTPUT_FOLDER, f"{output_basename}_combined.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_yolo = cv2.VideoWriter(output_yolo, fourcc, fps / FRAME_SKIP, (width, height))
    out_rcnn = cv2.VideoWriter(output_rcnn, fourcc, fps / FRAME_SKIP, (width, height))
    out_combined = cv2.VideoWriter(output_combined, fourcc, fps / FRAME_SKIP, (width, height))

    # ========================
    # PROCESSING LOOP
    # ========================
    frame_idx = 0
    processed = 0
    start_time = time.time()
    fod_count = 0
    accepted_count = 0
    rejected_count = 0

    print(f"Starting processing with frame skip = {FRAME_SKIP}")
    print(f"Detection thresholds: YOLO = {CONF_THRESHOLD_YOLO}, R-CNN = {CONF_THRESHOLD_RCNN}, IoU = {IOU_THRESHOLD}")
    print(f"Input processing: Grayscale conversion (R-CNN model was trained on B&W images)")
    if autoencoder and USE_AUTOENCODER:
        print(f"Autoencoder anomaly threshold: {ANOMALY_THRESHOLD}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        # Extract ROI if specified
        if roi:
            x, y, w, h = roi
            roi_frame = frame[y:y + h, x:x + w]
        else:
            roi_frame = frame

        # Preprocess - convert to grayscale since RCNN was trained on black and white images
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        # Convert to 3-channel grayscale (required for both YOLO and R-CNN input formats)
        gray_roi_3ch = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2RGB)
        image_tensor = F.to_tensor(gray_roi_3ch).unsqueeze(0).to(DEVICE)

        # YOLOv8 Detection on ROI - use 3-channel grayscale for IR imagery
        with torch.amp.autocast('cuda', enabled=USE_AMP and DEVICE == 'cuda'):
            yolo_results = yolo.predict(source=gray_roi_3ch, conf=CONF_THRESHOLD_YOLO, verbose=False, device=DEVICE)[0]

        # Extract YOLO boxes and scores
        yolo_boxes = yolo_results.boxes.xyxy.cpu().numpy()
        yolo_scores = yolo_results.boxes.conf.cpu().numpy()

        # Faster R-CNN Detection on ROI
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=USE_AMP and DEVICE == 'cuda'):
                rcnn_output = rcnn(image_tensor)[0]

        rcnn_boxes = rcnn_output['boxes'].cpu().numpy()
        rcnn_scores = rcnn_output['scores'].cpu().numpy()
        rcnn_labels = rcnn_output['labels'].cpu().numpy()

        # Prepare frames for visualization
        frame_yolo = frame.copy()
        frame_rcnn = frame.copy()
        frame_combined = frame.copy()

        # If ROI was used, need to adjust bounding box coordinates back to original frame
        if roi:
            roi_x, roi_y = roi[0], roi[1]
            # Adjust YOLO boxes
            if len(yolo_boxes) > 0:
                yolo_boxes[:, [0, 2]] += roi_x
                yolo_boxes[:, [1, 3]] += roi_y

            # Adjust RCNN boxes
            if len(rcnn_boxes) > 0:
                rcnn_boxes[:, [0, 2]] += roi_x
                rcnn_boxes[:, [1, 3]] += roi_y

        # Draw ROI rectangle on all output frames
        if roi:
            x, y, w, h = roi
            roi_color = (255, 255, 0)  # Yellow for ROI
            cv2.rectangle(frame_yolo, (x, y), (x + w, y + h), roi_color, 2)
            cv2.rectangle(frame_rcnn, (x, y), (x + w, y + h), roi_color, 2)
            cv2.rectangle(frame_combined, (x, y), (x + w, y + h), roi_color, 2)

        # Process YOLO detections
        valid_yolo_boxes = []
        valid_yolo_scores = []

        # Draw YOLO detections first
        for idx, (box, score) in enumerate(zip(yolo_boxes, yolo_scores)):
            x1, y1, x2, y2 = box.astype(int)
            crop = frame[y1:y2, x1:x2]

            # Skip empty crops
            if crop.size == 0:
                continue

            # Check with autoencoder if enabled
            if USE_AUTOENCODER:
                is_fod, error = is_anomaly(autoencoder, crop)
            else:
                # If autoencoder disabled, consider all detections as FOD
                is_fod, error = True, 0.0

            # Draw on YOLO output frame
            box_color = (255, 0, 0) if is_fod else (128, 128, 255)  # Blue if FOD, Light blue if not
            cv2.rectangle(frame_yolo, (x1, y1), (x2, y2), box_color, 2)

            label_text = f"FOD: {score:.2f}"
            if USE_AUTOENCODER and autoencoder:
                label_text += f" | Err: {error:.4f}"
            cv2.putText(frame_yolo, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Save crop for analysis
            if SAVE_ALL_CROPS or USE_AUTOENCODER:
                save_detection_crop(frame, box, frame_idx, idx, is_fod, error)

            # IMPORTANT CHANGE: Keep all YOLO detections for combined view
            # (using OR logic for detection combination)
            valid_yolo_boxes.append(box)
            valid_yolo_scores.append(score)

            if is_fod:
                accepted_count += 1
            else:
                rejected_count += 1

        # Process RCNN detections
        rcnn_filtered_boxes = []
        rcnn_filtered_scores = []
        rcnn_filtered_labels = []

        for idx, (box, score, label) in enumerate(zip(rcnn_boxes, rcnn_scores, rcnn_labels)):
            if score >= CONF_THRESHOLD_RCNN:
                x1, y1, x2, y2 = box.astype(int)
                crop = frame[y1:y2, x1:x2]

                # Skip empty crops
                if crop.size == 0:
                    continue

                # Check with autoencoder if enabled
                if USE_AUTOENCODER:
                    is_fod, error = is_anomaly(autoencoder, crop)
                else:
                    # If autoencoder disabled, consider all detections as FOD
                    is_fod, error = True, 0.0

                # Draw on RCNN output frame
                box_color = (0, 0, 255) if is_fod else (128, 128, 255)  # Red if FOD, Light blue if not
                cv2.rectangle(frame_rcnn, (x1, y1), (x2, y2), box_color, 2)

                label_text = f"FOD: {score:.2f}"
                if USE_AUTOENCODER and autoencoder:
                    label_text += f" | Err: {error:.4f}"
                cv2.putText(frame_rcnn, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Save crop for analysis
                if SAVE_ALL_CROPS or USE_AUTOENCODER:
                    save_detection_crop(frame, box, frame_idx, len(yolo_boxes) + idx, is_fod, error)

                # IMPORTANT CHANGE: Keep all RCNN detections for combined view
                # (using OR logic for detection combination)
                rcnn_filtered_boxes.append(box)
                rcnn_filtered_scores.append(score)
                rcnn_filtered_labels.append(label)

                if is_fod:
                    accepted_count += 1
                else:
                    rejected_count += 1

        # Initialize variables for combined detection
        merged_boxes = []
        merged_scores = []

        # Combined: Merge overlapping boxes from both models
        # First, combine all valid boxes from both models
        all_boxes = []
        all_scores = []
        all_sources = []  # 0 for YOLO, 1 for RCNN

        # Add YOLO boxes
        for box, score in zip(valid_yolo_boxes, valid_yolo_scores):
            all_boxes.append(box)
            all_scores.append(score)
            all_sources.append(0)

        # Add RCNN boxes
        for box, score in zip(rcnn_filtered_boxes, rcnn_filtered_scores):
            all_boxes.append(box)
            all_scores.append(score)
            all_sources.append(1)

        # Merge overlapping boxes
        if len(all_boxes) > 0:
            # Convert to numpy arrays
            all_boxes = np.array(all_boxes)
            all_scores = np.array(all_scores)
            all_sources = np.array(all_sources)

            # Merge boxes - this combines overlapping boxes from either model
            merged_boxes, merged_scores = merge_boxes(all_boxes, all_scores, IOU_THRESHOLD)

            # Draw merged boxes on combined output
            frame_detections = 0
            for box, score in zip(merged_boxes, merged_scores):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame_combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_combined, f"FOD: {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                frame_detections += 1

            # Update FOD count
            fod_count += frame_detections

        # Add frame info to all outputs
        frame_info = f"Frame: {frame_idx}/{frame_count} | Combined: {len(merged_boxes)} | YOLO: {len(yolo_boxes)} | RCNN: {len(rcnn_filtered_boxes)}"
        if USE_AUTOENCODER and autoencoder:
            frame_info += f" | Accept/Reject: {accepted_count}/{rejected_count}"

        cv2.putText(frame_yolo, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_rcnn, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_combined, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write to videos
        out_yolo.write(frame_yolo)
        out_rcnn.write(frame_rcnn)
        out_combined.write(frame_combined)

        processed += 1
        frame_idx += 1

        if processed % 10 == 0 or processed == 1:
            percent = 100 * frame_idx / frame_count
            elapsed_time = time.time() - start_time
            remaining = (elapsed_time / frame_idx) * (frame_count - frame_idx) if frame_idx > 0 else 0
            print(f"Processed {frame_idx}/{frame_count} frames ({percent:.2f}%) - ETA: {remaining:.2f}s")

    # ========================
    # CLEANUP AND FINAL OUTPUT
    # ========================
    cap.release()
    out_yolo.release()
    out_rcnn.release()
    out_combined.release()

    total_time = time.time() - start_time
    fps_processed = processed / total_time

    print(f"\n===== PROCESSING COMPLETE =====")
    print(f"✅ {processed} frames processed in {total_time:.2f} seconds ({fps_processed:.2f} fps)")
    print(f"✅ Video outputs saved to:")
    print(f"   - {output_yolo}")
    print(f"   - {output_rcnn}")
    print(f"   - {output_combined}")
    print(f"✅ Total FOD detections: {fod_count}")
    if USE_AUTOENCODER and autoencoder:
        print(f"✅ Detections analyzed by autoencoder:")
        print(f"   - Accepted as FOD: {accepted_count}")
        print(f"   - Rejected as false positives: {rejected_count}")
        print(f"   - Detection crops saved to: {CROPS_FOLDER}")
    print(f"===== END OF PROCESSING =====\n")


if __name__ == "__main__":
    main()
