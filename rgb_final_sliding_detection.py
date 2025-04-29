import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import os
import re
from datetime import datetime
import math
from PIL import Image, ImageDraw, ImageFont
import torchvision.ops as ops

#################################################
# CONFIGURATION SETTINGS
#################################################
# Paths
INPUT_FOLDER = 'input'  # Input folder for source files
OUTPUT_FOLDER = 'output'  # Output folder for processed files
VIDEO_PATH = os.path.join(INPUT_FOLDER, 'runway_video.mp4')
SRT_PATH = os.path.join(INPUT_FOLDER, 'runway_video.srt')
RUNWAY_IMAGE_PATH = os.path.join(INPUT_FOLDER, 'runway_scan_s.png')
YOLO_MODEL_PATH = os.path.join(INPUT_FOLDER, 'yolov8m.pt')
RCNN_MODEL_PATH = os.path.join(INPUT_FOLDER, 'fasterrcnn_sd_fod.pth')

# Output paths
OUTPUT_YOLO = os.path.join(OUTPUT_FOLDER, 'output_yolo.mp4')
OUTPUT_RCNN = os.path.join(OUTPUT_FOLDER, 'output_rcnn.mp4')
OUTPUT_COMBINED = os.path.join(OUTPUT_FOLDER, 'output_combined.mp4')

# Processing parameters
FRAME_SKIP = 10  # Process every nth frame
CONF_THRESHOLD_YOLO = 0.30  # Confidence threshold for YOLO detections
CONF_THRESHOLD_RCNN = 0.01  # Confidence threshold for R-CNN detections
IOU_THRESHOLD = 0.45  # IoU threshold for NMS
NMS_THRESHOLD = 0.3  # NMS threshold for merging sliding window detections

# Sliding window parameters
WINDOW_SIZE = 512  # Sliding window size
WINDOW_STRIDE = 384  # Step size for sliding window (creates overlap of 128 pixels)

# Model parameters
NUM_CLASSES = 2  # Number of classes for R-CNN
CLASS_NAMES = ["background", "FOD"]  # Class names

# GPU Acceleration settings
USE_AMP = False  # Use Automatic Mixed Precision
USE_HALF = False  # Use half precision (FP16)
CUDA_MEMORY_FRACTION = 0.8  # Use 80% of available GPU memory

# Runway GPS coordinates and pixel mapping
RUNWAY_GPS_CORNERS = {
    'top_left': {'lat': 40.485603, 'lng': 30.163068, 'x': 175, 'y': 115},
    'top_right': {'lat': 40.488899, 'lng': 30.175695, 'x': 26340, 'y': 110},
    'bottom_left': {'lat': 40.485383, 'lng': 30.163164, 'x': 180, 'y': 735},
    'bottom_right': {'lat': 40.488680, 'lng': 30.175774, 'x': 26340, 'y': 730}
}

# Output image configuration
OUTPUT_WIDTH_SEGMENT = 6625  # Width of each output segment
OUTPUT_HEIGHT = 1066  # Height of each output segment
TOTAL_SEGMENTS = 4  # Number of segments to split the runway image into

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


#################################################
# DEVICE INITIALIZATION
#################################################
def initialize_device():
    """
    Initialize and configure the processing device (GPU or CPU)
    """
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


#################################################
# HELPER FUNCTIONS
#################################################
def iou(boxA, boxB):
    """
    Calculate intersection over union between two bounding boxes

    Args:
        boxA: First bounding box [x1, y1, x2, y2]
        boxB: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate intersection area
    inter = max(0, xB - xA) * max(0, yB - yA)

    # Calculate box areas
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calculate IoU
    return inter / float(areaA + areaB - inter + 1e-6)


def parse_srt_file(srt_path):
    """
    Parse SRT file to extract timestamps and GPS coordinates

    Args:
        srt_path: Path to SRT subtitle file

    Returns:
        List of dictionaries with timestamp and GPS data
    """
    print(f"Parsing SRT file: {srt_path}")
    gps_data = []

    try:
        if not os.path.exists(srt_path):
            print(f"❌ ERROR: SRT file not found: {srt_path}")
            print(f"Please place the SRT file in the {INPUT_FOLDER} folder.")
            exit(1)

        with open(srt_path, 'r') as file:
            content = file.read()

        # Regular expression to extract timestamp and GPS coordinates
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\nLat: ([\d.]+), Lng: ([\d.]+)'
        matches = re.findall(pattern, content)

        for match in matches:
            entry_number, start_time, end_time, lat, lng = match

            # Convert timestamp to seconds
            h, m, s = map(float, re.split('[:,]', start_time)[:3])
            start_seconds = h * 3600 + m * 60 + s

            gps_data.append({
                'entry': int(entry_number),
                'timestamp': start_seconds,
                'lat': float(lat),
                'lng': float(lng)
            })

        print(f"✅ Parsed {len(gps_data)} GPS coordinates from SRT file")

        # Print GPS data range
        if gps_data:
            lat_min = min(entry['lat'] for entry in gps_data)
            lat_max = max(entry['lat'] for entry in gps_data)
            lng_min = min(entry['lng'] for entry in gps_data)
            lng_max = max(entry['lng'] for entry in gps_data)
            print(f"GPS data range: Lat [{lat_min:.6f} to {lat_max:.6f}], Lng [{lng_min:.6f} to {lng_max:.6f}]")

        return gps_data

    except Exception as e:
        print(f"❌ ERROR: Failed to parse SRT file: {e}")
        print(f"Please ensure the SRT file has the correct format.")
        exit(1)


def gps_to_pixel(lat, lng, runway_gps_corners):
    """
    Convert GPS coordinates to pixel coordinates on the runway image

    Args:
        lat: Latitude coordinate
        lng: Longitude coordinate
        runway_gps_corners: Dictionary with runway corner coordinates

    Returns:
        Tuple of (x, y) pixel coordinates
    """
    # Use all four corners to determine the actual GPS bounds of the runway
    lats = [runway_gps_corners['top_left']['lat'],
            runway_gps_corners['top_right']['lat'],
            runway_gps_corners['bottom_left']['lat'],
            runway_gps_corners['bottom_right']['lat']]

    lngs = [runway_gps_corners['top_left']['lng'],
            runway_gps_corners['top_right']['lng'],
            runway_gps_corners['bottom_left']['lng'],
            runway_gps_corners['bottom_right']['lng']]

    lat_min = min(lats)
    lat_max = max(lats)
    lng_min = min(lngs)
    lng_max = max(lngs)

    # Check if coordinate is outside bounds and clamp it
    if lat < lat_min or lat > lat_max or lng < lng_min or lng > lng_max:
        lat = max(lat_min, min(lat, lat_max))
        lng = max(lng_min, min(lng, lng_max))

    # Normalize the GPS coordinates (0 to 1)
    lat_norm = (lat - lat_min) / (lat_max - lat_min)
    lng_norm = (lng - lng_min) / (lng_max - lng_min)

    # Perform bilinear interpolation for more accurate mapping
    # Handle the fact that runway might not be perfectly rectangular in GPS space
    if lng_norm <= 0.5 and lat_norm <= 0.5:
        # Top-left quadrant
        x = runway_gps_corners['top_left']['x'] + 2 * lng_norm * (
                runway_gps_corners['top_right']['x'] - runway_gps_corners['top_left']['x'])
        y = runway_gps_corners['top_left']['y'] + 2 * lat_norm * (
                runway_gps_corners['bottom_left']['y'] - runway_gps_corners['top_left']['y'])
    elif lng_norm > 0.5 and lat_norm <= 0.5:
        # Top-right quadrant
        x = runway_gps_corners['top_right']['x'] - 2 * (1 - lng_norm) * (
                runway_gps_corners['top_right']['x'] - runway_gps_corners['top_left']['x'])
        y = runway_gps_corners['top_right']['y'] + 2 * lat_norm * (
                runway_gps_corners['bottom_right']['y'] - runway_gps_corners['top_right']['y'])
    elif lng_norm <= 0.5 and lat_norm > 0.5:
        # Bottom-left quadrant
        x = runway_gps_corners['bottom_left']['x'] + 2 * lng_norm * (
                runway_gps_corners['bottom_right']['x'] - runway_gps_corners['bottom_left']['x'])
        y = runway_gps_corners['bottom_left']['y'] - 2 * (1 - lat_norm) * (
                runway_gps_corners['bottom_left']['y'] - runway_gps_corners['top_left']['y'])
    else:
        # Bottom-right quadrant
        x = runway_gps_corners['bottom_right']['x'] - 2 * (1 - lng_norm) * (
                runway_gps_corners['bottom_right']['x'] - runway_gps_corners['bottom_left']['x'])
        y = runway_gps_corners['bottom_right']['y'] - 2 * (1 - lat_norm) * (
                runway_gps_corners['bottom_right']['y'] - runway_gps_corners['top_right']['y'])

    x = int(x)
    y = int(y)

    # Ensure coordinates are within image bounds (use the maximum x and y from corners)
    max_x = max(runway_gps_corners['top_right']['x'], runway_gps_corners['bottom_right']['x'])
    max_y = max(runway_gps_corners['bottom_left']['y'], runway_gps_corners['bottom_right']['y'])

    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))

    return x, y


def get_gps_for_timestamp(timestamp, gps_data):
    """
    Find the closest GPS entry for a given timestamp

    Args:
        timestamp: Video timestamp in seconds
        gps_data: List of GPS data entries

    Returns:
        Dictionary with closest GPS entry or None if no data available
    """
    if not gps_data:
        return None

    # Find the entry with the closest timestamp
    closest_entry = min(gps_data, key=lambda x: abs(x['timestamp'] - timestamp))
    return closest_entry


def create_sliding_windows(frame, window_size, stride):
    """
    Create sliding windows from a frame with specified size and stride

    Args:
        frame: Input frame
        window_size: Size of each window (square)
        stride: Step size between windows

    Returns:
        List of tuples (window, (x, y)) where (x, y) is the top-left coordinate
    """
    height, width = frame.shape[:2]
    windows = []

    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            # Extract window
            window = frame[y:y + window_size, x:x + window_size]
            windows.append((window, (x, y)))

    # Handle edge cases to ensure full coverage
    # Right edge
    if width % stride != 0:
        for y in range(0, height - window_size + 1, stride):
            x = width - window_size
            window = frame[y:y + window_size, x:x + window_size]
            windows.append((window, (x, y)))

    # Bottom edge
    if height % stride != 0:
        for x in range(0, width - window_size + 1, stride):
            y = height - window_size
            window = frame[y:y + window_size, x:x + window_size]
            windows.append((window, (x, y)))

    # Bottom-right corner
    if width % stride != 0 and height % stride != 0:
        x = width - window_size
        y = height - window_size
        window = frame[y:y + window_size, x:x + window_size]
        windows.append((window, (x, y)))

    return windows


def convert_window_boxes_to_frame(boxes, window_pos):
    """
    Convert bounding box coordinates from window to full frame

    Args:
        boxes: Bounding boxes in window coordinates [x1, y1, x2, y2]
        window_pos: (x, y) position of window in frame

    Returns:
        Bounding boxes in frame coordinates
    """
    x_offset, y_offset = window_pos
    frame_boxes = boxes.copy()

    # Add offsets to convert to frame coordinates
    if len(frame_boxes) > 0:
        frame_boxes[:, 0] += x_offset  # x1
        frame_boxes[:, 1] += y_offset  # y1
        frame_boxes[:, 2] += x_offset  # x2
        frame_boxes[:, 3] += y_offset  # y2

    return frame_boxes


def merge_window_detections(all_boxes, all_scores, nms_threshold=0.3):
    """
    Merge detections from multiple windows using non-maximum suppression

    Args:
        all_boxes: List of bounding boxes in frame coordinates
        all_scores: List of confidence scores for each box
        nms_threshold: IoU threshold for NMS

    Returns:
        Tuple of merged boxes and scores
    """
    if len(all_boxes) == 0:
        return np.array([]), np.array([])

    # Convert to tensor for torchvision NMS
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)

    # Apply NMS
    keep_indices = ops.nms(boxes_tensor, scores_tensor, nms_threshold)

    # Convert back to numpy
    merged_boxes = boxes_tensor[keep_indices].numpy()
    merged_scores = scores_tensor[keep_indices].numpy()

    return merged_boxes, merged_scores


def detect_with_sliding_windows(frame, yolo_model, window_size, stride, conf_threshold):
    """
    Detect objects using sliding windows with YOLO model

    Args:
        frame: Input frame
        yolo_model: YOLO model
        window_size: Size of sliding window
        stride: Step size between windows
        conf_threshold: Confidence threshold for detections

    Returns:
        Merged boxes and scores across all windows
    """
    # Create sliding windows
    windows = create_sliding_windows(frame, window_size, stride)

    # Lists to collect all detections
    all_boxes = []
    all_scores = []

    # Process each window
    for window, window_pos in windows:
        # Run YOLO on the window
        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE == 'cuda'):
            results = yolo_model.predict(source=window, conf=conf_threshold, verbose=False, device=DEVICE)[0]

        # Get boxes and scores
        window_boxes = results.boxes.xyxy.cpu().numpy()
        window_scores = results.boxes.conf.cpu().numpy()

        # Convert window coordinates to frame coordinates
        if len(window_boxes) > 0:
            frame_boxes = convert_window_boxes_to_frame(window_boxes, window_pos)
            all_boxes.extend(frame_boxes)
            all_scores.extend(window_scores)

    # Convert to numpy arrays
    all_boxes = np.array(all_boxes) if all_boxes else np.array([])
    all_scores = np.array(all_scores) if all_scores else np.array([])

    # Merge overlapping detections using NMS
    if len(all_boxes) > 0:
        merged_boxes, merged_scores = merge_window_detections(all_boxes, all_scores, NMS_THRESHOLD)
        return merged_boxes, merged_scores
    else:
        return np.array([]), np.array([])


def find_fod_in_frame(yolo_boxes, rcnn_boxes, rcnn_scores, rcnn_labels, iou_threshold, conf_threshold_rcnn):
    """
    Combine YOLO and R-CNN detections to find FOD in the frame

    Args:
        yolo_boxes: Bounding boxes from YOLO
        rcnn_boxes: Bounding boxes from R-CNN
        rcnn_scores: Confidence scores from R-CNN
        rcnn_labels: Class labels from R-CNN
        iou_threshold: IoU threshold for detection overlap
        conf_threshold_rcnn: Confidence threshold for R-CNN

    Returns:
        Boolean indicating if FOD was detected
    """
    fod_detected = False

    # Check YOLO detections
    if len(yolo_boxes) > 0:
        fod_detected = True

    # Check R-CNN detections
    for box, score, label in zip(rcnn_boxes, rcnn_scores, rcnn_labels):
        if score >= conf_threshold_rcnn:
            # Check if this R-CNN detection overlaps with any YOLO detection
            overlap = any(iou(box, ybox) > iou_threshold for ybox in yolo_boxes)

            # If no overlap, it's a unique R-CNN detection
            if not overlap:
                fod_detected = True

    return fod_detected


def initialize_runway_image():
    """
    Initialize the runway image for visualization

    Returns:
        PIL Image object of the runway
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(INPUT_FOLDER, exist_ok=True)

        # Check if the runway image exists
        if not os.path.exists(RUNWAY_IMAGE_PATH):
            print(f"❌ ERROR: Runway image not found: {RUNWAY_IMAGE_PATH}")
            print(f"Please place the runway image in the {INPUT_FOLDER} folder.")
            exit(1)
        else:
            # Load the runway image
            runway_img = Image.open(RUNWAY_IMAGE_PATH)
            print(f"✅ Loaded runway image: {runway_img.size}")

        return runway_img

    except Exception as e:
        print(f"❌ ERROR: Failed to initialize runway image: {e}")
        print(f"Please ensure all input files are in the {INPUT_FOLDER} folder.")
        exit(1)


def split_and_save_runway_image(runway_img, drone_path, fod_locations, total_fod_count):
    """
    Split the runway image into segments and save them

    Args:
        runway_img: PIL Image of the runway
        drone_path: List of (x, y) coordinates representing the drone path
        fod_locations: List of (x, y) coordinates where FOD was detected
        total_fod_count: Total number of FOD detections
    """
    print(f"\n===== GENERATING RUNWAY VISUALIZATION =====")
    print(f"Drone path points: {len(drone_path)}, FOD locations: {len(fod_locations)}")

    total_width = runway_img.width
    segment_width = OUTPUT_WIDTH_SEGMENT

    # Calculate how many segments are needed
    num_segments = math.ceil(total_width / segment_width)
    num_segments = min(num_segments, TOTAL_SEGMENTS)  # Limit to 4 segments as requested
    print(f"Splitting image into {num_segments} segments of width {segment_width}px")

    fod_counts_per_segment = [0] * num_segments

    # Count FOD in each segment
    for x, y in fod_locations:
        segment_idx = min(int(x / segment_width), num_segments - 1)
        fod_counts_per_segment[segment_idx] += 1

    # Create and save each segment
    for i in range(num_segments):
        # Calculate the boundaries for this segment
        left = i * segment_width
        right = min((i + 1) * segment_width, total_width)
        width = right - left

        # Create a new image for this segment
        segment = Image.new('RGB', (width, OUTPUT_HEIGHT), color=(200, 200, 200))
        segment_draw = ImageDraw.Draw(segment)

        # Copy the relevant portion of the runway image
        segment.paste(runway_img.crop((left, 0, right, OUTPUT_HEIGHT)), (0, 0))

        # Draw drone path segments that fall within this segment
        prev_x, prev_y = None, None
        path_points_in_segment = 0

        for x, y in drone_path:
            if x >= left and x < right:
                x_offset = x - left
                path_points_in_segment += 1
                if prev_x is not None and prev_y is not None:
                    # Only draw line if previous point is also in this segment
                    if prev_x >= left and prev_x < right:
                        prev_x_offset = prev_x - left
                        segment_draw.line([(prev_x_offset, prev_y), (x_offset, y)], fill=(255, 0, 0),
                                          width=8)  # Doubled line width
                prev_x, prev_y = x, y

        # Draw FOD markers that fall within this segment
        fod_in_segment = 0
        for x, y in fod_locations:
            if x >= left and x < right:
                x_offset = x - left
                segment_draw.ellipse([(x_offset - 15, y - 15), (x_offset + 15, y + 15)],
                                     fill=(0, 0, 128))  # 25% reduced as requested
                fod_in_segment += 1

        # Add text information with direct CV2 approach
        # Convert PIL image to OpenCV format temporarily for text rendering
        segment_cv = np.array(segment)
        segment_cv = cv2.cvtColor(segment_cv, cv2.COLOR_RGB2BGR)

        # Create text
        text = f"WEST <- {i + 1}/{num_segments} -> EAST | FOD: {fod_counts_per_segment[i]} | TOTAL: {total_fod_count}"

        # Get text size to determine background width
        font_scale = 3.0  # Very large text
        thickness = 6  # Very thick text
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)

        # Add text with OpenCV - first add background just for the text width plus padding
        padding = 40  # Padding on each side
        cv2.rectangle(segment_cv,
                      (10, OUTPUT_HEIGHT - 200),
                      (10 + text_width + padding, OUTPUT_HEIGHT - 80),
                      (0, 0, 0), -1)

        # Then add text in a very large font
        cv2.putText(segment_cv, text, (20, OUTPUT_HEIGHT - 120),
                    font_face, font_scale, (255, 255, 255), thickness)

        # Convert back to PIL for saving
        segment_cv = cv2.cvtColor(segment_cv, cv2.COLOR_BGR2RGB)
        segment = Image.fromarray(segment_cv)

        # Save the segment
        output_path = os.path.join(OUTPUT_FOLDER, f"runway_processed_{i + 1}of{num_segments}.png")
        segment.save(output_path)
        print(f"✅ Saved segment {i + 1}/{num_segments} to {output_path}")


#################################################
# MODEL LOADING
#################################################
def load_models():
    """
    Load the YOLO and Faster R-CNN detection models

    Returns:
        Tuple of (YOLO model, R-CNN model)
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


#################################################
# MAIN FUNCTION
#################################################
def main():
    """
    Main function to process video and create visualizations
    """
    print("\n===== FOD DETECTION AND DRONE PATH ANALYSIS =====\n")

    # Initialize device
    initialize_device()

    # Ensure input and output directories exist
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Check if required input files exist
    required_files = [VIDEO_PATH, SRT_PATH, RUNWAY_IMAGE_PATH, YOLO_MODEL_PATH, RCNN_MODEL_PATH]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ ERROR: Required file not found: {file_path}")
            print(f"Please place all input files in the {INPUT_FOLDER} folder.")
            exit(1)

    # Parse SRT file to get GPS data
    gps_data = parse_srt_file(SRT_PATH)

    # Initialize runway image for visualization
    runway_img = initialize_runway_image()
    runway_draw = ImageDraw.Draw(runway_img)

    # Load detection models
    yolo, rcnn = load_models()

    # ========================
    # VIDEO INITIALIZATION
    # ========================
    print(f"\n===== PROCESSING VIDEO =====")
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video {VIDEO_PATH}")
        exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height} @ {fps}fps, {frame_count} frames")
    print(f"Using sliding window detection with size={WINDOW_SIZE}x{WINDOW_SIZE}, stride={WINDOW_STRIDE}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_yolo = cv2.VideoWriter(OUTPUT_YOLO, fourcc, fps / FRAME_SKIP, (width, height))
    out_rcnn = cv2.VideoWriter(OUTPUT_RCNN, fourcc, fps / FRAME_SKIP, (width, height))
    out_combined = cv2.VideoWriter(OUTPUT_COMBINED, fourcc, fps / FRAME_SKIP, (width, height))

    # Lists to store drone path and FOD locations
    drone_path = []
    fod_locations = []

    # ========================
    # PROCESSING LOOP
    # ========================
    frame_idx = 0
    processed = 0
    start_time = time.time()
    last_pixel_x, last_pixel_y = None, None

    print(f"Starting processing with frame skip = {FRAME_SKIP}")
    print(f"Detection thresholds: YOLO = {CONF_THRESHOLD_YOLO}, R-CNN = {CONF_THRESHOLD_RCNN}, IoU = {IOU_THRESHOLD}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        # Calculate current timestamp in the video
        current_timestamp = frame_idx / fps

        # Get GPS data for current frame
        gps_entry = get_gps_for_timestamp(current_timestamp, gps_data)

        # Preprocess for R-CNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(rgb_frame).unsqueeze(0).to(DEVICE)

        # ========================
        # SLIDING WINDOW YOLO DETECTION
        # ========================
        # Process frame with sliding windows for YOLO detection
        yolo_boxes, yolo_scores = detect_with_sliding_windows(
            frame, yolo, WINDOW_SIZE, WINDOW_STRIDE, CONF_THRESHOLD_YOLO
        )

        # ========================
        # R-CNN DETECTION
        # ========================
        # Faster R-CNN Detection on the full frame
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE == 'cuda'):
                rcnn_output = rcnn(image_tensor)[0]
        rcnn_boxes = rcnn_output['boxes'].cpu().numpy()
        rcnn_scores = rcnn_output['scores'].cpu().numpy()
        rcnn_labels = rcnn_output['labels'].cpu().numpy()

        # ========================
        # PREPARE OUTPUT FRAMES
        # ========================
        # Prepare frames
        frame_yolo = frame.copy()
        frame_rcnn = frame.copy()
        frame_combined = frame.copy()

        # YOLO: Blue Boxes
        for box, score in zip(yolo_boxes, yolo_scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame_yolo, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Add score text
            cv2.putText(frame_yolo, f"FOD: {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # R-CNN: Red Boxes
        rcnn_filtered_boxes = []
        rcnn_filtered_labels = []
        for box, score, label in zip(rcnn_boxes, rcnn_scores, rcnn_labels):
            if score >= CONF_THRESHOLD_RCNN:
                rcnn_filtered_boxes.append(box)
                rcnn_filtered_labels.append(label)
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame_rcnn, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Add label text
                label_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"Class {label}"
                cv2.putText(frame_rcnn, f"{label_name}: {score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Combined: Green Boxes
        # First add YOLO detections
        for box, score in zip(yolo_boxes, yolo_scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame_combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add score text
            cv2.putText(frame_combined, f"FOD: {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Then add non-overlapping RCNN detections
        for box, label, score in zip(rcnn_filtered_boxes, rcnn_filtered_labels, rcnn_scores):
            overlap = any(iou(box, ybox) > IOU_THRESHOLD for ybox in yolo_boxes)
            if not overlap:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame_combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"Class {label}"
                cv2.putText(frame_combined, f"{label_name}: {score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ========================
        # RUNWAY VISUALIZATION
        # ========================
        if gps_entry:
            # Convert GPS to pixel coordinates
            pixel_x, pixel_y = gps_to_pixel(gps_entry['lat'], gps_entry['lng'], RUNWAY_GPS_CORNERS)

            # Add current position to drone path
            drone_path.append((pixel_x, pixel_y))

            # Draw line between current and previous position
            if last_pixel_x is not None and last_pixel_y is not None:
                runway_draw.line([(last_pixel_x, last_pixel_y), (pixel_x, pixel_y)], fill=(255, 0, 0),
                                 width=8)  # Doubled line width

            # Check if FOD was detected in this frame
            fod_detected = find_fod_in_frame(
                yolo_boxes, rcnn_filtered_boxes, rcnn_scores, rcnn_filtered_labels,
                IOU_THRESHOLD, CONF_THRESHOLD_RCNN
            )

            if fod_detected:
                # Add FOD location to the list
                fod_locations.append((pixel_x, pixel_y))
                # Draw FOD marker on the runway image - with 25% reduced size as requested
                runway_draw.ellipse([(pixel_x - 15, pixel_y - 15), (pixel_x + 15, pixel_y + 15)], fill=(0, 0, 128))

            # Update last position
            last_pixel_x, last_pixel_y = pixel_x, pixel_y

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

    # Split and save the runway image
    total_fod_count = len(fod_locations)
    split_and_save_runway_image(runway_img, drone_path, fod_locations, total_fod_count)

    total_time = time.time() - start_time
    fps_processed = processed / total_time

    print(f"\n===== PROCESSING COMPLETE =====")
    print(f"✅ {processed} frames processed in {total_time:.2f} seconds ({fps_processed:.2f} fps)")
    print(f"✅ Video outputs saved to:")
    print(f"   - {OUTPUT_YOLO}")
    print(f"   - {OUTPUT_RCNN}")
    print(f"   - {OUTPUT_COMBINED}")
    print(f"✅ Runway visualization saved to {OUTPUT_FOLDER}/runway_processed_*.png")
    print(f"✅ Total FOD detected: {total_fod_count}")
    print(f"===== END OF PROCESSING =====\n")


if __name__ == "__main__":
    main()
