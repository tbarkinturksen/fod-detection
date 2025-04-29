import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import os
from pathlib import Path

#################################################
# CONFIGURATION SETTINGS
#################################################
# Input/Output paths
INPUT_VIDEO = "runway_video.mp4"  # Path to input video file
CONF_THRESHOLD = 0.45  # Confidence threshold for detection
OUTPUT_VIDEO = f"{CONF_THRESHOLD} output-yolo.mp4"  # Path to output video file
CROP_OUTPUT_DIR = "detected_objects"  # Directory to save cropped objects

# Model settings
MODEL_NAME = "yolov8x.pt"  # YOLOv8 model name
IOU_THRESHOLD = 0.45  # IoU threshold for NMS

# Processing parameters
FRAME_SKIP = 10  # Number of frames to skip for processing

# Device configuration
DEVICE = ""  # Device to run inference on (empty string for auto-detection)


def main():
    """
    Main function to run YOLO object detection on video
    """
    # Create directory for cropped objects if it doesn't exist
    crop_dir = Path(CROP_OUTPUT_DIR)
    crop_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    device = DEVICE if DEVICE else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load YOLOv8 model
    print(f"Loading YOLOv8 model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    model.conf = CONF_THRESHOLD
    model.iou = IOU_THRESHOLD

    # Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video file {INPUT_VIDEO}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps / (FRAME_SKIP + 1), (width, height))

    # Counters and timer
    frame_count = 0
    processed_frames = 0
    start_time = time.time()

    print("Starting object detection...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Display progress periodically
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            progress = (frame_count / total_frames) * 100
            fps_processing = processed_frames / elapsed if elapsed > 0 else 0
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), "
                  f"Processing speed: {fps_processing:.2f} FPS")

        # Process every Nth frame
        if frame_count % (FRAME_SKIP + 1) == 0:
            # Run YOLO detection
            results = model(frame, imgsz=640, augment=False)  # Force inference at 640x640
            result = results[0]
            processed_frames += 1

            # Extract detection information
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            # Save cropped objects from detections
            for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                # Ensure box coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # Skip if box is invalid
                cropped_obj = frame[y1:y2, x1:x2]
                if cropped_obj.size == 0:
                    continue
                    
                # Get class name and save cropped object
                class_name = result.names[class_id]
                crop_filename = f"frame_{frame_count:06d}_{class_name}_{confidence:.2f}_{i}.jpg"
                crop_path = crop_dir / crop_filename
                cv2.imwrite(str(crop_path), cropped_obj)

            # Draw detections on the frame
            annotated_frame = frame.copy()
            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                if confidence < CONF_THRESHOLD:
                    continue  # Skip low-confidence detections
                    
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with class name and confidence
                class_name = result.names[class_id]
                label = f"{class_name}: {confidence:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 10), 
                             (x1 + text_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Add frame information
            cv2.putText(annotated_frame, f"Frame: {frame_count} / Processing every {FRAME_SKIP + 1} frames", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Write frame to output video
            out.write(annotated_frame)

        # Increment frame counter
        frame_count += 1

    # Release resources
    cap.release()
    out.release()

    # Print summary statistics
    elapsed_time = time.time() - start_time
    print("\n✅ Processing complete!")
    print(f"Total frames: {frame_count}")
    print(f"Processed frames: {processed_frames}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Average processing speed: {processed_frames / elapsed_time:.2f} FPS")
    print(f"Output saved to: {OUTPUT_VIDEO}")
    print(f"Cropped objects saved to: {CROP_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
