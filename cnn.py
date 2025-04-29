import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import os

#################################################
# CONFIGURATION SETTINGS
#################################################
# Model configuration
USE_CUSTOM_MODEL = False  # Set to False to use COCO pretrained model
CUSTOM_MODEL_PATH = "fasterrcnn_fod.pth"  # Path to custom trained model

# Input/Output paths
INPUT_VIDEO_PATH = "runway_video.mp4"  # Path to input video
OUTPUT_VIDEO_PATH = "output_fasterrcnn.mp4"  # Path to output video

# Detection parameters
CONFIDENCE_THRESHOLD = 0.15  # Min confidence score for detection
FRAME_SKIP = 10  # Process every Nth frame

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Load the appropriate model
if USE_CUSTOM_MODEL:
    print("Loading custom model...")
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 2  # background + FOD
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(CUSTOM_MODEL_PATH, map_location=DEVICE))
    class_names = ["__background__", "FOD"]
else:
    print("Loading default COCO-pretrained model...")
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # COCO class names
    class_names = [
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

# Set model to evaluation mode and move to device
model.to(DEVICE)
model.eval()


def main():
    """
    Main function to run object detection on video
    """
    # Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps / (FRAME_SKIP + 1), 
                          (frame_width, frame_height))

    frame_count = 0

    print(f"Input video: {INPUT_VIDEO_PATH}")
    print(f"Output video: {OUTPUT_VIDEO_PATH}")
    print(f"Total frames: {total_frames}, Skipping every {FRAME_SKIP} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # Convert frame to tensor for model input
        image_tensor = F.to_tensor(frame).to(DEVICE).unsqueeze(0)

        # Run detection
        with torch.no_grad():
            predictions = model(image_tensor)

        # Extract predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # Apply confidence threshold
        valid_indices = scores > CONFIDENCE_THRESHOLD
        boxes = boxes[valid_indices]
        labels = labels[valid_indices]

        # Draw detections on the frame
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write the processed frame to output video
        out.write(frame)

        # Display progress
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.2f}% ({frame_count}/{total_frames} frames)")

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("✅ Video processing completed.")
    print("Saved to:", OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    main()
