import cv2
import os

#################################################
# CONFIGURATION SETTINGS
#################################################
# Paths
VIDEO_PATH = "data/drone_footage.mp4"  # Path to input video
OUTPUT_FOLDER = "extracted"  # Folder to save extracted frames

# Extraction parameters
FRAME_INTERVAL = 1  # Extract one frame per second


def main():
    """
    Main function to extract frames from a video at specified intervals
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps  # Video duration in seconds

    print(f"Video FPS: {fps}, Total Frames: {frame_count}, Duration: {duration:.2f} seconds")
    print(f"Extracting one frame every {FRAME_INTERVAL} second(s)...")

    frame_idx = 0
    saved_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame if it's at the correct interval
        if frame_idx % (fps * FRAME_INTERVAL) == 0:
            frame_filename = os.path.join(OUTPUT_FOLDER, f"frame_{saved_frames:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1

        frame_idx += 1

    cap.release()
    print(f"✅ Extracted {saved_frames} frames and saved in '{OUTPUT_FOLDER}'")


if __name__ == "__main__":
    main()
