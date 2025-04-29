import cv2
import os

#################################################
# CONFIGURATION SETTINGS
#################################################
# Paths
VIDEO_PATH = 'runway_surface_4k.mp4'  # Path to input video
OUTPUT_FOLDER = 'patches_output'  # Folder to save extracted patches

# Patch extraction parameters
WINDOW_SIZE = 1024  # Sliding window size (pixels)
RESIZE_DIM = 512  # Output patch size after resizing
FRAME_INTERVAL = 1  # Seconds between processed frames
STEP = 1024  # Step size for sliding window (no overlap if equal to WINDOW_SIZE)


def main():
    """
    Extract and resize patches from video frames using a sliding window approach
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps)

    print(f"Processing video: {VIDEO_PATH}")
    print(f"FPS: {fps}, Total frames: {frame_count}, Duration: {duration}s")
    print(f"Window size: {WINDOW_SIZE}x{WINDOW_SIZE}, Output size: {RESIZE_DIM}x{RESIZE_DIM}")
    print(f"Processing one frame every {FRAME_INTERVAL} second(s)")

    # Process frames at specified intervals
    for sec in range(0, duration, FRAME_INTERVAL):
        # Set video position to current time
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, frame = cap.read()

        if not success:
            print(f"Could not read frame at {sec}s")
            continue

        # Get frame dimensions
        height, width, _ = frame.shape
        patch_id = 0

        # Extract patches using sliding window
        for y in range(0, height - WINDOW_SIZE + 1, STEP):
            for x in range(0, width - WINDOW_SIZE + 1, STEP):
                # Extract patch
                patch = frame[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                
                # Resize patch to target dimensions
                resized_patch = cv2.resize(patch, (RESIZE_DIM, RESIZE_DIM), interpolation=cv2.INTER_AREA)
                
                # Save patch
                patch_filename = f"{OUTPUT_FOLDER}/frame{sec:04d}_patch{patch_id:04d}.png"
                cv2.imwrite(patch_filename, resized_patch)
                patch_id += 1

        print(f"Processed frame at {sec}s → {patch_id} resized patches")

    # Release video resource
    cap.release()
    print("✅ Processing completed")


if __name__ == "__main__":
    main()
