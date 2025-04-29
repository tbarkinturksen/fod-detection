import cv2
import numpy as np
import os
from datetime import datetime
import argparse

#################################################
# CONFIGURATION SETTINGS
#################################################
# Input/Output paths
INPUT_IMAGE_PATH = "images/test.jpg"  # Path to input image for single frame testing
INPUT_VIDEO_PATH = "runway_video.mp4"  # Path to input video
OUTPUT_DIR_NAME = "images/output"  # Output directory for processed images/video
OUTPUT_VIDEO_NAME = "opencv_result.mp4"  # Name of the output video file

# Processing mode
PROCESSING_MODE = "process_video"  # Options: "debug_single_frame", "process_video"

# Video processing parameters
FRAME_SKIP = 10  # Process every Nth frame (higher values = faster processing, less smooth video)
VIDEO_FPS = 30  # Output video frame rate
DISPLAY_FRAMES = False  # Whether to display frames during video processing
RESIZE_DISPLAY = False  # Whether to resize large frames for display
MAX_DISPLAY_WIDTH = 1200  # Maximum width for display window if resizing

# Detection parameters
MIN_CONTOUR_AREA = 250    # Minimum area for contour detection
CANNY_THRESHOLD_LOW = 60  # Lower threshold for Canny edge detection
CANNY_THRESHOLD_HIGH = 150  # Upper threshold for Canny edge detection
GAUSSIAN_BLUR_KERNEL = (7, 7)  # Gaussian blur kernel size
MIN_BLOB_AREA = 200  # Minimum area for blob detection
MIN_BBOX_WIDTH = 15  # Minimum bounding box width
MIN_BBOX_HEIGHT = 15  # Minimum bounding box height
MORPH_KERNEL_SIZE = 7  # Morphological operations kernel size


def parse_arguments():
    """
    Parse command line arguments to override configuration settings
    
    Returns:
        Dictionary of configuration settings
    """
    parser = argparse.ArgumentParser(description='FOD Detection on Runway Surfaces')
    parser.add_argument('--mode', type=str, choices=['debug_single_frame', 'process_video'],
                        help='Processing mode: debug single frame or process video')
    parser.add_argument('--input', type=str, help='Path to input image or video')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--skip', type=int, help='Number of frames to skip in video processing')
    parser.add_argument('--fps', type=int, help='Output video FPS')
    parser.add_argument('--display', action='store_true', help='Display frames during processing')
    parser.add_argument('--no-display', action='store_false', dest='display',
                        help='Do not display frames during processing')

    # Set defaults for display
    parser.set_defaults(display=DISPLAY_FRAMES)

    args = parser.parse_args()

    # Override configuration settings if command line arguments are provided
    config = {
        'PROCESSING_MODE': PROCESSING_MODE,
        'INPUT_PATH': INPUT_IMAGE_PATH if PROCESSING_MODE == 'debug_single_frame' else INPUT_VIDEO_PATH,
        'OUTPUT_DIR_NAME': OUTPUT_DIR_NAME,
        'FRAME_SKIP': FRAME_SKIP,
        'VIDEO_FPS': VIDEO_FPS,
        'DISPLAY_FRAMES': DISPLAY_FRAMES
    }

    if args.mode:
        config['PROCESSING_MODE'] = args.mode

    if args.input:
        config['INPUT_PATH'] = args.input

    if args.output:
        config['OUTPUT_DIR_NAME'] = args.output

    if args.skip:
        config['FRAME_SKIP'] = args.skip

    if args.fps:
        config['VIDEO_FPS'] = args.fps

    # Use the parsed display value (either from argument or default)
    config['DISPLAY_FRAMES'] = args.display

    return config


def create_output_dir(output_dir_name):
    """
    Create output directory with timestamp if needed
    
    Args:
        output_dir_name: Name or path for the output directory
        
    Returns:
        Path to created output directory
    """
    if output_dir_name:
        output_dir = output_dir_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"fod_detection_results_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def basic_thresholding(image, output_dir=None, save_results=True):
    """
    Apply various thresholding methods to the input image
    
    Args:
        image: Input image
        output_dir: Directory to save thresholding results
        save_results: Whether to save intermediate results
        
    Returns:
        Thresholded image (Otsu's method)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Simple thresholding
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Adaptive thresholding
    adaptive_thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
    adaptive_thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)

    # Otsu's thresholding
    ret, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save results if in debug mode
    if save_results and output_dir:
        cv2.imwrite(f"{output_dir}/1_binary_threshold.jpg", thresh1)
        cv2.imwrite(f"{output_dir}/2_binary_inv_threshold.jpg", thresh2)
        cv2.imwrite(f"{output_dir}/3_adaptive_mean_threshold.jpg", adaptive_thresh1)
        cv2.imwrite(f"{output_dir}/4_adaptive_gaussian_threshold.jpg", adaptive_thresh2)
        cv2.imwrite(f"{output_dir}/5_otsu_threshold.jpg", otsu_thresh)

    return otsu_thresh  # Return Otsu's threshold for further processing


def edge_detection(image, output_dir=None, save_results=True):
    """
    Apply edge detection with noise reduction
    
    Args:
        image: Input image
        output_dir: Directory to save edge detection results
        save_results: Whether to save intermediate results
        
    Returns:
        Edge image (Canny method)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply stronger Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)

    # Apply morphological operations to eliminate small variations
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    # Opening (erosion followed by dilation) removes small objects
    morph_img = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

    # Canny edge detection with higher thresholds
    edges_canny = cv2.Canny(morph_img, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)

    # Apply closing to connect nearby edges and fill small gaps
    edges_canny = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel)

    # Save results if in debug mode
    if save_results and output_dir:
        # Sobel edge detection
        sobelx = cv2.Sobel(morph_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(morph_img, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.convertScaleAbs(sobely)
        sobel_combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

        # Laplacian edge detection
        laplacian = cv2.Laplacian(morph_img, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)

        cv2.imwrite(f"{output_dir}/6_canny_edges.jpg", edges_canny)
        cv2.imwrite(f"{output_dir}/7_sobel_edges.jpg", sobel_combined)
        cv2.imwrite(f"{output_dir}/8_laplacian_edges.jpg", laplacian)

    return edges_canny  # Return Canny edges for further processing


def contour_detection(image, edges, output_dir=None, save_results=True):
    """
    Detect contours from edge image
    
    Args:
        image: Original input image
        edges: Edge image (from edge detection)
        output_dir: Directory to save contour results
        save_results: Whether to save intermediate results
        
    Returns:
        List of filtered contours
    """
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area to reduce noise
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]

    # Draw contours if saving results
    if save_results and output_dir:
        contour_image = image.copy()
        cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f"{output_dir}/9_contours.jpg", contour_image)

    return filtered_contours


def blob_detection(image, output_dir=None, save_results=True):
    """
    Apply blob detection with robust filtering
    
    Args:
        image: Input image
        output_dir: Directory to save blob detection results
        save_results: Whether to save intermediate results
        
    Returns:
        List of detected keypoints
    """
    # Set up the detector with parameters
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds - increase to reduce sensitivity
    params.minThreshold = 30
    params.maxThreshold = 220
    params.thresholdStep = 10

    # Filter by area - significantly increase minimum area
    params.filterByArea = True
    params.minArea = MIN_BLOB_AREA
    params.maxArea = 0.05 * image.shape[0] * image.shape[1]  # Limit max area to 5% of image

    # Filter by circularity - increase to get more circular objects
    params.filterByCircularity = True
    params.minCircularity = 0.4  # Higher value to get more circular objects

    # Filter by convexity - increase for more solid shapes
    params.filterByConvexity = True
    params.minConvexity = 0.7  # Higher value for more solid shapes

    # Filter by inertia - increase to reject elongated shapes
    params.filterByInertia = True
    params.minInertiaRatio = 0.2  # Higher value to reject elongated shapes

    # Filter by color - focus on brighter objects against dark runway
    params.filterByColor = True
    params.blobColor = 255  # 255 for light blobs against dark background

    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing to enhance contrast between objects and background
    gray = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)

    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Detect blobs
    keypoints = detector.detect(enhanced)

    # Save results if in debug mode
    if save_results and output_dir:
        blob_image = image.copy()
        for keypoint in keypoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            r = int(keypoint.size / 2)
            cv2.circle(blob_image, (x, y), r, (0, 0, 255), 2)

        cv2.imwrite(f"{output_dir}/10_blob_detection.jpg", blob_image)
        cv2.imwrite(f"{output_dir}/10a_enhanced_image.jpg", enhanced)

    return keypoints


def watershed_segmentation(image, output_dir=None, save_results=True):
    """
    Apply watershed algorithm for image segmentation
    
    Args:
        image: Input image
        output_dir: Directory to save watershed results
        save_results: Whether to save intermediate results
        
    Returns:
        Watershed markers
    """
    # Convert to grayscale and apply blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)

    # Threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image, markers)

    # Save results if in debug mode
    if save_results and output_dir:
        image_copy = image.copy()
        image_copy[markers == -1] = [0, 0, 255]  # Mark watershed boundaries in red
        cv2.imwrite(f"{output_dir}/11_watershed.jpg", image_copy)

    return markers


def find_fod_bounding_boxes(contours, image):
    """
    Find bounding boxes for FOD objects with filtering for false positives
    
    Args:
        contours: List of contours
        image: Original image (for dimension reference)
        
    Returns:
        List of bounding boxes as (x, y, w, h) tuples
    """
    bounding_boxes = []

    # Calculate image area for relative comparison
    img_height, img_width = image.shape[:2]
    img_area = img_height * img_width

    for cnt in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Get aspect ratio to filter out very elongated shapes (likely cracks)
        aspect_ratio = float(w) / h if h > 0 else 0

        # Filter by multiple criteria
        if (w > MIN_BBOX_WIDTH and
                h > MIN_BBOX_HEIGHT and
                area > MIN_CONTOUR_AREA and
                # Reject very elongated shapes (aspect ratio too high or too low)
                0.25 < aspect_ratio < 4.0 and
                # Reject very large areas (likely shadow or texture changes)
                area < 0.05 * img_area):

            # Calculate solidity (area / convex hull area) to filter irregular shapes
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(cv2.contourArea(cnt)) / hull_area if hull_area > 0 else 0

            # Only keep solid shapes (more likely to be actual objects)
            if solidity > 0.5:
                bounding_boxes.append((x, y, w, h))

    # Remove overlapping boxes
    filtered_boxes = []
    if bounding_boxes:
        # Sort by area (largest first)
        bounding_boxes.sort(key=lambda box: box[2] * box[3], reverse=True)

        while bounding_boxes:
            current_box = bounding_boxes.pop(0)
            filtered_boxes.append(current_box)

            # Remove boxes that significantly overlap with the current box
            remaining_boxes = []
            for box in bounding_boxes:
                # Calculate overlap area
                x1 = max(current_box[0], box[0])
                y1 = max(current_box[1], box[1])
                x2 = min(current_box[0] + current_box[2], box[0] + box[2])
                y2 = min(current_box[1] + current_box[3], box[1] + box[3])

                if x2 > x1 and y2 > y1:
                    overlap_area = (x2 - x1) * (y2 - y1)
                    box_area = box[2] * box[3]
                    # If overlap is less than 50% of the box area, keep it
                    if overlap_area < 0.5 * box_area:
                        remaining_boxes.append(box)
                else:
                    remaining_boxes.append(box)

            bounding_boxes = remaining_boxes

    return filtered_boxes


def draw_bounding_boxes(image, bounding_boxes):
    """
    Draw bounding boxes on image to visualize detections
    
    Args:
        image: Input image
        bounding_boxes: List of bounding boxes as (x, y, w, h) tuples
        
    Returns:
        Image with bounding boxes drawn
    """
    result_image = image.copy()
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result_image, "FOD", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return result_image


def process_frame(frame, save_debug_info=False, output_dir=None):
    """
    Process a single frame with all detection methods
    
    Args:
        frame: Input video frame
        save_debug_info: Whether to save debug information
        output_dir: Directory to save debug information
        
    Returns:
        Tuple of (processed frame, number of detections)
    """
    # Apply thresholding
    threshold_result = basic_thresholding(frame, output_dir if save_debug_info else None, save_debug_info)

    # Apply edge detection
    edges = edge_detection(frame, output_dir if save_debug_info else None, save_debug_info)

    # Detect contours
    contours = contour_detection(frame, edges, output_dir if save_debug_info else None, save_debug_info)

    # Apply blob detection (we don't use the keypoints directly but they help in debugging)
    if save_debug_info:
        keypoints = blob_detection(frame, output_dir, save_debug_info)

    # Apply watershed segmentation (mainly for debugging)
    if save_debug_info:
        markers = watershed_segmentation(frame, output_dir, save_debug_info)

    # Find and draw FOD bounding boxes
    bounding_boxes = find_fod_bounding_boxes(contours, frame)
    result_frame = draw_bounding_boxes(frame, bounding_boxes)

    return result_frame, len(bounding_boxes)


def process_single_frame(input_path, output_dir):
    """
    Process a single image with full debugging output
    
    Args:
        input_path: Path to input image
        output_dir: Directory to save results
    """
    # Read the input image
    image = cv2.imread(input_path)

    if image is None:
        print(f"Error: Could not read image at {input_path}")
        return

    # Save original image
    cv2.imwrite(f"{output_dir}/0_original.jpg", image)

    # Process the frame with debugging enabled
    result_image, num_detections = process_frame(image, save_debug_info=True, output_dir=output_dir)

    # Save final result with bounding boxes
    cv2.imwrite(f"{output_dir}/12_final_fod_detection.jpg", result_image)

    print(f"✅ Processing complete. {num_detections} potential FOD objects detected.")
    print(f"   All results saved to {output_dir}")


def process_video(input_path, output_dir, frame_skip, fps, display_frames=True):
    """
    Process a video with frame skipping, display frames (if enabled), and save result
    
    Args:
        input_path: Path to input video
        output_dir: Directory to save results
        frame_skip: Number of frames to skip between processing
        fps: Output video frame rate
        display_frames: Whether to display frames during processing
    """
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    output_video_path = f"{output_dir}/{OUTPUT_VIDEO_NAME}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Create display window if display is enabled
    if display_frames:
        cv2.namedWindow('FOD Detection', cv2.WINDOW_NORMAL)
        # Resize window to something reasonable if the video is large
        if RESIZE_DISPLAY and (width > MAX_DISPLAY_WIDTH or height > 800):
            display_width = min(MAX_DISPLAY_WIDTH, width)
            display_height = int(height * (display_width / width))
            cv2.resizeWindow('FOD Detection', display_width, display_height)

    # Counters for progress tracking
    frame_count = 0
    processed_count = 0
    total_detections = 0

    print(f"Processing video with {total_frames} frames, saving every {frame_skip} frame...")
    if display_frames:
        print("Press 'q' to stop processing, 'p' to pause/resume")

    paused = False

    while cap.isOpened():
        # Check for key presses (non-blocking) if display is enabled
        if display_frames:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nProcessing stopped by user.")
                break
            elif key == ord('p'):
                paused = not paused
                if paused:
                    print("\nProcessing paused. Press 'p' to resume.")
                else:
                    print("\nProcessing resumed.")

            # If paused, continue to check for key presses but don't process frames
            if paused:
                continue

        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_count % frame_skip == 0:
            # Display progress
            progress = (frame_count / total_frames) * 100
            print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)...", end="\r")

            # Process the frame
            result_frame, num_detections = process_frame(frame, save_debug_info=False)
            total_detections += num_detections
            processed_count += 1

            # Add detection count to the frame
            info_text = f"Frame: {frame_count}, FOD: {num_detections}"
            cv2.putText(result_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display the processed frame if display is enabled
            if display_frames:
                cv2.imshow('FOD Detection', result_frame)

            # Write the processed frame to the output video
            out.write(result_frame)

        frame_count += 1

    # Release resources
    cap.release()
    out.release()
    if display_frames:
        cv2.destroyAllWindows()

    print(f"\n✅ Video processing complete.")
    print(f"   Processed {processed_count} frames out of {total_frames} total frames.")
    print(f"   Detected {total_detections} potential FOD objects in total.")
    print(f"   Output video saved to {output_video_path}")


def main():
    """
    Main function to run the FOD detection pipeline
    """
    # Parse command line arguments
    config = parse_arguments()

    # Create output directory
    output_dir = create_output_dir(config['OUTPUT_DIR_NAME'])
    print(f"Saving results to {output_dir}")

    # Process based on selected mode
    if config['PROCESSING_MODE'] == 'debug_single_frame':
        print("Running in single frame debug mode")
        process_single_frame(config['INPUT_PATH'], output_dir)
    else:  # process_video mode
        print("Running in video processing mode")
        print(f"Display frames: {'Enabled' if config['DISPLAY_FRAMES'] else 'Disabled'}")
        process_video(
            config['INPUT_PATH'],
            output_dir,
            config['FRAME_SKIP'],
            config['VIDEO_FPS'],
            config['DISPLAY_FRAMES']
        )


if __name__ == "__main__":
    main()
