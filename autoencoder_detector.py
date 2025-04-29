import cv2
import numpy as np
import os
import time
import torch
from torch import nn
from torchvision import transforms

#################################################
# CONFIGURATION SETTINGS
#################################################
# Input/Output paths
INPUT_VIDEO_PATH = "runway_video.mp4"  # Path to input video
OUTPUT_VIDEO_PATH = "ae_output.mp4"  # Path to output video
MODEL_PATH = "autoencoder_model.pth"  # Path to trained autoencoder model

# Detection parameters
THRESHOLD = 0.1  # Reconstruction error threshold for FOD detection
FRAME_SKIP = 10  # Process every nth frame (1 = no skip, 2 = every other frame, etc.)
WINDOW_SIZE = 512  # Size of the sliding window
WINDOW_STRIDE = 256  # Step size for the sliding window
DISPLAY_RESULTS = False  # Show real-time processing visualization

# Device configuration
USE_CUDA = torch.cuda.is_available()  # Use GPU if available


# Define the Autoencoder architecture (must match the trained model)
class Autoencoder(nn.Module):
    """Autoencoder architecture that matches the saved model"""

    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder with 9 layers (0-8)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 0
            nn.ReLU(),  # 1
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 2
            nn.ReLU(),  # 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 4
            nn.ReLU(),  # 5
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 6
            nn.ReLU(),  # 7
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)  # 8
        )

        # Decoder with 9 layers (0-8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 0
            nn.ReLU(),  # 1
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 2
            nn.ReLU(),  # 3
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4
            nn.ReLU(),  # 5
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 6
            nn.ReLU(),  # 7
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8
            nn.Sigmoid()  # 9
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_model(model_path):
    """
    Load the pre-trained PyTorch autoencoder model
    
    Args:
        model_path: Path to the saved model state dict
        
    Returns:
        Loaded model in evaluation mode
    """
    # Create the model instance
    model = Autoencoder()

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location='cuda' if USE_CUDA else 'cpu')

    # Handle the case where the state_dict might have "module." prefix due to DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Move to GPU if available
    if USE_CUDA:
        model = model.cuda()

    # Set to evaluation mode
    model.eval()
    return model


def preprocess_window(window):
    """
    Preprocess the window for the autoencoder
    
    Args:
        window: Image window to preprocess
        
    Returns:
        Tensor ready for model input
    """
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add any other transformations your model was trained with
    ])

    # Apply transformation
    tensor = transform(window)

    # Move to GPU if available
    if USE_CUDA:
        tensor = tensor.cuda()

    return tensor


def calculate_reconstruction_error(original, reconstruction):
    """
    Calculate the reconstruction error between original window and its reconstruction
    
    Args:
        original: Original image tensor
        reconstruction: Reconstructed image tensor
        
    Returns:
        Mean squared error as a float
    """
    # Convert tensors to numpy if they're on GPU
    if original.is_cuda:
        original = original.cpu()
    if reconstruction.is_cuda:
        reconstruction = reconstruction.cpu()

    # Convert to numpy arrays
    original = original.detach().numpy()
    reconstruction = reconstruction.detach().numpy()

    # Calculate mean squared error
    return np.mean(np.square(original - reconstruction))


def process_frame_with_sliding_window(frame, model, threshold):
    """
    Process a frame with sliding window approach
    
    Args:
        frame: Input video frame
        model: Trained autoencoder model
        threshold: Error threshold for FOD detection
        
    Returns:
        Frame with detection visualizations
    """
    height, width = frame.shape[:2]
    result_frame = frame.copy()

    # Calculate valid window positions
    x_positions = range(0, width - WINDOW_SIZE + 1, WINDOW_STRIDE)
    y_positions = range(0, height - WINDOW_SIZE + 1, WINDOW_STRIDE)

    # If there are no valid positions (frame smaller than window), return original frame
    if not x_positions or not y_positions:
        return frame

    for y in y_positions:
        for x in x_positions:
            # Extract window
            window = frame[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE]

            # Skip if window is not the expected size
            if window.shape[:2] != (WINDOW_SIZE, WINDOW_SIZE):
                continue

            # Preprocess window
            input_tensor = preprocess_window(window)

            # Forward pass through the model
            with torch.no_grad():
                reconstruction = model(input_tensor.unsqueeze(0))

            # Calculate error
            error = calculate_reconstruction_error(input_tensor, reconstruction.squeeze(0))

            # Determine color based on threshold
            color = (0, 0, 255) if error > threshold else (255, 0, 0)  # Red if FOD, Blue if clean

            # Draw rectangle
            cv2.rectangle(result_frame, (x, y), (x + WINDOW_SIZE, y + WINDOW_SIZE), color, 2)

            # Optionally, add text with error value
            cv2.putText(result_frame, f"{error:.4f}", (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return result_frame


def main():
    """
    Main function to run the FOD detection pipeline
    """
    # Start timer
    start_time = time.time()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_VIDEO_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the model
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully. Using {'CUDA' if USE_CUDA else 'CPU'} for inference.")

    # Open the video file
    print(f"Opening video: {INPUT_VIDEO_PATH}")
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {INPUT_VIDEO_PATH}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Print video info
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")

    # Calculate new fps with frame skipping
    output_fps = fps / FRAME_SKIP

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, output_fps, (width, height))

    frame_count = 0
    processed_frames = 0

    print(f"Processing video with frame skip = {FRAME_SKIP}...")
    print(f"FOD detection threshold = {THRESHOLD}")

    while True:
        ret, frame = cap.read()

        # Break the loop if frame reading failed (end of video)
        if not ret:
            break

        frame_count += 1

        # Skip frames according to FRAME_SKIP
        if (frame_count - 1) % FRAME_SKIP != 0:
            continue

        processed_frames += 1

        # Process the frame
        processed_frame = process_frame_with_sliding_window(frame, model, THRESHOLD)

        # Write the processed frame
        out.write(processed_frame)

        # Display the processed frame
        if DISPLAY_RESULTS:
            # Resize for display if too large
            display_frame = processed_frame
            if width > 1280 or height > 720:
                scale = min(1280 / width, 720 / height)
                display_frame = cv2.resize(processed_frame,
                                           (int(width * scale), int(height * scale)))

            cv2.imshow('FOD Detection', display_frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Print progress every 10 frames
        if processed_frames % 10 == 0:
            elapsed_time = time.time() - start_time
            frames_per_second = processed_frames / elapsed_time
            estimated_total_time = total_frames / FRAME_SKIP / frames_per_second
            remaining_time = estimated_total_time - elapsed_time

            print(f"Processed {processed_frames}/{total_frames // FRAME_SKIP} frames "
                  f"({100 * processed_frames / (total_frames / FRAME_SKIP):.1f}%) | "
                  f"Speed: {frames_per_second:.2f} fps | "
                  f"Remaining: {remaining_time / 60:.1f} minutes")

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print final stats
    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    print(f"Processed {processed_frames} frames at {processed_frames / elapsed_time:.2f} fps")
    print(f"Output saved to: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
