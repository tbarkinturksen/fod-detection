import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

#################################################
# CONFIGURATION SETTINGS
#################################################
# Paths
MODEL_PATH = 'autoencoder_model.pth'  # Path to trained model weights
TEST_IMAGE_NO_FOD = './data/test_no_fod.jpg'  # Test image without FOD
TEST_IMAGE_WITH_FOD = './data/test_with_fod.jpg'  # Test image with FOD

# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Define the Autoencoder neural network architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (B, 3, 512, 512) -> (B, 64, 256, 256)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 64, 256, 256) -> (B, 128, 128, 128)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (B, 128, 128, 128) -> (B, 256, 64, 64)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # (B, 256, 64, 64) -> (B, 512, 32, 32)
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # (B, 512, 32, 32) -> (B, 1024, 16, 16)
            nn.ReLU()
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (B, 1024, 16, 16) -> (B, 512, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (B, 512, 32, 32) -> (B, 256, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (B, 256, 64, 64) -> (B, 128, 128, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (B, 128, 128, 128) -> (B, 64, 256, 256)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (B, 64, 256, 256) -> (B, 3, 512, 512)
            nn.Sigmoid()  # Sigmoid activation to get output in range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Function to crop the center of a 1920x1080 image and then resize it to 512x512
def crop_and_resize(image_path, target_size=(512, 512)):
    """
    Crops the center portion of an image and resizes it to the target size
    
    Args:
        image_path: Path to the input image
        target_size: Desired output dimensions (width, height)
        
    Returns:
        Cropped and resized PIL Image
    """
    # Load the image
    img = Image.open(image_path).convert("RGB")

    # Crop the center 1080x1080
    width, height = img.size
    left = (width - 1080) // 2
    top = (height - 1080) // 2
    right = (width + 1080) // 2
    bottom = (height + 1080) // 2

    # Crop the image
    img_cropped = img.crop((left, top, right, bottom))

    # Resize to target size
    img_resized = img_cropped.resize(target_size)

    return img_resized


# Function to compute MSE between two images
def compute_mse(original_img, regenerated_img):
    """
    Computes Mean Squared Error between original and regenerated images
    
    Args:
        original_img: Original image as numpy array (0-255 range)
        regenerated_img: Regenerated image as numpy array (0-1 range)
        
    Returns:
        MSE value
    """
    return np.mean((original_img / 255.0 - regenerated_img) ** 2)


# Test function for regenerating frames
def test_frame_regeneration(test_image_path, model, transform):
    """
    Tests the autoencoder by regenerating an image and computing the reconstruction error
    
    Args:
        test_image_path: Path to the test image
        model: Trained autoencoder model
        transform: Preprocessing transforms to apply
        
    Returns:
        Tuple of (original image, regenerated image, MSE value)
    """
    # Preprocess image: crop center and resize to 512x512
    img_resized = crop_and_resize(test_image_path, target_size=(512, 512))

    # Convert to tensor and add batch dimension
    img_tensor = transform(img_resized).unsqueeze(0)

    # Pass the image through the autoencoder to regenerate it
    with torch.no_grad():
        regenerated_img_tensor = model(img_tensor)

    # Convert the tensors back to images
    original_img = np.array(img_resized)
    regenerated_img = regenerated_img_tensor.squeeze(0).permute(1, 2, 0).numpy()  # Remove batch dimension and convert to numpy

    # Compute Mean Squared Error (MSE)
    mse = compute_mse(original_img, regenerated_img)
    print(f"Mean Squared Error between original and regenerated image: {mse:.6f}")

    return original_img, regenerated_img, mse


def main():
    # Load the trained autoencoder model
    model = Autoencoder()
    model.load_state_dict(torch.load(MODEL_PATH))  # Load the saved model weights
    model.eval()  # Set the model to evaluation mode

    # Define the transform to resize the test image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Test frame regeneration for both images
    original_no_fod, regenerated_no_fod, mse_no_fod = test_frame_regeneration(TEST_IMAGE_NO_FOD, model, transform)
    original_with_fod, regenerated_with_fod, mse_with_fod = test_frame_regeneration(TEST_IMAGE_WITH_FOD, model, transform)

    # Plot the 2x2 grid for both images with their corresponding MSE below regenerated images
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original image without FOD
    axes[0, 0].imshow(original_no_fod)
    axes[0, 0].set_title("Original Image (No FOD)")
    axes[0, 0].axis('off')

    # Regenerated image without FOD
    axes[0, 1].imshow(regenerated_no_fod)
    axes[0, 1].set_title(f"Regenerated Image (No FOD)\nMSE: {mse_no_fod:.6f}")
    axes[0, 1].axis('off')

    # Original image with FOD
    axes[1, 0].imshow(original_with_fod)
    axes[1, 0].set_title("Original Image (With FOD)")
    axes[1, 0].axis('off')

    # Regenerated image with FOD
    axes[1, 1].imshow(regenerated_with_fod)
    axes[1, 1].set_title(f"Regenerated Image (With FOD)\nMSE: {mse_with_fod:.6f}")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
