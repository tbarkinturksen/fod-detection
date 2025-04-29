import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import argparse
from pathlib import Path

# ========================
# CONFIGURATION SETTINGS
# ========================
INPUT_FOLDER = 'input'  # Input folder for source files
CLEAN_PATCHES_FOLDER = os.path.join(INPUT_FOLDER, 'clean_patches')  # Folder with clean runway patches
AUTOENCODER_MODEL_PATH = os.path.join(INPUT_FOLDER, 'ir_autoencoder.pth')  # Path to save autoencoder model

# Autoencoder settings
AUTOENCODER_CROP_SIZE = 128  # Size to resize detection crops for autoencoder
AUTOENCODER_BATCH_SIZE = 32
AUTOENCODER_EPOCHS = 20
AUTOENCODER_LEARNING_RATE = 0.001

# GPU settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


# ========================
# DATASET FOR AUTOENCODER
# ========================
class RunwayPatchesDataset(Dataset):
    """Dataset for clean runway patches to train the autoencoder."""

    def __init__(self, folder_path, transform=None):
        self.image_paths = glob.glob(os.path.join(folder_path, "*.png")) + \
                           glob.glob(os.path.join(folder_path, "*.jpg"))
        self.transform = transform
        print(f"Found {len(self.image_paths)} clean runway patch images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load as grayscale for IR images
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize to expected size
        image = cv2.resize(image, (AUTOENCODER_CROP_SIZE, AUTOENCODER_CROP_SIZE))

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor and normalize
            image = torch.from_numpy(image).float() / 255.0
            image = image.unsqueeze(0)  # Add channel dimension

        return image


# ========================
# TRAINING FUNCTION
# ========================
def train_autoencoder(epochs=None, batch_size=None, learning_rate=None):
    """Train the autoencoder on clean runway patches."""
    print("\n===== TRAINING AUTOENCODER FOR FALSE POSITIVE FILTERING =====")

    # Use provided parameters or defaults
    epochs = epochs or AUTOENCODER_EPOCHS
    batch_size = batch_size or AUTOENCODER_BATCH_SIZE
    learning_rate = learning_rate or AUTOENCODER_LEARNING_RATE

    # Check if clean patches folder exists and contains images
    if not os.path.exists(CLEAN_PATCHES_FOLDER):
        print(f"❌ Clean patches folder not found: {CLEAN_PATCHES_FOLDER}")
        print(f"Please create this folder and add clean runway patch images.")
        return None

    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = RunwayPatchesDataset(CLEAN_PATCHES_FOLDER, transform=None)

    if len(dataset) < 10:
        print(f"❌ Not enough clean patches found ({len(dataset)}). Need at least 10 for training.")
        return None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Create model, loss function, and optimizer
    model = IRAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Print model summary
    print(f"\nAutoencoder Model Architecture:")
    print(f"- Input size: {AUTOENCODER_CROP_SIZE}x{AUTOENCODER_CROP_SIZE}x1")
    print(f"- Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"- Device: {DEVICE}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Epochs: {epochs}")

    # Training loop
    best_loss = float('inf')

    print(f"\nStarting training with {len(dataset)} images...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_idx, data in enumerate(dataloader):
            data = data.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, data)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch + 1}/{epochs} | Batch: {batch_idx}/{len(dataloader)} | Loss: {loss.item():.6f}")

        avg_loss = train_loss / len(dataloader)
        print(f"Epoch: {epoch + 1}/{epochs} completed | Avg Loss: {avg_loss:.6f}")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), AUTOENCODER_MODEL_PATH)
            print(f"Model saved with loss: {best_loss:.6f}")

    print(f"\nAutoencoder training completed. Best loss: {best_loss:.6f}")
    print(f"Model saved to: {AUTOENCODER_MODEL_PATH}")

    return model


# ========================
# TESTING FUNCTION
# ========================
def test_autoencoder():
    """Load the trained model and test on a few samples."""
    print("\n===== TESTING AUTOENCODER =====")

    # Check if model exists
    if not os.path.exists(AUTOENCODER_MODEL_PATH):
        print(f"❌ Trained model not found: {AUTOENCODER_MODEL_PATH}")
        return

    # Load model
    model = IRAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(AUTOENCODER_MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Create a small test dataset (using the same dataset for simplicity)
    dataset = RunwayPatchesDataset(CLEAN_PATCHES_FOLDER, transform=None)

    if len(dataset) == 0:
        print("No test images available.")
        return

    # Create output directory for visualizations
    test_output_dir = os.path.join("output", "autoencoder_test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Test on a few random samples
    n_samples = min(5, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            sample = dataset[idx].to(DEVICE)

            # Get reconstruction
            reconstruction = model(sample.unsqueeze(0)).squeeze(0)

            # Calculate error
            error = torch.mean((reconstruction - sample) ** 2).item()

            # Convert tensors to numpy for visualization
            sample_np = (sample.cpu().numpy()[0] * 255).astype(np.uint8)
            recon_np = (reconstruction.cpu().numpy()[0] * 255).astype(np.uint8)

            # Create side-by-side comparison
            comparison = np.hstack((sample_np, recon_np))

            # Add a title with error
            title = f"Original | Reconstruction (Error: {error:.6f})"
            img_with_title = np.ones((30 + comparison.shape[0], comparison.shape[1]), dtype=np.uint8) * 255
            img_with_title[30:, :] = comparison

            cv2.putText(img_with_title, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

            # Save the comparison
            output_path = os.path.join(test_output_dir, f"test_sample_{i + 1}.png")
            cv2.imwrite(output_path, img_with_title)

            print(f"Sample {i + 1}: Reconstruction error = {error:.6f}")

    print(f"\nTest visualizations saved to {test_output_dir}")


# ========================
# MAIN FUNCTION
# ========================
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Train IR Autoencoder for FOD Detection')
    parser.add_argument('--epochs', type=int, default=AUTOENCODER_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=AUTOENCODER_BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=AUTOENCODER_LEARNING_RATE, help='Learning rate')
    parser.add_argument('--test', action='store_true', help='Test the trained model after training')
    parser.add_argument('--test_only', action='store_true', help='Only test the model, no training')
    parser.add_argument('--input_dir', type=str, default=CLEAN_PATCHES_FOLDER,
                        help='Directory with clean runway patches')
    parser.add_argument('--model_path', type=str, default=AUTOENCODER_MODEL_PATH, help='Path to save/load the model')

    args = parser.parse_args()

    # Update paths if custom ones provided
    if args.input_dir != CLEAN_PATCHES_FOLDER:
        CLEAN_PATCHES_FOLDER = args.input_dir

    if args.model_path != AUTOENCODER_MODEL_PATH:
        AUTOENCODER_MODEL_PATH = args.model_path

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(AUTOENCODER_MODEL_PATH), exist_ok=True)

    # Print configuration
    print("IR Autoencoder Training Tool")
    print(f"Clean patches folder: {CLEAN_PATCHES_FOLDER}")
    print(f"Model path: {AUTOENCODER_MODEL_PATH}")
    print(f"Device: {DEVICE}")

    # Run training or testing based on arguments
    if args.test_only:
        test_autoencoder()
    else:
        train_autoencoder(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
        if args.test:
            test_autoencoder()
