import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import random
import cv2

#################################################
# CONFIGURATION SETTINGS
#################################################
# Paths
NORMAL_TEXTURES_DIR = "input/runway_textures"  # Directory with normal runway textures
OUTPUT_MODEL_PATH = "output/small_texture_ae.pth"  # Where to save the trained model
OUTPUT_DIR = "output/autoencoder"  # Directory for visualization outputs

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
VALID_SPLIT = 0.2  # Validation split ratio
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Patch extraction parameters
SMALL_PATCH_SIZE = 64  # Size of small patches to extract from each larger image
PATCHES_PER_IMAGE = 20  # Number of small patches to extract from each training image
MIN_PATCH_SIZE = 32  # Minimum patch size for training diversity
MAX_PATCH_SIZE = 96  # Maximum patch size for training diversity
FINAL_SIZE = 64  # Final size all patches will be resized to


#################################################
# AUTOENCODER MODEL DEFINITION
#################################################
class SmallTextureAutoencoder(nn.Module):
    """
    Autoencoder for small runway texture patches.
    Designed specifically for small FOD detection.
    """

    def __init__(self):
        super(SmallTextureAutoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            # Input: 3x64x64
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 32x32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128x8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 256x8x8
            nn.ReLU(inplace=True)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            # Input: 256x8x8
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),  # 128x8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 3x64x64
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


#################################################
# DATASET CLASS WITH SMALL PATCH EXTRACTION
#################################################
class SmallPatchDataset(Dataset):
    """
    Dataset class for extracting and loading small runway texture patches
    """

    def __init__(self, image_dirs, patch_size=SMALL_PATCH_SIZE, patches_per_image=PATCHES_PER_IMAGE,
                 min_size=MIN_PATCH_SIZE, max_size=MAX_PATCH_SIZE, final_size=FINAL_SIZE, transform=None):
        """
        Args:
            image_dirs: List of directories containing images
            patch_size: Base size of patches to extract
            patches_per_image: Number of patches to extract from each image
            min_size/max_size: Range of patch sizes for diversity
            final_size: Final size all patches will be resized to
            transform: Optional transform to be applied on a patch
        """
        self.transform = transform
        self.patches_per_image = patches_per_image
        self.min_size = min_size
        self.max_size = max_size
        self.final_size = final_size

        # Gather all image paths
        self.image_paths = []
        for directory in image_dirs:
            if os.path.isdir(directory):
                self.image_paths.extend(glob.glob(os.path.join(directory, "**/*.jpg"), recursive=True))
                self.image_paths.extend(glob.glob(os.path.join(directory, "**/*.png"), recursive=True))
                self.image_paths.extend(glob.glob(os.path.join(directory, "**/*.jpeg"), recursive=True))

        print(f"Found {len(self.image_paths)} source images for patch extraction")

        # Extract small patches from all images
        self.patches = self._extract_patches()
        print(f"Extracted {len(self.patches)} small patches for training")

    def _extract_patches(self):
        """Extract small patches from all training images"""
        patches = []

        for image_path in tqdm(self.image_paths, desc="Extracting patches"):
            try:
                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                height, width = img.shape[:2]

                # Extract multiple patches of varying sizes
                for _ in range(self.patches_per_image):
                    # Randomly select patch size
                    size = random.randint(self.min_size, self.max_size)

                    # Make sure patch fits within image
                    if size >= min(height, width):
                        size = min(height, width) - 1

                    # Random position
                    x = random.randint(0, width - size)
                    y = random.randint(0, height - size)

                    # Extract patch
                    patch = img[y:y + size, x:x + size]

                    # Resize to final size
                    patch = cv2.resize(patch, (self.final_size, self.final_size))

                    # Convert to PIL image
                    patch = Image.fromarray(patch)
                    patches.append(patch)

            except Exception as e:
                print(f"Error extracting patches from {image_path}: {e}")

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]

        # Apply transform if provided
        if self.transform:
            patch = self.transform(patch)

        return patch


#################################################
# TRAINING FUNCTIONS
#################################################
def create_data_loaders():
    """
    Create data loaders for training and validation

    Returns:
        train_loader, valid_loader
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = SmallPatchDataset([NORMAL_TEXTURES_DIR], transform=transform)

    # Split into training and validation sets
    dataset_size = len(dataset)
    valid_size = int(VALID_SPLIT * dataset_size)
    train_size = dataset_size - valid_size

    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    print(f"Dataset created with {dataset_size} patches")
    print(f"Training set: {train_size} patches")
    print(f"Validation set: {valid_size} patches")

    return train_loader, valid_loader


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch

    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use

    Returns:
        Average training loss
    """
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        # Move batch to device
        inputs = batch.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Validate the model

    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use

    Returns:
        Average validation loss
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move batch to device
            inputs = batch.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def plot_reconstructions(model, dataloader, device, num_images=10, epoch=0):
    """
    Plot original and reconstructed images

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device
        num_images: Number of images to plot
        epoch: Current epoch number for filename
    """
    model.eval()

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with torch.no_grad():
        # Get a batch of images
        images = next(iter(dataloader))

        # Limit to num_images
        images = images[:num_images].to(device)

        # Generate reconstructions
        reconstructions = model(images)

        # Move tensors to CPU and convert to numpy arrays
        images = images.cpu().numpy()
        reconstructions = reconstructions.cpu().numpy()

        # Plot original and reconstructed images side by side
        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 2.5))

        for i in range(num_images):
            # Original image
            orig_img = np.transpose(images[i], (1, 2, 0))
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            # Reconstructed image
            recon_img = np.transpose(reconstructions[i], (1, 2, 0))
            axes[i, 1].imshow(recon_img)
            axes[i, 1].set_title(f"Reconstructed (MSE: {np.mean((orig_img - recon_img) ** 2):.4f})")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/small_patch_reconstructions_epoch_{epoch}.png")
        plt.close()


def compute_mse_distribution(model, dataloader, device):
    """
    Compute MSE distribution for the dataset

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device

    Returns:
        List of MSE values
    """
    model.eval()
    mse_values = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing MSE distribution"):
            inputs = batch.to(device)
            outputs = model(inputs)

            # Calculate MSE per image
            for i in range(inputs.size(0)):
                input_img = inputs[i].cpu().numpy()
                output_img = outputs[i].cpu().numpy()
                mse = np.mean((input_img - output_img) ** 2)
                mse_values.append(mse)

    return mse_values


def plot_mse_histogram(mse_values, epoch=0):
    """
    Plot histogram of MSE values

    Args:
        mse_values: List of MSE values
        epoch: Current epoch number for filename
    """
    plt.figure(figsize=(10, 6))
    plt.hist(mse_values, bins=50, alpha=0.75)
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Reconstruction Errors for Small Patches")

    # Calculate statistics
    mean_mse = np.mean(mse_values)
    median_mse = np.median(mse_values)
    std_mse = np.std(mse_values)

    # Add statistics to plot
    plt.axvline(mean_mse, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_mse:.4f}')
    plt.axvline(mean_mse + 2 * std_mse, color='g', linestyle='dashed', linewidth=1,
                label=f'Mean + 2*Std: {mean_mse + 2 * std_mse:.4f}')

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/small_patch_mse_histogram_epoch_{epoch}.png")
    plt.close()

    # Print statistics
    print(f"MSE Statistics:")
    print(f"  - Mean: {mean_mse:.6f}")
    print(f"  - Median: {median_mse:.6f}")
    print(f"  - Std Dev: {std_mse:.6f}")
    print(f"  - Suggested threshold (Mean + 2*Std): {mean_mse + 2 * std_mse:.6f}")

    return mean_mse, std_mse


#################################################
# MAIN TRAINING FUNCTION
#################################################
def train_autoencoder():
    """
    Main function to train the autoencoder
    """
    print("\n===== SMALL PATCH TEXTURE AUTOENCODER TRAINING =====")
    print(f"Training on small patches sized {FINAL_SIZE}x{FINAL_SIZE} pixels")
    print(
        f"Extracting {PATCHES_PER_IMAGE} patches per image with sizes ranging from {MIN_PATCH_SIZE} to {MAX_PATCH_SIZE}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create data loaders
    train_loader, valid_loader = create_data_loaders()

    # Initialize model
    model = SmallTextureAutoencoder().to(DEVICE)
    print(f"Model created and moved to {DEVICE}")

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize variables for early stopping
    best_valid_loss = float('inf')
    patience = 5
    patience_counter = 0

    # Training loop
    train_losses = []
    valid_losses = []

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Train and validate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        valid_loss = validate(model, valid_loader, criterion, DEVICE)

        # Store losses
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Print progress
        print(f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}")

        # Plot reconstructions every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            plot_reconstructions(model, valid_loader, DEVICE, num_images=5, epoch=epoch + 1)

        # Update best model if improved
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            patience_counter = 0
            print(f"✅ Saved new best model with validation loss: {valid_loss:.6f}")
        else:
            patience_counter += 1
            print(f"⚠️ Validation loss did not improve. Patience: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("\nTraining completed.")

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/small_patch_loss_curve.png")
    plt.close()

    # Load best model
    model.load_state_dict(torch.load(OUTPUT_MODEL_PATH))

    # Compute MSE distribution
    print("\nComputing MSE distribution on validation set...")
    mse_values = compute_mse_distribution(model, valid_loader, DEVICE)
    mean_mse, std_mse = plot_mse_histogram(mse_values)

    # Save MSE statistics
    with open(f"{OUTPUT_DIR}/small_patch_mse_statistics.txt", "w") as f:
        f.write(f"Mean MSE: {mean_mse}\n")
        f.write(f"Standard Deviation: {std_mse}\n")
        f.write(f"Suggested Threshold (Mean + 2*Std): {mean_mse + 2 * std_mse}\n")

    print(f"\n✅ Small patch texture autoencoder trained and saved to {OUTPUT_MODEL_PATH}")
    print(f"✅ Results and visualizations saved to {OUTPUT_DIR}")
    print(f"✅ Suggested threshold for FOD detection: {mean_mse + 2 * std_mse:.6f}")


if __name__ == "__main__":
    train_autoencoder()
