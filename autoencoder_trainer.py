import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from multiprocessing import freeze_support

#################################################
# CONFIGURATION SETTINGS
#################################################
# Training parameters
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_WORKERS = 4

# Paths
TRAIN_DIR = './data/train_images'  # Path to training images 
MODEL_SAVE_PATH = 'autoencoder_model.pth'  # Where to save the trained model


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


# Define the dataset class for loading images
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Dataset for loading images for autoencoder training
        
        Args:
            image_dir: Directory containing training images
            transform: Preprocessing transforms to apply
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if
                            fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img


# Define the main training function
def train_autoencoder():
    """
    Trains the autoencoder model on the dataset
    """
    # Set the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the image transformations (resize to 512x512)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
    ])

    # Set up dataset and dataloaders
    train_dataset = ImageDataset(TRAIN_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"Dataset loaded with {len(train_dataset)} images")

    # Initialize the Autoencoder model
    model = Autoencoder().to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            # Forward pass
            output = model(batch)

            # Compute loss
            loss = criterion(output, batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    freeze_support()  # Ensure compatibility for Windows
    train_autoencoder()
