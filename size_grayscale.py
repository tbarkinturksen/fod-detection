import os
from PIL import Image

#################################################
# CONFIGURATION SETTINGS
#################################################
# Paths
SOURCE_DIR = "/Volumes/BT EXTERNAL/TEZ/FOOTAGE/IR/IR FOD Square SD Enhanced/FOD"  # Source folder path
OUTPUT_DIR = "/Volumes/BT EXTERNAL/TEZ/FOOTAGE/IR/FOD Patches"  # Output folder path

# Processing parameters
TARGET_SIZE = (512, 512)  # Width, height in pixels
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Valid image file extensions
VERBOSE = True  # Set to False to disable progress messages


def process_images(source_dir, output_dir, extensions=IMAGE_EXTENSIONS):
    """
    Process all images in source_dir and its subdirectories:
    - Convert to grayscale
    - Resize to 512x512
    - Save to output_dir with unique filenames
    
    Args:
        source_dir: Path to source folder containing images
        output_dir: Path to output folder where processed images will be saved
        extensions: Tuple of valid image file extensions
        
    Returns:
        Number of processed images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert source_dir to absolute path
    source_dir = os.path.abspath(source_dir)

    # Counter for processed images
    processed_count = 0

    # Determine the appropriate resampling filter based on Pillow version
    try:
        RESAMPLE_FILTER = Image.Resampling.LANCZOS
    except AttributeError:
        # For older Pillow versions
        RESAMPLE_FILTER = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS

    if VERBOSE:
        print(f"Processing images from: {source_dir}")
        print(f"Saving to: {output_dir}")
        print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} (grayscale)")

    # Walk through source directory and all subdirectories
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Check if file is an image based on extension
            if file.lower().endswith(extensions):
                try:
                    # Construct full file path
                    file_path = os.path.join(root, file)

                    # Open image
                    img = Image.open(file_path)

                    # Convert to grayscale
                    img_gray = img.convert('L')

                    # Resize to target size
                    img_resized = img_gray.resize(TARGET_SIZE, RESAMPLE_FILTER)

                    # Create a unique filename based on relative path
                    rel_path = os.path.relpath(root, source_dir)
                    if rel_path == '.':
                        # File is in the source directory, use original filename
                        unique_name = file
                    else:
                        # Replace path separators with underscores for subdirectory files
                        subdir_name = rel_path.replace(os.sep, '_')
                        unique_name = f"{subdir_name}_{file}"

                    # Construct output path
                    output_path = os.path.join(output_dir, unique_name)

                    # Save processed image
                    img_resized.save(output_path)

                    processed_count += 1
                    if VERBOSE and processed_count % 10 == 0:
                        print(f"Processed {processed_count} images...")
                        
                except Exception as e:
                    if VERBOSE:
                        print(f"Error processing {file}: {e}")

    if VERBOSE:
        print(f"\n✅ Total images processed: {processed_count}")

    return processed_count


def main():
    """
    Main function to process images
    """
    process_images(SOURCE_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()
