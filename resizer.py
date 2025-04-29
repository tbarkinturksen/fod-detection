import os
from PIL import Image

#################################################
# CONFIGURATION SETTINGS
#################################################
# Paths
INPUT_FOLDER = "input"  # Folder containing input images
OUTPUT_FOLDER = "output"  # Folder to save resized images

# Resize parameters
TARGET_SIZE = (640, 640)  # Width, height in pixels


def process_images(input_folder, output_folder, size=TARGET_SIZE):
    """
    Resize all images in input folder and save to output folder
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to folder where resized images will be saved
        size: Target size as (width, height) tuple
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Get list of image files
    image_count = 0
    error_count = 0
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # Open, convert to RGB, resize, and save the image
                with Image.open(input_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    img.save(output_path)
                    image_count += 1
                    
                    # Print progress every 10 images
                    if image_count % 10 == 0:
                        print(f"Processed {image_count} images...")
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                error_count += 1

    # Print summary
    print(f"\n✅ Processing completed:")
    print(f"   - {image_count} images resized to {size[0]}x{size[1]}")
    if error_count > 0:
        print(f"   - {error_count} images failed to process")


def main():
    """
    Main function to resize images
    """
    print(f"Resizing images from '{INPUT_FOLDER}' to '{OUTPUT_FOLDER}'")
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} pixels")
    
    process_images(INPUT_FOLDER, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
