import json
import os
import argparse

#################################################
# CONFIGURATION SETTINGS
#################################################
# Default file paths (can be overridden with command line arguments)
DEFAULT_INPUT_PATH = 'instances_val.json'  # Path to COCO JSON file
DEFAULT_OUTPUT_DIR = 'yolo_labels'  # Output directory for YOLO format files


def convert_coco_to_yolo(coco_json_path, output_dir):
    """
    Convert COCO format JSON annotations to YOLO format txt files
    
    Args:
        coco_json_path: Path to the COCO JSON file
        output_dir: Directory to save the YOLO format txt files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Load COCO JSON
    print(f"Loading COCO annotations from: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create a dictionary to map image_id to image info
    image_dict = {img['id']: img for img in coco_data['images']}
    
    # Get number of images and annotations
    num_images = len(image_dict)
    num_annotations = len(coco_data['annotations'])
    print(f"Found {num_images} images with {num_annotations} annotations")

    # Create a dictionary to collect annotations by image_id
    annotations_by_image = {}

    # Group annotations by image_id
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # Counter for processed files
    processed_count = 0

    # Process each image
    for image_id, image_info in image_dict.items():
        # Get image dimensions
        image_width = image_info['width']
        image_height = image_info['height']

        # Get image filename without extension
        image_filename = os.path.splitext(image_info['file_name'])[0]

        # Create output file path
        output_file_path = os.path.join(output_dir, f"{image_filename}.txt")

        # Check if image has annotations
        if image_id in annotations_by_image:
            # Process annotations for this image
            with open(output_file_path, 'w') as f:
                for annotation in annotations_by_image[image_id]:
                    # Get bounding box coordinates (COCO format: [x, y, width, height])
                    bbox = annotation['bbox']
                    x, y, width, height = bbox

                    # Convert to YOLO format (normalized center_x, center_y, width, height)
                    center_x = (x + width / 2) / image_width
                    center_y = (y + height / 2) / image_height
                    norm_width = width / image_width
                    norm_height = height / image_height

                    # Class ID is always 0 as specified (FOD class)
                    class_id = 0

                    # Create YOLO format line with 6 decimal places
                    yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                    f.write(yolo_line + '\n')
        else:
            # Create empty file for images with no annotations
            with open(output_file_path, 'w') as f:
                pass
        
        processed_count += 1
        
        # Print progress
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{num_images} images")

    print(f"\n✅ Conversion completed. {processed_count} YOLO format files saved to {output_dir}")


def main():
    """
    Parse command line arguments and run the conversion
    """
    parser = argparse.ArgumentParser(description='Convert COCO JSON to YOLO format')
    parser.add_argument('--input', default=DEFAULT_INPUT_PATH, 
                        help=f'Path to COCO JSON file (default: {DEFAULT_INPUT_PATH})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR, 
                        help=f'Output directory for YOLO format files (default: {DEFAULT_OUTPUT_DIR})')

    args = parser.parse_args()
    
    print("COCO to YOLO Annotation Converter")
    print("=================================")
    
    convert_coco_to_yolo(args.input, args.output)


if __name__ == "__main__":
    main()
