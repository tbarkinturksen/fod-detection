# fod-detection
"ENHANCING AVIATION SAFETY THROUGH AUTONOMOUS DETECTION OF FOREIGN OBJECT DEBRIS: AN ACCESSIBLE COMPUTER VISION &amp; DEEP LEARNING APPROACH"

## 📁 Python Codes File Overview
| File                          | Description |
|------------------------------|-------------|
| `autoencoder_tester.py`      | Tests a trained autoencoder by comparing original images with their reconstructions to identify anomalies (FOD) based on reconstruction errors. |
| `autoencoder_trainer.py`     | Trains an autoencoder neural network on runway surface images to learn normal surface patterns for later anomaly detection. |
| `autoencoder_detector.py`    | Processes video footage using a trained autoencoder to detect FOD on runway surfaces by identifying areas with high reconstruction error. |
| `cnn_trainer.py`             | Trains a Faster R-CNN object detection model on COCO-format annotations for identifying FOD on runways. |
| `cnn.py`                     | Performs object detection on runway video using a pre-trained or custom Faster R-CNN model. |
| `srt_cleaner.py`             | Processes DJI drone subtitle files to extract and format GPS coordinates for geo-referencing runway footage. |
| `frame_extractor.py`         | Extracts frames from video at specified intervals to create image datasets for training or analysis. |
| `namer.py`                   | Trims characters from the beginning of text file lines, useful for cleaning up dataset filenames or labels. |
| `patch_extractor.py`         | Extracts and resizes fixed-size patches from video frames using a sliding window for creating training datasets. |
| `resizer.py`                 | Batch resizes images to a specified resolution for dataset preparation. |
| `size_grayscale.py`          | Converts images to grayscale and resizes them to a standard size while preserving folder hierarchy information. |
| `coco_to_yolo.py`            | Converts object detection annotations from COCO JSON format to YOLO's normalized text format. |
| `yolo.py`                    | Performs FOD detection on runway video using YOLOv8 object detection and saves detections and cropped objects. |
| `yolo_training.py`           | Trains YOLOv8 models for small object detection with extensive dataset analysis, augmentation visualization, and optimization for FOD detection. |
| `runway_aoi_extractor.py`    | Provides a GUI tool for selecting specific areas of interest (AOIs) from runway videos and extracting them as patches. |
| `opencv.py`                  | Implements traditional computer vision methods (edge detection, contours, blobs) to detect FOD on runway surfaces without deep learning. |
| `final_rgb_detector.py`      | Comprehensive RGB FOD detection combining YOLO, R-CNN, and autoencoder texture verification with GPS mapping. |
| `ir_ae_trainer.py`           | Training script for IR-specific autoencoder model on grayscale runway images for anomaly detection. |
| `ir_detector.py`             | IR FOD detection system using YOLO and R-CNN with grayscale preprocessing and autoencoder verification. |
| `rgb_final_sliding_detection.py` | RGB FOD detector using sliding window approach with YOLO and R-CNN models and GPS mapping. |
| `small_ae_trainer.py`        | Specialized trainer for small texture patch autoencoder to establish anomaly thresholds from runway samples. |

## 📦 Dataset Access Link
The full dataset used in this thesis project is available for public access via Google Drive:
👉 [Click here to access the dataset](https://drive.google.com/drive/folders/1wUHemVtKnmjgpBtemwsJSwSYaLxsp6w3?usp=sharing)

The dataset includes all relevant models trained, image and video files, and much more resources used in this research.
> Note: Due to GitHub's file size limits, the dataset is hosted externally and not included in this repository.
