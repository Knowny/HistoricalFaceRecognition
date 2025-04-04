import os
import jsonlines
from collections import defaultdict


DETECTIONS_PATH = "../datasets/WikiFaceDetectionOutput/detections"
CONFIDENCE_INTERVALS = 10  # Group confidence scores into 10 intervals (0.0-0.1, ..., 0.9-1.0)
SIZE_BINS = [0, 64, 128, 256, 512, 1024, 2048]
SIZE_LABELS = [f"<{s1}px" for s1 in SIZE_BINS[1:]] + [f">{SIZE_BINS[-1]}px"]

def analyze_detections(detections_path):
    """
    Analyzes YOLO11 detection results in .jsonl format to calculate dataset metrics.

    Args:
        detections_path (str): Path to the directory containing identity subdirectories
                                with .jsonl detection files.

    Returns:
        tuple: A tuple containing dataset metrics.
    """
    num_identities = 0
    num_all_portraits = 0
    num_detected_portraits = 0

    confidence_histogram = defaultdict(int)
    bbox_dimensions = defaultdict(int)

    if not os.path.exists(detections_path):
        print(f"Error: Detections path '{detections_path}' does not exist.")
        return 0, 0, 0, {}, {}

    # * COUNT IDENTITIES
    identity_dirs = [
        d for d in os.listdir(detections_path)
        if os.path.isdir(os.path.join(detections_path, d))
    ]
    num_identities = len(identity_dirs)

    # * ANALYZE DETECTIONS IN EACH DIRECTORY
    for identity_dir in identity_dirs:
        identity_path = os.path.join(detections_path, identity_dir)
        jsonl_files = [f for f in os.listdir(identity_path) if f.endswith(".jsonl")]
        num_all_portraits += len(jsonl_files)

        for jsonl_file in jsonl_files:
            file_path = os.path.join(identity_path, jsonl_file)
            try:
                with jsonlines.open(file_path) as reader:
                    for detection in reader:
                        num_detected_portraits += 1

                        # * "ANALYZE" CONFIDENDE
                        confidence = detection.get("confidence")
                        if confidence is not None:
                            confidence_bin = int(confidence * CONFIDENCE_INTERVALS)
                            confidence_histogram[confidence_bin] += 1

                        # * ANALYZE BOUNDING BOX DIMENSIONS
                        width = detection.get("bounding_box_width")
                        height = detection.get("bounding_box_height")
                        if width is not None and height is not None:
                            width_category = _get_size_category(width, SIZE_BINS, SIZE_LABELS)
                            height_category = _get_size_category(height, SIZE_BINS, SIZE_LABELS)
                            if width_category and height_category:
                                bbox_dimensions[f"width:{width_category}, height:{height_category}"] += 1

            except FileNotFoundError:
                print(f"Error: File not found: {file_path}")
            except jsonlines.jsonlines.InvalidJSONError:
                print(f"Error: Invalid JSON in file: {file_path}")

    return num_identities, num_all_portraits, num_detected_portraits, confidence_histogram, bbox_dimensions

def _get_size_category(size, bins, labels):
    """Helper function to categorize a size based on predefined bins."""
    for i in range(len(bins) - 1):
        if bins[i] < size <= bins[i+1]:
            return labels[i]
    if size > bins[-1]:
        return labels[-1]
    return None

if __name__ == "__main__":
    num_identities, num_all_portraits, num_detected_portraits, confidence_histogram, bbox_dimensions = analyze_detections(DETECTIONS_PATH)

    print(f"Number of identities: {num_identities}")
    print(f"Number of all portraits (JSONL files): {num_all_portraits}")
    print(f"Number of detected portraits (detections): {num_detected_portraits}")

    print("\nConfidence Histogram:")
    for interval, count in sorted(confidence_histogram.items()):
        lower_bound = interval / CONFIDENCE_INTERVALS
        upper_bound = (interval + 1) / CONFIDENCE_INTERVALS
        print(f"[{lower_bound:.1f} - {upper_bound:.1f}): {count}")

    print("\nBounding Box Dimensions:")
    for dimensions, count in sorted(bbox_dimensions.items()):
        print(f"{dimensions}: {count}")