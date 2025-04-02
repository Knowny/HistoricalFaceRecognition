import jsonlines
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# Path to Yolo11 detections of WikiFace 
detections_path = "../datasets/WikiFaceOutput/detections"

def analyze_detections(detections_path):
    """
    Analyzes YOLO11 detection results in .jsonl format to calculate dataset metrics.

    Args:
        detections_path (str): Path to the directory containing identity subdirectories
                                with .jsonl detection files.

    Returns:
        tuple: A tuple containing the following metrics:
            - num_identities (int): Number of detected identities.
            - num_all_portraits (int): Total number of .jsonl files.
            - num_detected_portraits (int): Total number of detections across all files.
            - confidence_histogram (dict): Histogram of confidence intervals.
            - bbox_dimensions (dict): Counts of bounding boxes within different size categories.
    """
    num_identities = 0
    num_all_portraits = 0
    num_detected_portraits = 0

    confidence_histogram = defaultdict(int) # used to store the counts of confidences for different intervals
    bbox_dimensions = defaultdict(int)      # used to store the counts of bounding boxes for each combination of width and height categories.
    
    size_bins = [0, 64, 128, 256, 512, 1024, 2048]
    size_labels = [f"<{s1}*{s1}px" for s1 in size_bins[1:]] + [f">{size_bins[-1]}*{size_bins[-1]}px"]

    if not os.path.exists(detections_path):
        print(f"Error: Detections path '{detections_path}' does not exist.")
        return 0, 0, 0, {}, {}

    # * Number of identities
    identity_directories = [d for d in os.listdir(detections_path) if os.path.isdir(os.path.join(detections_path, d))]
    num_identities = len(identity_directories)

    # * Number of all portraits, 
    for identity_dir in identity_directories:
        identity_path = os.path.join(detections_path, identity_dir)
        jsonl_files = [f for f in os.listdir(identity_path) if f.endswith(".jsonl")]
        num_all_portraits += len(jsonl_files)

        for jsonl_file in jsonl_files:
            file_path = os.path.join(identity_path, jsonl_file)
            try:
                # reader allows to iterate over the lines in the .jsonl file, where each line is parsed as a JSON object
                with jsonlines.open(file_path) as reader:
                    # iterate through each line in the file
                    for detection in reader:
                        num_detected_portraits += 1
                        confidence = detection.get("confidence")
                        if confidence is not None:
                            # Convert to int for easy indexation
                            confidence_interval = int(confidence * 10)  # Group into 0.0-0.1, 0.1-0.2, etc.
                            confidence_histogram[confidence_interval] += 1

                        # * Bounding box dimensions
                        width = detection.get("bounding_box_width")
                        height = detection.get("bounding_box_height")
                        if width is not None and height is not None:
                            width_category = None
                            height_category = None
                            # iterate through the indices of the size_bins list [0, 64, 128, 256, 512, 1024, 2048].
                            for i in range(len(size_bins) - 1):
                                # check if the bounding box width falls within the current size interval: size_bins[i] (the lower bound); size_bins[i+1] (the upper bound)
                                if size_bins[i] < width <= size_bins[i+1]:
                                    width_category = size_labels[i]
                                # check if the bounding box height falls within the current size interval: size_bins[i] (the lower bound); size_bins[i+1] (the upper bound)
                                if size_bins[i] < height <= size_bins[i+1]:
                                    height_category = size_labels[i]
                            # check if the width/height is greater than the last value in the size_bins list
                            if width > size_bins[-1]:
                                width_category = size_labels[-1]
                            if height > size_bins[-1]:
                                height_category = size_labels[-1]

                            # increment the count for the specific width and height category
                            if width_category and height_category:
                                bbox_dimensions[f"width:{width_category}, height:{height_category}"] += 1
            except FileNotFoundError:
                print(f"Error: File not found: {file_path}")
            except jsonlines.jsonlines.InvalidJSONError:
                print(f"Error: Invalid JSON in file: {file_path}")

    return num_identities, num_all_portraits, num_detected_portraits, confidence_histogram, bbox_dimensions

if __name__ == "__main__":
    num_identities, num_all_portraits, num_detected_portraits, confidence_histogram, bbox_dimensions = analyze_detections(detections_path)

    print(f"Number of identities: {num_identities}")
    print(f"Number of all portraits (JSONL files): {num_all_portraits}")
    print(f"Number of detected portraits (detections): {num_detected_portraits}")

    print("\nConfidence Histogram:")
    for interval, count in sorted(confidence_histogram.items()):
        print(f"[{interval/10:.1f} - {(interval+1)/10:.1f}): {count}")

    print("\nBounding Box Dimensions:")
    for dimensions, count in sorted(bbox_dimensions.items()):
        print(f"{dimensions}: {count}")

    # Visualize the confidence histogram
    # create a list where each element is the midpoint of a confidence interval for which we have detection counts in confidence_histogram 
    # useful for plotting the histogram, as we want to position the bars in the middle of intervals on the x-axis.
    confidence_values = [interval / 10 + 0.05 for interval in confidence_histogram.keys()]
    confidence_counts = list(confidence_histogram.values())

    plt.figure(figsize=(10, 6))
    plt.bar(confidence_values, confidence_counts, width=0.08, alpha=0.7)
    plt.xlabel("Confidence")
    plt.ylabel("Number of Detections")
    plt.title("Histogram of Detection Confidences")
    plt.xticks([i/10 for i in range(11)])
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig("confidence_histogram.png")
