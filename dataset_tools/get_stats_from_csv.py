import pandas as pd
from collections import defaultdict

# Load CSV
CSV_FILE = "../datasets/stylized_images_cleaned/stylized_images_cleaned_dataset.csv"
df = pd.read_csv(CSV_FILE)

# Constants for binning
CONFIDENCE_INTERVALS = 10
SIZE_BINS = [0, 64, 128, 256, 512, 1024, 2048]
SIZE_LABELS = [f"<{s1}px" for s1 in SIZE_BINS[1:]] + [f">{SIZE_BINS[-1]}px"]

# Helper function to categorize sizes
def get_size_category(size, bins, labels):
    for i in range(len(bins) - 1):
        if bins[i] < size <= bins[i+1]:
            return labels[i]
    if size > bins[-1]:
        return labels[-1]
    return None

# Stats
num_images = len(df)
num_identities = df["identity"].nunique()

# Confidence distribution
confidence_histogram = defaultdict(int)
for conf in df["confidence"]:
    bin_index = int(conf * CONFIDENCE_INTERVALS)
    confidence_histogram[bin_index] += 1

# Width/height distribution
bbox_dimensions = defaultdict(int)
for _, row in df.iterrows():
    w_cat = get_size_category(row["width"], SIZE_BINS, SIZE_LABELS)
    h_cat = get_size_category(row["height"], SIZE_BINS, SIZE_LABELS)
    if w_cat and h_cat:
        key = f"width:{w_cat}, height:{h_cat}"
        bbox_dimensions[key] += 1

# Output results
print(f"Number of images: {num_images}")
print(f"Number of identities: {num_identities}")

print("\nConfidence score distribution:")
for i in sorted(confidence_histogram):
    low = i / CONFIDENCE_INTERVALS
    high = (i + 1) / CONFIDENCE_INTERVALS
    print(f"[{low:.1f} - {high:.1f}): {confidence_histogram[i]}")

print("\nImages with confidence < 0.7:")
low_conf_df = df[df["confidence"] < 0.7]
for path in low_conf_df["path_to_image"]:
    print(path)

print("\nImage width/height distribution:")
for k, v in sorted(bbox_dimensions.items()):
    print(f"{k}: {v}")
