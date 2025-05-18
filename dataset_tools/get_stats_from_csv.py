# code used only during the dataset cleanup
# creates a txt file containing dataset statistics (used for the manual cleaning)

import pandas as pd
from collections import defaultdict
import contextlib

# * Define where to print
# OUTPUT_FILE = "stylized_images_aligned_112_stats_output_02.txt"
OUTPUT_FILE = "stylized_images_112_stats_output.txt"
OUTPUT_FILE = "wiki_face_112_stats_output.txt"

# * Load CSV (align_faces.py/create_csv_after_cleaning.py output)
# CSV_FILE = "stylized_images_112_fin.csv"
CSV_FILE = "wiki_face_112_fin.csv"

df = pd.read_csv(CSV_FILE, sep=';')

# Constants for binning
CONFIDENCE_INTERVALS = 10   # 0.0-0.1; 0.1-0.2; ...

# Stats
num_images = len(df)
num_identities = df["identity"].nunique()

# Confidence distribution
confidence_histogram = defaultdict(int)
for conf in df["confidence"]:
    bin_index = int(conf * CONFIDENCE_INTERVALS)
    confidence_histogram[bin_index] += 1

# Redirect standard output to file
with open(OUTPUT_FILE, 'w') as f, contextlib.redirect_stdout(f):

    print(f"Number of images: {num_images}")
    print(f"Number of identities: {num_identities}")

    print("\nConfidence score distribution:")
    for i in sorted(confidence_histogram):
        low = i / CONFIDENCE_INTERVALS
        high = (i + 1) / CONFIDENCE_INTERVALS
        print(f"[{low:.1f} - {high:.1f}): {confidence_histogram[i]}")

    print("\nImages with confidence < 0.7:")
    low_conf_df = df[df["confidence"] < 0.7]
    for path in low_conf_df["aligned_image"]:
        print(path)

    # note: ! AFTER MANUAL CLEANING, THIS SHOULD NOT PRINT ANY

    print("\nOriginal images with multiple aligned versions:")
    duplicates = df["original_image"].value_counts()
    duplicates = duplicates[duplicates > 1]

    for original_image, count in duplicates.items():
        print(f"\nOriginal image: {original_image} (count: {count})")
        aligned_versions = df[df["original_image"] == original_image]["aligned_image"].tolist()
        for aligned_image in aligned_versions:
            print(f"  - {aligned_image}")

    # note: ! AFTER MANUAL CLEANING, THIS SHOULD NOT PRINT ANY
    print("\nIdentities with only one original image:")
    identity_grouped = df.groupby("identity")["original_image"].nunique()
    single_image_identities = identity_grouped[identity_grouped == 1]

    for identity in single_image_identities.index:
        print(f"- {identity}")