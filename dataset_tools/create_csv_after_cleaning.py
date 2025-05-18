# code used only during the dataset cleanup
# creates a new CSV file for the dataset after manual cleaning from the original CSV and folder structure

import os
import pandas as pd

# * path to dataset after manual cleaning
# CLEANED_DATASET_DIRECTORY = "../datasets/stylized_images_112_fin"
CLEANED_DATASET_DIRECTORY = "../datasets/wiki_face_112_fin"

# * path to output csv file
# OUTPUT_CSV_FILE = "stylized_images_112_fin.csv"
OUTPUT_CSV_FILE = "wiki_face_112_fin.csv"

# * load the CSV file from the align_faces.py output
# CSV_FILE = "../datasets/stylized_images_aligned_112/aligned_faces_metadata.csv"
CSV_FILE = "../datasets/WikiFaceAligned112/aligned_faces_metadata.csv"

df = pd.read_csv(CSV_FILE, sep=';')

# Check which aligned images exist
def file_exists(aligned_path):
    full_path = os.path.join(CLEANED_DATASET_DIRECTORY, aligned_path)
    return os.path.isfile(full_path)

# Filter rows where the aligned image exist
existing_df = df[df["aligned_image"].apply(file_exists)]

# Count how many were skipped
missing_count = len(df) - len(existing_df)

# Save filtered dataframe to file
existing_df.to_csv(OUTPUT_CSV_FILE, sep=';', index=False)

# Print result
print(f"Total rows in original CSV: {len(df)}")
print(f"Rows written to {OUTPUT_CSV_FILE}: {len(existing_df)}")
print(f"Deleted aligned images (Detections): {missing_count}")