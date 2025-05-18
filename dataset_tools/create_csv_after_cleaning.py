#!/usr/bin/env python3
"""
Filters the original face metadata CSV based on actual image files present after manual cleaning.

- Reads the metadata CSV produced by `align_faces.py`
- Checks which aligned face images still exist after cleaning
- Outputs a new CSV that includes only valid (existing) rows

This script ensures the final CSV matches the cleaned dataset folder structure.

Filename: create_csv_after_cleaning.py
Project: KNN Face Recognition
Version: 1.0
Author: Tomas Husar, xhusar11
"""
import os
import pandas as pd

# * Path to manually cleaned dataset (contains aligned face images)
# CLEANED_DATASET_DIRECTORY = "../datasets/stylized_images_112_fin"
CLEANED_DATASET_DIRECTORY = "../datasets/wiki_face_112_fin"

# * Output path for the new cleaned CSV
# OUTPUT_CSV_FILE = "stylized_images_112_fin.csv"
OUTPUT_CSV_FILE = "wiki_face_112_fin.csv"

# * Original CSV file with all aligned images (produced by align_faces.py)
# CSV_FILE = "../datasets/stylized_images_aligned_112/aligned_faces_metadata.csv"
CSV_FILE = "../datasets/WikiFaceAligned112/aligned_faces_metadata.csv"

# * Load the CSV
df = pd.read_csv(CSV_FILE, sep=';')

# * Function checks if an aligned image exists in the cleaned dataset
def file_exists(aligned_path):
    full_path = os.path.join(CLEANED_DATASET_DIRECTORY, aligned_path)
    return os.path.isfile(full_path)

# * Filter to rows where the aligned image still exist
existing_df = df[df["aligned_image"].apply(file_exists)]

# * How many rows were dropped
missing_count = len(df) - len(existing_df)

# * Save filtered CSV
existing_df.to_csv(OUTPUT_CSV_FILE, sep=';', index=False)

# * Print summary
print(f"Total rows in original CSV: {len(df)}")
print(f"Rows written to {OUTPUT_CSV_FILE}: {len(existing_df)}")
print(f"Deleted aligned images (Detections): {missing_count}")