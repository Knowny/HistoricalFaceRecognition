#!/usr/bin/env python3
"""
This code aligns face images from the src dataset and creates CSV metadata file.
note: Multiple detected faces from one image are saved as two separate images.

filename: align_faces.py
project: KNN Face Recognition
version: 1.0
author: Tomas Husar, xhusar11
"""
import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import csv
from tqdm import tqdm

from face_alignment import allign_face

# * CONFIGURATION
DETECTOR_PATH = "../archival_faces_detector/ArchivalFaces_2024_08_07_fold_0_yolo11l.pt"
CROP_SIZE = (112, 112)
DATASET_PATH = "../datasets/stylized_images"
ALIGNED_DATASET_PATH = "../datasets/stylized_images_aligned_112"
OUTPUT_CSV = "../datasets/stylized_images_aligned_112/aligned_faces_metadata.csv"

# Create output directory if it doesn't exist
os.makedirs(ALIGNED_DATASET_PATH, exist_ok=True)

# Initialize detector
detector = YOLO(DETECTOR_PATH, verbose=False)

# Gather all image paths
all_images = []
for identity in sorted(os.listdir(DATASET_PATH)):
    identity_folder = os.path.join(DATASET_PATH, identity)
    if not os.path.isdir(identity_folder):
        continue

    for file_name in sorted(os.listdir(identity_folder)):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            all_images.append((identity, os.path.join(identity_folder, file_name)))

# Write output CSV
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=";")
    csvwriter.writerow(["identity", "original_image", "aligned_image", "confidence"])

    for identity, image_path in tqdm(all_images, desc="Processing images"):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            continue

        # Get results from YOLO face detector
        results = detector(image, verbose=False)
        if not results:
            continue

        result = results[0].cpu()
        keypoints = result.keypoints
        confidences = result.boxes.conf.tolist()

        # Skip if no keypoints detected
        if keypoints is None or len(keypoints) == 0:
            continue

        keypoints = keypoints.xy    # shape [N, 5, 2] (N faces, 5 keypoints per face)

        aligned_identity_folder = os.path.join(ALIGNED_DATASET_PATH, identity)
        os.makedirs(aligned_identity_folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Process each detected face in the image
        for idx, (face_keypoints, confidence) in enumerate(zip(keypoints.data[:, :, :2], confidences), 1):
            try:
                # Align the face using 5 keypoints
                aligned_face = allign_face(image, None, face_keypoints, CROP_SIZE)
                if aligned_face is None:
                    raise ValueError("allign_face returned None")

                # Save the face
                aligned_filename = f"{base_name}_{idx:02d}.jpg"
                aligned_path = os.path.join(aligned_identity_folder, aligned_filename)
                aligned_face.save(aligned_path)

                # Write to CSV
                csvwriter.writerow([
                    identity,
                    os.path.relpath(image_path, DATASET_PATH),
                    os.path.relpath(aligned_path, ALIGNED_DATASET_PATH),
                    float(confidence)
                ])
            except Exception as e:
                print(f"Failed to process detection #{idx} in {image_path}: {e}")
