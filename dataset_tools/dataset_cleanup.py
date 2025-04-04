# filename: dataset_cleanup.py
# This file cleans up identities from data that are not usable
#     1. remove multi identity samples
#     2. remove samples where confidence of detection is lower than 50%
# source of data is expected to be WikiFace dataset (and its expected folder format)
# project: KNN Face Recognition
# version: 1.0
# author: xmager00

import os
import shutil
import json
from tqdm import tqdm


def get_clean_data(image_dir, json_dir, output_image_dir, confidence_threshold=0.5):
    """
    Cleans the data and returns the cleaned image directory

    @param image_dir: Directory containing images
    @param json_dir: Directory containing JSONL files
    @param output_image_dir: Directory to save cleaned images
    @param confidence_threshold: Confidence threshold for detection
    """

    os.makedirs(output_image_dir, exist_ok=True)

    person_dirs = os.listdir(image_dir)

    for person_dir in tqdm(person_dirs, desc="Cleaning data", unit=" person"):

        # match directories by person name
        person_img_path = os.path.join(image_dir, person_dir)
        person_json_path = os.path.join(json_dir, person_dir)

        if os.path.isdir(person_img_path) and os.path.isdir(person_json_path):
            os.makedirs(os.path.join(output_image_dir, person_dir), exist_ok=True)

            img_files = os.listdir(person_img_path)

            for file in img_files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        format_parts = file.split('_face_')
                        json_file_identity = format_parts[0]
                        json_file = os.path.join(person_json_path, json_file_identity + '.jsonl')

                        if os.path.exists(json_file):
                            with open(json_file, 'r') as f:
                                lines = f.readlines()
                            
                            if len(lines) == 1: 
                                try:
                                    json_data = json.loads(lines[0])
                                    if json_data['confidence'] >= confidence_threshold:
                                        shutil.copy(os.path.join(person_img_path, file), os.path.join(output_image_dir, person_dir, file))
                                except json.JSONDecodeError:
                                    print(f"Error decoding JSON file: {json_file}")
                        else:
                            print(f"JSON file not found: {json_file}")
                    except IndexError:
                        print(f"Error processing file: {file}")

    return output_image_dir

def main():

    image_dir = "../datasets/WikiFaceCropped"
    json_dir = "../datasets/WikiFaceDetectionOutput/detections"
    image_dir_clean = "../datasets/WikiFaceCleaned2"

    if not os.path.exists(image_dir_clean):
        root_dir = get_clean_data(image_dir, json_dir, image_dir_clean, confidence_threshold=0.5)
        print(f"Cleaned data saved to: {root_dir} successfully.")
        if not root_dir:
            print("Cleanup Failed.")
            return
    else:
        root_dir = image_dir_clean
        print(f"Clean data directory already exists, using: {root_dir}")

if __name__ == "__main__":
    main()