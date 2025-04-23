# this will be final file which will clean all at once

# filename: dataset_cleaner.py
# project: KNN Face Recognition
# version: 1.0
# author: xjanos19

import os
import json
import math
import shutil
import pandas as pd
from PIL import Image

WIKIFACE_DIR = "WikiFace"
DETECTIONS_DIR = "WikiFaceOutput/detections"
CLEANED_DIR = "WikiFaceCleaned"
REMOVED_DIR = "WikiFaceRemoved"

os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(REMOVED_DIR, exist_ok=True)

removed_entries = []


def move_removed_file(identity, img_path, reason):
    try:
        dest_dir = os.path.join(REMOVED_DIR, identity)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(img_path, dest_dir)
    except Exception as e:
        reason += f" (copy error: {e})"
    removed_entries.append(
        {"identity": identity, "path_to_image": img_path, "reason": reason}
    )


def list_identities(base_dir):
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]


def list_images(identity_dir):
    return sorted(
        [
            f
            for f in os.listdir(identity_dir)
            if f.endswith(".jpeg") or f.endswith(".jpg")
        ]
    )


def jsonl_path(identity, img_name):
    return os.path.join(
        DETECTIONS_DIR,
        identity,
        img_name.replace(".jpeg", ".jsonl").replace(".jpg", ".jsonl"),
    )


def parse_jsonl(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except:
        return None


def process_images():
    rows = []

    for identity in list_identities(WIKIFACE_DIR):
        image_dir = os.path.join(WIKIFACE_DIR, identity)
        for img_name in list_images(image_dir):
            img_path = os.path.join(image_dir, img_name)
            det_path = jsonl_path(identity, img_name)

            if not os.path.exists(det_path):
                move_removed_file(identity, img_path, "missing JSONL detection")
                continue

            detections = parse_jsonl(det_path)
            if detections is None:
                move_removed_file(identity, img_path, "invalid JSONL file")
                continue

            if len(detections) == 0:
                move_removed_file(identity, img_path, "no detections")
                continue

            rows.append(
                {
                    "identity": identity,
                    "path_to_image": img_path,
                    "num_detections": len(detections),
                    "detections": detections,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv("step1_detections.csv", index=False)

    return rows


def crop_images(rows):
    cropped_rows = []

    for row in rows:
        identity = row["identity"]
        img_path = row["path_to_image"]
        detections = row["detections"]

        # Select detection with highest confidence
        det = max(detections, key=lambda d: d.get("confidence", 1.0))
        conf = det.get("confidence", 1.0)

        try:
            x_center = round(det["x_center"])
            y_center = round(det["y_center"])
            width = math.ceil(det["bounding_box_width"])
            height = math.ceil(det["bounding_box_height"])
        except Exception as e:
            move_removed_file(identity, img_path, f"invalid bounding box values: {e}")
            continue

        left = max(0, x_center - width // 2)
        upper = max(0, y_center - height // 2)
        right = left + width
        lower = upper + height

        try:
            img = Image.open(img_path)
            cropped = img.crop((left, upper, right, lower))
        except Exception as e:
            move_removed_file(
                identity, img_path, f"error while reading or cropping: {e}"
            )
            continue

        out_dir = os.path.join(CLEANED_DIR, identity)
        os.makedirs(out_dir, exist_ok=True)

        cropped_name = os.path.basename(img_path)
        out_path = os.path.join(out_dir, cropped_name)

        try:
            cropped.save(out_path)
        except Exception as e:
            move_removed_file(
                identity, img_path, f"error while saving cropped image: {e}"
            )
            continue

        cropped_rows.append(
            {
                "identity": identity,
                "path_to_image": out_path,
                "confidence": conf,
                "width": width,
                "height": height,
            }
        )

    df_crop = pd.DataFrame(cropped_rows)
    df_crop.to_csv("step2_cropped.csv", index=False)

    return df_crop


def filter_images_by_confidence(df_crop):
    df_crop_filtered = df_crop[df_crop["confidence"] >= 0.7].copy()
    rejected_conf = df_crop[df_crop["confidence"] < 0.7]
    for _, row in rejected_conf.iterrows():
        move_removed_file(row["identity"], row["path_to_image"], "confidence < 0.7")

    df_crop_filtered.to_csv("step3_filtered_by_confidence.csv", index=False)

    return df_crop_filtered


def filter_by_aspect_ratio(df_crop_filtered):
    final_rows = []

    for _, row in df_crop_filtered.iterrows():
        identity = row["identity"]
        img_path = row["path_to_image"]
        width = row["width"]
        height = row["height"]

        if width <= height:
            final_rows.append(row)
        else:
            try:
                img = Image.open(img_path)
                img.show()
            except Exception as e:
                move_removed_file(
                    identity, img_path, f"could not display for manual decision: {e}"
                )
                continue

            answer = (
                input(
                    f"Image {img_path} has width > height. Delete (y), rotate right (r), or rotate left (l)? "
                )
                .strip()
                .lower()
            )
            if answer == "y":
                move_removed_file(
                    identity, img_path, "width > height – manually removed"
                )
            elif answer == "r":
                try:
                    rotated = img.rotate(-90, expand=True)  # Rotate right (clockwise)
                    rotated.save(img_path)  # overwrite original file
                    row["width"], row["height"] = height, width
                    final_rows.append(row)
                except Exception as e:
                    move_removed_file(
                        identity, img_path, f"error while rotating right: {e}"
                    )
            elif answer == "l":
                try:
                    rotated = img.rotate(
                        90, expand=True
                    )  # Rotate left (counter-clockwise)
                    rotated.save(img_path)  # overwrite original file
                    row["width"], row["height"] = height, width
                    final_rows.append(row)
                except Exception as e:
                    move_removed_file(
                        identity, img_path, f"error while rotating left: {e}"
                    )
            else:
                print("Invalid input – skipping image.")

    df_final = pd.DataFrame(final_rows)
    df_final.to_csv("step4_final_cleaned_dataset.csv", index=False)

    return df_final


def cleanup():
    temp_files = [
        "step1_detections.csv",
        "step2_cropped.csv",
        "step3_filtered_by_confidence.csv",
    ]

    for f in temp_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"⚠️ Could not delete {f}: {e}")


if __name__ == "__main__":
    rows = process_images()
    df_crop = crop_images(rows)
    df_crop_filtered = filter_images_by_confidence(df_crop)
    df_final = filter_by_aspect_ratio(df_crop_filtered)

    # Save list of removed images with reasons
    df_removed = pd.DataFrame(removed_entries)
    df_removed.to_csv("removed_images_log.csv", index=False)

    # Cleanup temporary step files
    cleanup()

    print(f"✅ Done: {len(df_final)} final images | {len(df_removed)} removed")
