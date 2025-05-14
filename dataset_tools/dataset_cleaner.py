# filename: dataset_cleaner.py
# project: KNN Face Recognition
# version: 2.0
# author: xjanos19

import os
import json
import math
import shutil
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# * UNCOMMENT, if dataset == WikiFace
# DATASET_DIR = "datasets/WikiFace"
# DETECTIONS_DIR = "datasets/WikiFaceOutput/detections"
# CLEANED_DIR = "datasets/WikiFaceCleaned"
# REMOVED_DIR = "datasets/WikiFaceRemoved"
# DATASET_NAME = 'WikiFace'

# * UNCOMMENT, if dataset == stylized_imaged
DATASET_DIR = "datasets/style/stylized_images"
DETECTIONS_DIR = "datasets/style/stylized_images_detection_output/detections"
CLEANED_DIR = "datasets/style/stylized_images_cleaned"
REMOVED_DIR = "datasets/style/stylized_images_removed"
DATASET_NAME = 'stylized_images'

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

    for identity in list_identities(DATASET_DIR):
        image_dir = os.path.join(DATASET_DIR, identity)
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

        if DATASET_NAME == "WikiFace":
            detection_img_path = img_path.replace("WikiFace", "WikiFaceOutput/images")
            detection_img_path = detection_img_path.replace(".jpeg", ".jpg")

        if DATASET_NAME == "stylized_images":
            detection_img_path = img_path.replace("stylized_images", "stylized_images_detection_output/images")
            detection_img_path = detection_img_path.replace(".jpeg", ".jpg")

        try:
            img = Image.open(detection_img_path).convert("RGB")
        except Exception as e:
            move_removed_file(identity, img_path, f"error opening detection image: {e}")
            continue

        if len(detections) == 1:
            det = detections[0]
            conf = det.get("confidence", 1.0)

            try:
                x_center = round(det["x_center"])
                y_center = round(det["y_center"])
                width = math.ceil(det["bounding_box_width"])
                height = math.ceil(det["bounding_box_height"])
            except Exception as e:
                move_removed_file(
                    identity, img_path, f"invalid bounding box values: {e}"
                )
                continue

            left = max(0, x_center - width // 2)
            upper = max(0, y_center - height // 2)
            right = left + width
            lower = upper + height

            try:
                original_img = Image.open(img_path)
                cropped = original_img.crop((left, upper, right, lower))
            except Exception as e:
                move_removed_file(identity, img_path, f"error during crop: {e}")
                continue

            out_dir = os.path.join(CLEANED_DIR, identity)
            os.makedirs(out_dir, exist_ok=True)

            cropped_name = os.path.basename(img_path)
            out_path = os.path.join(out_dir, cropped_name)

            try:
                cropped.save(out_path)
            except Exception as e:
                move_removed_file(
                    identity, img_path, f"error saving cropped image: {e}"
                )
                continue

            cropped_rows.append(
                {
                    "identity": identity,
                    "path_to_image": out_path,
                    "confidence": conf,
                    "width": width,
                    "height": height,
                    "x_center": x_center,
                    "y_center": y_center,
                }
            )

        else:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", size=40)
            except:
                font = ImageFont.load_default()

            for idx, det in enumerate(detections):
                x_center = round(det["x_center"])
                y_center = round(det["y_center"])
                width = math.ceil(det["bounding_box_width"])
                height = math.ceil(det["bounding_box_height"])

                left = max(0, x_center - width // 2)
                upper = max(0, y_center - height // 2)
                right = left + width
                lower = upper + height

                text = str(idx)
                text_size = draw.textbbox((0, 0), text, font=font)
                text_width = text_size[2] - text_size[0]
                text_height = text_size[3] - text_size[1]

                text_x = left + 10
                text_y = upper + 10

                padding = 6
                draw.rectangle(
                    [
                        (text_x - padding, text_y - padding),
                        (text_x + text_width + padding, text_y + text_height + padding),
                    ],
                    fill="black",
                )
                draw.text((text_x, text_y-8), text, fill="white", font=font)

            print(f"\nImage: {img_path}")
            img.show()

            while True:
                try:
                    choice = int(
                        input(
                            f"Select detection index (0-{len(detections)-1}) or -1 to skip image: "
                        ).strip()
                    )
                    if choice == -1:
                        move_removed_file(
                            identity,
                            img_path,
                            "user skipped image during crop selection",
                        )
                        break
                    if 0 <= choice < len(detections):
                        det = detections[choice]
                        conf = det.get("confidence", 1.0)

                        try:
                            x_center = round(det["x_center"])
                            y_center = round(det["y_center"])
                            width = math.ceil(det["bounding_box_width"])
                            height = math.ceil(det["bounding_box_height"])
                        except Exception as e:
                            move_removed_file(
                                identity, img_path, f"invalid bounding box values: {e}"
                            )
                            break

                        left = max(0, x_center - width // 2)
                        upper = max(0, y_center - height // 2)
                        right = left + width
                        lower = upper + height

                        try:
                            original_img = Image.open(img_path)
                            cropped = original_img.crop((left, upper, right, lower))
                        except Exception as e:
                            move_removed_file(
                                identity, img_path, f"error during crop: {e}"
                            )
                            break

                        out_dir = os.path.join(CLEANED_DIR, identity)
                        os.makedirs(out_dir, exist_ok=True)

                        cropped_name = os.path.basename(img_path)
                        out_path = os.path.join(out_dir, cropped_name)

                        try:
                            cropped.save(out_path)
                        except Exception as e:
                            move_removed_file(
                                identity, img_path, f"error saving cropped image: {e}"
                            )
                            break

                        cropped_rows.append(
                            {
                                "identity": identity,
                                "path_to_image": out_path,
                                "confidence": conf,
                                "width": width,
                                "height": height,
                                "x_center": x_center,
                                "y_center": y_center,
                            }
                        )
                        break
                    else:
                        print(
                            f"Invalid choice. Please select between 0 and {len(detections)-1}, or -1 to skip."
                        )
                except ValueError:
                    print("Invalid input. Please enter a number.")

    df_crop = pd.DataFrame(cropped_rows)
    df_crop.to_csv("step2_cropped.csv", index=False)

    return df_crop


def filter_by_confidence(df_crop):
    df_crop_filtered = []

    for _, row in df_crop.iterrows():
        identity = row["identity"]
        img_path = row["path_to_image"]
        confidence = row["confidence"]

        if confidence >= 0.7:
            df_crop_filtered.append(row)
        else:
            try:
                img = Image.open(img_path)
                img.show()
            except Exception as e:
                move_removed_file(
                    identity,
                    img_path,
                    f"could not display for manual confidence check: {e}",
                )
                continue

            while True:
                answer = (
                    input(
                        f"Image {img_path} has confidence {confidence:.2f} < 0.7. Delete (y) or Keep (k)? "
                    )
                    .strip()
                    .lower()
                )
                if answer == "y":
                    move_removed_file(
                        identity,
                        img_path,
                        f"confidence {confidence:.2f} < 0.7 - manually removed",
                    )
                    break
                elif answer == "k":
                    df_crop_filtered.append(row)
                    break
                else:
                    print("Invalid input. Please type 'y' to delete or 'k' to keep.")

    df_crop_filtered = pd.DataFrame(df_crop_filtered)
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

            while True:
                answer = (
                    input(
                        f"Image {img_path} has width > height. Delete (y), rotate right (r), rotate left (l), or do nothing (n)? "
                    )
                    .strip()
                    .lower()
                )
                if answer == "y":
                    move_removed_file(
                        identity, img_path, "width > height - manually removed"
                    )
                    break
                elif answer == "r":
                    try:
                        rotated = img.rotate(
                            -90, expand=True
                        )
                        rotated.save(img_path)
                        row["width"], row["height"] = height, width
                        final_rows.append(row)
                    except Exception as e:
                        move_removed_file(
                            identity, img_path, f"error while rotating right: {e}"
                        )
                    break
                elif answer == "l":
                    try:
                        rotated = img.rotate(
                            90, expand=True
                        )
                        rotated.save(img_path)
                        row["width"], row["height"] = height, width
                        final_rows.append(row)
                    except Exception as e:
                        move_removed_file(
                            identity, img_path, f"error while rotating left: {e}"
                        )
                    break
                elif answer == "n":
                    final_rows.append(row)
                    break
                else:
                    print("Invalid input. Please type 'y', 'r', 'l', or 'n'.")

    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(f"{DATASET_NAME}_cleaned_dataset.csv", index=False)

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
            print(f"Could not delete {f}: {e}")


if __name__ == "__main__":
    rows = process_images()
    df_crop = crop_images(rows)
    df_crop_filtered = filter_by_confidence(df_crop)
    df_final = filter_by_aspect_ratio(df_crop_filtered)

    print(f"Done: {len(df_final)} final images | {len(removed_entries)} removed")

    while True:
        answer = (
            input("Do you want to save 'removed_images_log.csv'? (y/n): ")
            .strip()
            .lower()
        )
        if answer == "y":
            df_removed = pd.DataFrame(removed_entries)
            df_removed.to_csv(f"{DATASET_NAME}_removed_images_log.csv", index=False)
            print("Saved 'removed_images_log.csv'.")
            break
        elif answer == "n":
            print("Skipped saving 'removed_images_log.csv'.")
            break
        else:
            print("Invalid input. Please type 'y' or 'n'.")

    while True:
        answer = (
            input(f"Do you want to delete the {REMOVED_DIR} folder? (y/n): ")
            .strip()
            .lower()
        )
        if answer == "y":
            try:
                shutil.rmtree(REMOVED_DIR)
                print(f"Deleted {REMOVED_DIR} folder.")
            except Exception as e:
                print(f"Error deleting folder: {e}")
            break
        elif answer == "n":
            print(f"Kept {REMOVED_DIR} folder.")
            break
        else:
            print("Invalid input. Please type 'y' or 'n'.")

    cleanup()
