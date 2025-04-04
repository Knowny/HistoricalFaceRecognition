import os
import json
import cv2
import jsonlines  # Import the jsonlines library

# * This program takes the output of the YOLO11 detector (predicted bounding boxes) and crops the original dataset

def crop_face(image, detection):

    x_center = detection.get('x_center')
    y_center = detection.get('y_center')
    width = detection.get('bounding_box_width')
    height = detection.get('bounding_box_height')

    if x_center is None or y_center is None or width is None or height is None:
        print(f"Warning: Incomplete detection data: {detection}")
        return None

    x = int(x_center - width / 2)
    y = int(y_center - height / 2)
    w = int(width)
    h = int(height)

    # check if bounding box is valid
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        print(f"Warning: Invalid bounding box: x={x}, y={y}, w={w}, h={h}, image_shape={image.shape}")
        return None

    face_crop = image[y:y + h, x:x + w]
    
    # ! No resize yet, the images will be preprocessed as neccesary for each CNN differently :)

    return face_crop

def process_images_and_detections(images_dir, detections_dir, output_dir):

    for folder_name in os.listdir(images_dir):
        image_folder = os.path.join(images_dir, folder_name)
        detection_folder = os.path.join(detections_dir, folder_name)
        output_face_folder = os.path.join(output_dir, folder_name)

        if not os.path.isdir(image_folder) or not os.path.isdir(detection_folder):
            continue

        os.makedirs(output_face_folder, exist_ok=True)

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                json_filename = os.path.splitext(filename)[0] + ".jsonl"
                json_path = os.path.join(detection_folder, json_filename)

                if os.path.exists(json_path):
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Error: Could not read image from {image_path}")
                        continue

                    try:
                        with jsonlines.open(json_path) as reader:
                            detection_count = 0
                            for detection in reader:
                                cropped_face = crop_face(image, detection)
                                if cropped_face is not None:
                                    output_filename = f"{os.path.splitext(filename)[0]}_face_{detection_count}.jpg"  # Add index for multiple faces
                                    output_face_path = os.path.join(output_face_folder, output_filename)
                                    cv2.imwrite(output_face_path, cropped_face)
                                    # print(f"Cropped and saved {output_face_path}")
                                    detection_count += 1
                                else:
                                    print(f"Warning: Invalid detection in {json_path} for {image_path}")
                            if detection_count == 0:
                                print(f"Warning: No valid detections found in {json_path} for {image_path}")
                    except FileNotFoundError:
                        print(f"Error: Detection file not found: {json_path}")
                    except jsonlines.jsonlines.InvalidJSONError as e:
                        print(f"Error: Invalid JSON line in {json_path}: {e}")
                else:
                    print(f"Warning: No detection file found for {filename}")

# edit the relative paths to images and detections if necessary
images_directory = "../datasets/WikiFace"
detections_directory = "../datasets/WikiFaceDetectionOutput/detections"
output_faces_directory = "../datasets/WikiFaceCropped2"

process_images_and_detections(images_directory, detections_directory, output_faces_directory)