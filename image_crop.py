import os
import json
import cv2

def crop_and_resize_face(image, detection, target_size=(160, 160)):

    x_center = detection['x_center']
    y_center = detection['y_center']
    width = detection['bounding_box_width']
    height = detection['bounding_box_height']

    x = int(x_center - width / 2)
    y = int(y_center - height / 2)
    w = int(width)
    h = int(height)

    # check if bounding box is valid
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        return None

    face_crop = image[y:y + h, x:x + w]
    face_resized = cv2.resize(face_crop, target_size)

    return face_resized

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

                    with open(json_path, 'r') as f:
                        detection = json.load(f)

                    cropped_face = crop_and_resize_face(image, detection)

                    if cropped_face is not None:
                        output_face_path = os.path.join(output_face_folder, filename)
                        cv2.imwrite(output_face_path, cropped_face)
                        # print(f"Cropped and saved {output_face_path}")
                    else:
                        print(f"Warning: Invalid detection for {filename}")
                else:
                    print(f"Warning: No detection file found for {filename}")

# edit the relative paths to images and detections
images_directory = "ArchivalFacesDetector/examples/images"
detections_directory = "ArchivalFacesDetector/examples/outputs/detections"
output_faces_directory = "cropped_faces"

process_images_and_detections(images_directory, detections_directory, output_faces_directory)