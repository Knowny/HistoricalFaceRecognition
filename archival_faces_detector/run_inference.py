from ultralytics import YOLO
from ultralytics.engine.results import Results
import argparse
from pydantic import BaseModel
from pathlib import Path
from typing import Iterable


class SingleDetectionResult(BaseModel):
    confidence: float
    # Bounding box
    x_center: float
    y_center: float
    bounding_box_width: float
    bounding_box_height: float
    # Keypoints
    right_eye_x: float | None
    right_eye_y: float | None
    left_eye_x: float | None
    left_eye_y: float | None
    nose_x: float | None
    nose_y: float | None
    right_mouth_x: float | None
    right_mouth_y: float | None
    left_mouth_x: float | None
    left_mouth_y: float | None


def run_inference_on_image(image_path: Path, model: YOLO):
    results: Results = model(image_path, verbose=False)[0]
    results = results.cpu().numpy()

    output = []
    for confidence, bounding_box, keypoints in zip(results.boxes.conf,  results.boxes.xywh, results.keypoints.xy):
        output.append(SingleDetectionResult(
            confidence=confidence,
            x_center=bounding_box[0],
            y_center=bounding_box[1],
            bounding_box_width=bounding_box[2],
            bounding_box_height=bounding_box[3],
            right_eye_x=keypoints[0][0],
            right_eye_y=keypoints[0][1],
            left_eye_x=keypoints[1][0],
            left_eye_y=keypoints[1][1],
            nose_x=keypoints[2][0],
            nose_y=keypoints[2][1],
            right_mouth_x=keypoints[3][0],
            right_mouth_y=keypoints[3][1],
            left_mouth_x=keypoints[4][0],
            left_mouth_y=keypoints[4][1]
        ))

    return output


def draw_detections_on_image(image_path: Path, detections: Iterable[SingleDetectionResult], output_image_path: Path):
    import cv2
    image = cv2.imread(str(image_path))
    for detection in detections:
        x_center = int(detection.x_center)
        y_center = int(detection.y_center)
        bounding_box_width = int(detection.bounding_box_width)
        bounding_box_height = int(detection.bounding_box_height)

        top_left = (x_center - bounding_box_width // 2,
                    y_center - bounding_box_height // 2)
        bottom_right = (x_center + bounding_box_width // 2,
                        y_center + bounding_box_height // 2)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        keypoints = [
            (
                int(detection.right_eye_x),
                int(detection.right_eye_y),
                (255, 0, 0)
            ),
            (
                int(detection.left_eye_x),
                int(detection.left_eye_y),
                (0, 255, 0)
            ),
            (
                int(detection.nose_x),
                int(detection.nose_y),
                (0, 0, 255)
            ),
            (
                int(detection.right_mouth_x),
                int(detection.right_mouth_y),
                (255, 255, 0)
            ),
            (
                int(detection.left_mouth_x),
                int(detection.left_mouth_y),
                (0, 255, 255)
            ),
        ]
        for (x, y, color) in keypoints:
            cv2.circle(image, (x, y), 5, color=color, thickness=-1)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image_path), image)


def save_detections_to_jsonl(detections: Iterable[SingleDetectionResult], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as output_file:
        for detection in detections:
            print(detection.model_dump_json(), file=output_file)


def run_inference_on_image_iterator(image_iterator: Iterable[Path], model: YOLO):
    for image_path in image_iterator:
        yield image_path, run_inference_on_image(image_path, model)


# ! default: use_tqdm: bool = False; image_directory.rglob("*.jpg"))
# unfortunatley ... Path.rglob() doesn’t support multiple patterns in a single string
def run_inference_on_image_directory(image_directory: Path, model: YOLO, use_tqdm: bool = True):
    image_iterator = filter(Path.is_file, image_directory.rglob("*.jpg"))
    if use_tqdm:
        from tqdm import tqdm
        image_iterator = tqdm(image_iterator, desc="Processing images")

    yield from run_inference_on_image_iterator(image_iterator, model)


def run_inference(
    model_path: Path,
    images_directory: Path,
    output_directory: Path,
    output_images_directory: Path,
    keep_relative_directory_structure: bool,
    use_tqdm: bool,
    device: str
):
    model: YOLO = YOLO(model_path, verbose=False).to(device)

    for image_path, detections in run_inference_on_image_directory(images_directory, model, use_tqdm=use_tqdm):
        output_path = (output_directory / (
            image_path.relative_to(
                images_directory) if keep_relative_directory_structure else image_path.name
        )).with_suffix(".jsonl")
        output_image_path = (output_images_directory / (
            image_path.relative_to(
                images_directory) if keep_relative_directory_structure else image_path.name
        )).with_suffix(".jpg")

        save_detections_to_jsonl(detections, output_path)
        draw_detections_on_image(image_path, detections, output_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path,
                        default=Path("ArchivalFaces_2024_08_07_fold_0_yolo11l.pt"))
    parser.add_argument("--images_directory", type=Path,
                        default=Path("examples/images"))
    parser.add_argument("--output_directory", type=Path,
                        default=Path("examples/outputs/detections"))
    parser.add_argument("--output_images_directory", type=Path,
                        default=Path("examples/outputs/images"))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    run_inference(
        args.model_path,
        args.images_directory,
        args.output_directory,
        args.output_images_directory,
        True,
        True,
        args.device
    )
