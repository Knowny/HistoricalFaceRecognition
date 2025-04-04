# HistoricalFaceRecognition
The goal of the project is to train/adapt CNN models for facial feature extraction for identification in historical portraits such as books and magazines...

## Repo structure
```
archival_faces_detector/        # (provided by our supervisor)
├── archival_faces_model.pt     # not provided in this repo
└── run_inference.py            # face detector based on YOLO11

baseline/
├── utils/
│   └── baseline_evaluate.py    # plotting of the ROC and DET
├── face_net.py                 # baseline sollution
└── roc_det_curves.png          # graph of baseline evaluation

dataset_tools/
├── dataset_cleanup.py          # Cleaning of the false positives (needs some more work)
├── dataset_crop.py             # Cropping based on the detection results
├── dataset_statistics.py       # Basic datasets statistics from the detections 
└── dataset_statistics.txt      # output of the dataset_statistics.py

datasets_examples/
├── WikiFace                    # Original dataset (provided by our supervisor)
├── WikiCleaned                 # output of the dataset_cleanup.py
├── WikiCropped                 # output of the dataset_crop.py
└── WikiDetectionOutput         # Output of the ArchivalFacesDetector
```
## Getting started
- **git clone**
- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- TBD

## Backbone
- TBD

## Datasets

### Training datasets
- TODO

### Testing datasets
- WikiFace: TODO

## Evaluation
- TBD

## Sources
- A Survey of Face Recognition: https://arxiv.org/abs/2212.13038
- (Anil Poudel) Face recognition on historical photographs: https://uu.diva-portal.org/smash/record.jsf?pid=diva2%3A1622968&dswid=-6352
