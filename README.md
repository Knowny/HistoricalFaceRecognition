# HistoricalFaceRecognition
The goal of this project was to adapt a CNN solution to work with historical document on the tsak of facial recognition/verification.

The theoretical background can be found in the project report disclosed with the project.

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
## Getting Started
To get started with the pipeline first clone the repository. Then install all necessary requirements
```
pip install -r requirements.txt
pip install -r finetuning_requirements.txt
pip install -r extration_requirements.txt
```
The use of an environment (Anaconda/Miniconda or venv) is highly recommended

> **Data**
>
> The data sources are **not** a part of this repository. <br>
> However a part of this project was about providing a cleaned and aligned dataset 
> which can be downloaded from here <br>
>
> After download please refer to the [repo structure](#repo-structure). <br>

## Evaluation
The evaluation metrics described in the report can be found in `eval.py`<br>
You can run it by
```
python eval.py --data <data_dir> --model <path_to_model> 
```
## Style Transfer
The style transfer has a high requirements on GPU resources ()

For personal run please download `train.rec` and `train.idx` from [kaggle](https://www.kaggle.com/datasets/debarghamitraroy/casia-webface) and save them to the `style_tranfer` folder provided.

The utility for sample extraction `extract_casia.py` is set as was in the project (*1000 identities each having 15 samples*). For different setting please edit the code directly. 

The prompt bank used is located in `prompts.txt`. The total number of prompts **must** stay the same as the number of samples.

```
python extract_casia.py
python run_photomaker.py
```

Any parameter tuning to the PhotoMaker run are possible within the `run_photomaker.py` script

> **Results** <br>
> 
> The results are already provided in the Data part. For a small sample refer to the examples folder
## Baseline
The baseline solution was evaluated for FaceNet and AdaFace model. The implementations can be found in the `./baseline` folder of the repository. 

In early stages of the project these were used separately, for a quick evaluation of FaceNet it is possible to use the [evaluation](#evaluation) util and set `--model baseline` 

## Face Alignment

Aligns faces based on outputs of the YOLO face detector using transformations.

- input: dataset containing face images
- output: dataset containing aligned face images (image resolution: 112x112px)
................
- usage: run `align_faces.py` (don't forget to change the paths in the file as needed)


## Dataset Cleaning

The datasets (both WikiFace and stylized_images) had to be cleaned from the false positives by manual inspection.

Helper scripts used during the cleanup:
- `get_stats_freom_csv.py`: 
    - prints the basic statistics of the dataset
    - prints paths to images: containing multiple identities / low detection confidence / having no pair to be compared with 
- `create_csv_after_cleaning`: 
    - creates a new CSV containing metadata of the dataset after the cleanup 


## Finetuning
To work with the finetuning pipeline it is required to have both the WikiFace and stylized dataset ready in the cleaned and aligned form (refer to Data)

After it can be run like so with the parameters from project set as default
```
python finetune_facenet.py [args]
```
for different parameter settings use these arguments
```
--data_dir [path]   # path to the root od stylized dataset
--lr [float]        # inital learning rate
--epochs [int]
--batch_size [int]   
--margin [float]    # margin from triplet loss
--ckpt [path]       # where to save the finetuned .pth file
```

To play with more fine hyperparameters please do so in file `finetune_facenet.py`

## Authors
This project was a term assignment for KNN 24/25L at BUT FIT

Tomas Husar(xhusar11)
Tereza Magerkova (xmager00)
Simona Janosikova (xjanos19)