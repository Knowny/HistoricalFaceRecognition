#!/usr/bin/env python3
"""
This code uses the InceptionResnetV1 model from the facenet-pytorch library to compute face embeddings

filename: face_net.py
pretrained model: InceptionResnetV1
project: KNN Face Recognition
version: 2.0
author: Tereza Magerkova, xmager00
"""
from facenet_pytorch import InceptionResnetV1
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
import numpy as np

from utils.baseline_evaluate import roc_det_plot

# import knn.HistoricalFaceRecognition.utils.dataset_cleanup as dataset_cleanup

MODEL_NAME = "FaceNet"

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               DataLoader
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class FaceNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit_identities=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []
        self.pairs_labels = []
        self.pairs_label_names = []  # sanity check

        self.person_images = {}
        person_dirs = sorted(os.listdir(root_dir))
        if limit_identities is not None:
            person_dirs = person_dirs[:limit_identities]

        for person_dir in person_dirs:
            person_path = os.path.join(root_dir, person_dir)
            if os.path.isdir(person_path):
                image_paths = sorted([os.path.join(person_path, file)
                                      for file in os.listdir(person_path)
                                      if file.lower().endswith(('.png', '.jpg', '.jpeg'))])
                self.person_images[person_dir] = image_paths

        self._create_pairs(person_dirs)

    def _create_pairs(self, person_dirs):
        num_persons = len(person_dirs)
        for i, person_dir in enumerate(person_dirs):
            images = self.person_images[person_dir]

            if len(images) >= 2:
                next_images = None
                start = (i + 1) % num_persons
                for j in range(num_persons):
                    next_person_dir = person_dirs[(start + j) % num_persons]
                    if next_person_dir != person_dir:
                        next_images = self.person_images.get(next_person_dir)
                        if next_images and len(next_images) > 0:
                            break
                
                if next_images and len(next_images) > 0:
                    self.pairs.append((images[0], next_images[0]))
                    self.pairs_labels.append(0)
                    self.pairs_label_names.append((person_dir, next_person_dir)) # sanity check

                    self.pairs.append((images[0], images[1]))
                    self.pairs_labels.append(1)
                    self.pairs_label_names.append((person_dir, person_dir)) # sanity check
            else:
                continue

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample1_path, sample2_path = self.pairs[idx]
        label = self.pairs_labels[idx]

        try:
            sample1 = Image.open(sample1_path).convert('RGB')
            sample2 = Image.open(sample2_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: File not found for pair at index {idx}: {sample1_path}, {sample2_path}")

        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        return sample1, sample2, label
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               FaceNet
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

model = InceptionResnetV1(pretrained='vggface2').eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               Main
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main():
    parser = argparse.ArgumentParser(description="FaceNet face verification.")
    parser.add_argument("--limit_identities", type=int, default=None, help="Limit the number of identities to process. There are not that many (on WikiFace data)")    
    parser.add_argument("--print_similarities", type=bool, default=False, help="Prints pair labels and cosine similarity.")
    parser.add_argument("--evaluate_model", type=bool, default=False, help="Plot the ROC and DET curves based on the labels and model predictions.")
    parser.add_argument("--dataset_root", type=str, default="../datasets/WikiFaceCleaned", help="Path to the root directory of the (cleaned) dataset images (e.g., ../datasets/WikiFaceCleaned).")
    args = parser.parse_args()
    
    # Path defined via argument (default: ../datasets/WikiFaceCleaned)
    root_dir = args.dataset_root

    dataset = FaceNetDataset(root_dir, transform=transform, limit_identities=args.limit_identities)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    def get_embedding(tensor):
        with torch.no_grad():
            embedding = model(tensor)
        return embedding.cpu().numpy()

    all_similarities = []
    all_labels = []

    loop = tqdm(dataloader, desc="Processing Batches", unit=" batch")

    # ! CHANGE - neporovnavat pary ... ale KAZDE S KAZDYM ... (neporovnavat identicke fotky)
    for sample1_batch, sample2_batch, labels_batch in loop:
        embedding1 = get_embedding(sample1_batch)
        embedding2 = get_embedding(sample2_batch)

        for i in range(len(embedding1)):
            similarity = cosine_similarity(embedding1[i].reshape(1, -1), embedding2[i].reshape(1, -1))[0][0] 
            all_similarities.append(similarity)
            all_labels.append(labels_batch[i].item())
        
    # * Similarities Printing
    if args.print_similarities == True:

        print("Pair Labels and Similarity:")
        for i in range(len(dataset.pairs)):
            label_pair = dataset.pairs_label_names[i]
            similarity = all_similarities[i]
            label = all_labels[i]
            print(f"Pair {i+1}: Labels {label_pair}, Similarity: {similarity}, Label: {label}")

    # ! CHANGE

    # * Model evaluation (plot the ROC and DET curves)
    if args.evaluate_model == True:

        # Convert lists to numpy arrays
        labels = np.array(all_labels)
        similarities = np.array(all_similarities)

        roc_det_plot(labels, similarities, MODEL_NAME)

if __name__ == "__main__":
    main()