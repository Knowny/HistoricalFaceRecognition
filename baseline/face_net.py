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
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt

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
        self.person_images = {}
        person_dirs = sorted(os.listdir(root_dir))
        if limit_identities is not None:
            person_dirs = person_dirs[:limit_identities]

        for person_dir in person_dirs:
            person_path = os.path.join(root_dir, person_dir)
            if os.path.isdir(person_path):
                image_paths = sorted([
                    os.path.join(person_path, f)
                    for f in os.listdir(person_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                self.person_images[person_dir] = image_paths

    def __len__(self):
        # Total number of images
        return sum(len(v) for v in self.person_images.values())

    def get_all_items(self):
        # Returns list of (image_path, person_id)
        items = []
        for pid, paths in self.person_images.items():
            for p in paths:
                items.append((p, pid))
        return items

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               FaceNet
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# * Pretrained
# model = InceptionResnetV1(pretrained='vggface2').eval()

# * finetuned model
model = InceptionResnetV1(pretrained=None, classify=False)
model.load_state_dict(torch.load("/home/tomas/1mit/knn/HistoricalFaceRecognition/models/finetuned_facenet/finetuned_facenet_01.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    # * Achieves higher AUC without the normalization
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               Main
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main():
    parser = argparse.ArgumentParser(description="FaceNet face verification.")
    parser.add_argument("--limit_identities", type=int, default=None, help="Limit the number of identities to process. There are not that many (on WikiFace data)")    
    parser.add_argument("--print_similarities", type=bool, default=False, help="Prints pair labels and cosine similarity.")
    parser.add_argument("--evaluate_model", type=bool, default=True, help="Plot the ROC and DET curves based on the labels and model predictions.")
    parser.add_argument("--dataset_root", type=str, default="../datasets/wiki_face_112_fin", help="Path to the root directory of the (cleaned) dataset images (e.g., ../datasets/WikiFaceCleaned).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding computation.")
    args = parser.parse_args()

    dataset = FaceNetDataset(args.dataset_root, transform=transform, limit_identities=args.limit_identities)

    def get_embedding(tensor):
        # Obtain embeddings from the FaceNet model without gradient tracking
        with torch.no_grad():
            return model(tensor).cpu().numpy()

    # 1) Gather all images and their labels
    items = dataset.get_all_items()
    image_paths, labels_list = zip(*items)
    image_paths = list(image_paths)
    labels_list = list(labels_list)

    # 2) Compute embeddings in batches
    embeddings = []
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Embedding images"):
        batch = image_paths[i:i+args.batch_size]
        tensors = [transform(Image.open(p).convert('RGB')) for p in batch]
        batch_tensor = torch.stack(tensors)
        embeddings.append(get_embedding(batch_tensor))
    all_embeddings = np.vstack(embeddings)

    # 3) Compute cosine similarity matrix between all embeddings
    sim_matrix = cosine_similarity(all_embeddings)

    # 4) Build lists of similarity scores and ground-truth labels (all-vs-all pairs)
    all_similarities = []
    all_labels = []
    n = sim_matrix.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            all_similarities.append(sim_matrix[i, j])
            # Label = 1 if same identity, 0 if different identities
            all_labels.append(int(labels_list[i] == labels_list[j]))

    # Write each pair index, similarity and label
    if args.print_similarities:
        print("All-vs-All Pair Similarities:")
        idx = 1
        for i in range(n):
            for j in range(i+1, n):
                label = int(labels_list[i] == labels_list[j])
                print(f"Pair {idx}: ({i},{j}), sim={sim_matrix[i,j]:.4f}, label={label}")
                idx += 1

    # Plot the full cosine similarity matrix for visualization
    plt.figure()
    plt.imshow(sim_matrix)
    plt.title('Cosine Similarity Matrix')
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    plt.colorbar()
    plt.savefig(f"cosine_similarity_matrix_facenet.png")

    # Model evaluation (plot the ROC and DET curves)
    if args.evaluate_model == True:

        # Convert lists to numpy arrays
        labels = np.array(all_labels)
        similarities = np.array(all_similarities)

        roc_det_plot(labels, similarities, MODEL_NAME)

if __name__ == "__main__":
    main()
