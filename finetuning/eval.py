"""
Finetuning experiments evaluation
1) get baseline evaluation
2) finetune the pretrained model
3) reevaluate on the new model

file: .py
project: KNN Face Recognition
author: Tereza Magerkova, xmager00
"""

import argparse
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, det_curve, roc_auc_score

# from baseline/face_net.py
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate finetuning experiments on FaceNet")
    parser.add_argument('--data_dir', type=str, default="../datasets/wiki_face_112_fin", help="Path to the test set (to evaluate)")
    parser.add_argument('--model', type=str, default="../models/finetuned_facenet/finetuned_facenet_01.pth", help="Path to model")

    return parser.parse_args()
def main():
    args = parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    ])
    dataset = FaceNetDataset(args.data_dir, transform=transform)

    if args.model == "baseline":
         model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
    else:
        model = InceptionResnetV1(pretrained=None, classify=False).to(device)
        ckpt = torch.load(args.model, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        model.eval()

    # ------------------ taken from baseline/face_net.py ------------------
    def get_embedding(tensor):
        with torch.no_grad():
            tensor = tensor.to(device)  # <-- move input to model's device
            return model(tensor).cpu().numpy()

    # gater images and labels
    items = dataset.get_all_items()
    image_paths, labels_list = zip(*items)
    image_paths = list(image_paths)
    labels_list = list(labels_list)

    # embeddings in batches
    embeddings = []
    for i in tqdm(range(0, len(image_paths), 32), desc="Embedding images"):
        batch = image_paths[i:i+32]
        tensors = [transform(Image.open(p).convert('RGB')) for p in batch]
        batch_tensor = torch.stack(tensors)
        embeddings.append(get_embedding(batch_tensor))
    all_embeddings = np.vstack(embeddings)

    sim_matrix = cosine_similarity(all_embeddings)

    # build lists of similarities for pairs
    all_similarities = []
    all_labels = []
    n = sim_matrix.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            all_similarities.append(sim_matrix[i, j])
            # Label = 1 if same identity, 0 if different identities
            all_labels.append(int(labels_list[i] == labels_list[j]))

    # ---------------------------------------------------------------------
    y_scores = np.array(all_similarities)
    y_true   = np.array(all_labels)

    # ROC curve fpr - false positive rate / tpr - true positive rate, fnr - false negative rate
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    auc = roc_auc_score(y_true, y_scores)
    print(f"ROC area under curve: {auc:.4f}")
    # DET curve
    det_fpr, det_fnr, _ = det_curve(y_true, y_scores)

    # EER - points where fpr ~ fnr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    print(f"Equal Error Rate (EER): {eer:.4f}")

    fmrs = [0.1, 0.01, 0.001, 0.0001]
    for target in fmrs:
        idx        = np.nanargmin(np.abs(fpr - target))
        tar        = tpr[idx]
        print(f"TAR@FMR={target}: {tar:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC (TPR vs FPR)")
    plt.plot(det_fpr, det_fnr,  label="DET (FNR vs FPR)", linestyle='--')
    plt.scatter(fpr[eer_idx], tpr[eer_idx], marker='o')
    plt.annotate(f"EER ≈ {eer:.4f}",
                (fpr[eer_idx], tpr[eer_idx]),
                textcoords="offset points", xytext=(5,-10))

    plt.xlabel("False Positive Rate")
    plt.ylabel("Rate")
    plt.title("ROC & DET Curves with EER")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ROC_DET_FINETINE.png")

if __name__ == "__main__":
    main()