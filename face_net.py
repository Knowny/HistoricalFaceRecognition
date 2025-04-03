# filename: face_net.py
# pretrained model: InceptionResnetV1
# This code uses the InceptionResnetV1 model from the facenet-pytorch library to compute face embeddings
# project: KNN Face Recognition
# version: 2.0
# author: xmager00

from facenet_pytorch import InceptionResnetV1
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import argparse

import data_cleanup
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               DataLoader
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class FaceNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit_identities=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.limit_identities = limit_identities

        self.person_images = {}

        person_dirs = sorted(os.listdir(root_dir))
        if limit_identities is not None:
            person_dirs = person_dirs[:limit_identities]

        for person_dir in person_dirs:
            person_path = os.path.join(root_dir, person_dir)
            if os.path.isdir(person_path):
                person_images = []
                for file in os.listdir(person_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(person_path, file)
                        person_images.append(image_path)
                        self.labels.append(person_dir)
                self.person_images[person_dir] = person_images

        # ++++++++++++++++ Pairs +++++++++++++++++++

        self.pairs = []
        self.pairs_labels = []

        num_persons = len(person_dirs)

        for i, person_dir in enumerate(person_dirs):
            images = self.person_images[person_dir]
            if len(images) >= 2:
                self.pairs.append((images[0], images[1]))
                self.pairs_labels.append(1)  # positive pair

            next_person_dir = person_dirs[(i + 1) % num_persons - 1]
            if self.person_images.get(next_person_dir) and self.person_images[next_person_dir]:
                self.pairs.append((images[0], self.person_images[next_person_dir][0]))
                self.pairs_labels.append(0)  # negative pair

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample1_path, sample2_path = self.pairs[idx]
        label = self.pairs_labels[idx]

        sample1 = Image.open(sample1_path)
        sample2 = Image.open(sample2_path)

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
    parser.add_argument("--limit_identities", type=int, default=None, help="Limit the number of pairs to process. They are not that many (on WikiFace data)")    
    args = parser.parse_args()

    image_dir = "../data/WikiFaceCropped"
    json_dir = "../data/WikiFaceOutput/detections"
    image_dir_clean = "../data/WikiFaceCropped_clean"

    if not os.path.exists(image_dir_clean):
        root_dir = data_cleanup.get_clean_data(image_dir, json_dir, image_dir_clean, confidence_threshold=0.5)
        print(f"Cleaned data saved to: {root_dir} successfully.")
        if not root_dir:
            print("Cleanup Failed.")
            return
    else:
        root_dir = image_dir_clean
        print(f"Clean data directory already exists, using: {root_dir}")
    
    dataset = FaceNetDataset(root_dir, transform=transform, limit_identities=args.limit_identities)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    def get_embedding(tensor):
        with torch.no_grad():
            embedding = model(tensor)
        return embedding.cpu().numpy()

    all_similarities = []
    all_labels = []

    loop = tqdm(dataloader, desc="Processing Batches", unit=" batch")

    for sample1_batch, sample2_batch, labels_batch in loop:
        embedding1 = get_embedding(sample1_batch)
        embedding2 = get_embedding(sample2_batch)

        for i in range(len(embedding1)):
            similarity = cosine_similarity(embedding1[i].reshape(1, -1), embedding2[i].reshape(1, -1))[0][0] 
            all_similarities.append(similarity)
            all_labels.append(labels_batch[i].item())

    print("Metrics implementation is under construction. But this one finished. GL HF")

if __name__ == "__main__":
    main()