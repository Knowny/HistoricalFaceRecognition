# filename: adaface.py
# pretrained model: ir50_ms1mv2
# This code uses the ir50_ms1mv2 model from the adaface-pytorch library to compute face embeddings
# project: KNN Face Recognition
# version: 1.0
# author: xjanos19

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
import net
from utils.baseline_evaluate import roc_det_plot

adaface_models = {
    'ir_50': "../models/adaface_ir50_ms1mv2.ckpt.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    checkpoint = torch.load("../models/adaface_ir50_ms1mv2.ckpt", map_location=torch.device('cpu'), weights_only=False)
    statedict = checkpoint['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

class WikiFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit_identities=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []
        self.pairs_labels = []
        self.pairs_label_names = []
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
                start = (i + 1) % num_persons
                for j in range(num_persons):
                    next_person_dir = person_dirs[(start + j) % num_persons]
                    if next_person_dir != person_dir:
                        next_images = self.person_images.get(next_person_dir)
                        if next_images and len(next_images) > 0:
                            break
                if next_images:
                    self.pairs.append((images[0], next_images[0]))
                    self.pairs_labels.append(0)
                    self.pairs_label_names.append((person_dir, next_person_dir))

                    self.pairs.append((images[0], images[1]))
                    self.pairs_labels.append(1)
                    self.pairs_label_names.append((person_dir, person_dir))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.pairs_labels[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def main():
    parser = argparse.ArgumentParser(description="AdaFace face verification")
    parser.add_argument("--limit_identities", type=int, default=None)
    parser.add_argument("--print_similarities", type=bool, default=False)
    parser.add_argument("--evaluate_model", type=bool, default=False)
    parser.add_argument("--dataset_root", type=str, default="../datasets_examples/WikiFaceCleaned")
    args = parser.parse_args()

    model = load_pretrained_model()
    dataset = WikiFaceDataset(args.dataset_root, transform=transform, limit_identities=args.limit_identities)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    def get_embedding(x):
        with torch.no_grad():
            output = model(x)
            if isinstance(output, tuple):
                embeddings = output[0]
            else:
                embeddings = output
        return embeddings.cpu().numpy()


    all_similarities = []
    all_labels = []

    loop = tqdm(dataloader, desc="Processing batches")

    for img1_batch, img2_batch, labels_batch in loop:
        emb1 = get_embedding(img1_batch)
        emb2 = get_embedding(img2_batch)

        for i in range(len(emb1)):
            similarity = cosine_similarity(emb1[i].reshape(1, -1), emb2[i].reshape(1, -1))[0][0]
            all_similarities.append(similarity)
            all_labels.append(labels_batch[i].item())

    if args.print_similarities:
        for i in range(len(dataset.pairs)):
            label_pair = dataset.pairs_label_names[i]
            print(f"Pair {i+1}: {label_pair}, Similarity: {all_similarities[i]:.4f}, Label: {all_labels[i]}")

    if args.evaluate_model:
        roc_det_plot(np.array(all_labels), np.array(all_similarities))

if __name__ == "__main__":
    main()
