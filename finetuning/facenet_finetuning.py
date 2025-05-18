# facenet_finetuning.py

import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import numpy as np

# Seed, so the results can be reproduced
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

# Triplet Dataset Loader
class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.person_images = self._load_images()
        # * number of images in the dataset
        self.image_count = sum(len(imgs) for imgs in self.person_images.values())

    def _load_images(self):
        person_images = {}
        for person in os.listdir(self.root_dir):
            full_path = os.path.join(self.root_dir, person)
            if os.path.isdir(full_path):
                images = [os.path.join(full_path, f)
                          for f in os.listdir(full_path)
                          if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                # Needs at least two images to create a triplet
                if len(images) >= 2:
                    person_images[person] = images
        return person_images

    def __len__(self):
        return self.image_count

    def __getitem__(self, idx):
        persons = list(self.person_images.keys())
        anchor_person = random.choice(persons)
        anchor_img, positive_img = random.sample(self.person_images[anchor_person], 2)

        negative_person = random.choice([p for p in persons if p != anchor_person])
        negative_img = random.choice(self.person_images[negative_person])

        def load(path):
            img = Image.open(path).convert('RGB')
            return self.transform(img) if self.transform else img

        return load(anchor_img), load(positive_img), load(negative_img)

# Main training function
def train_facenet(dataset_root, save_path, num_epochs=5, batch_size=32, lr=1e-4, margin=0.5):

    # Set seed to ensure reproducibility
    set_seed(42)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
        # * Model achieves higher results without the normalization
    ])

    dataset = TripletFaceDataset(dataset_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load pretrained FaceNet model
    model = InceptionResnetV1(pretrained='vggface2', classify=False)
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = TripletLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            anchor, positive, negative = [b.to(device) for b in batch]

            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Usage
if __name__ == "__main__":
    train_facenet(
        dataset_root="../datasets/stylized_images_112_fin",
        save_path="../models/finetuned_facenet/finetuned_facenet_08.pth",
        num_epochs=20,
        batch_size=32,
        lr=5e-6,
        margin=0.3
    )
