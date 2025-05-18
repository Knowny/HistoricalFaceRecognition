# fine_tune_facenet.py

import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import argparse


# Seed, so the results can be reproduced
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Triplet Loss (FaceNet original paper)
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
        self.image_count = sum(len(imgs) for imgs in self.person_images.values())

    def _load_images(self):
        person_images = {}
        for person in os.listdir(self.root_dir):
            full_path = os.path.join(self.root_dir, person)
            if os.path.isdir(full_path):
                images = [os.path.join(full_path, f)
                          for f in os.listdir(full_path)
                          if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                if len(images) >= 2:  # Need at least two images to create a triplet
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
    
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune")
    parser.add_argument('--data_dir', type=str, default="./data/stylized", help="Path to the train data")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch Size")
    parser.add_argument('--margin', type=float, default=0.5, help="Margin for TripletLoss")
    parser.add_argument('--ckpt', type=str, default="./models/model.pth", help="Save path for the model checkpoint")

    return parser.parse_args()

# Main training function
def train_facenet():
    args = parse_args()
    set_seed(42)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    dataset = TripletFaceDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model: load pretrained FaceNet
    model = InceptionResnetV1(pretrained='vggface2', classify=False)
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = TripletLoss(margin=args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(dataloader)
    max_lr = 1e-3
    final_lr = args.lr * 0.1
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=total_steps,
        epochs=args.epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=(max_lr/args.lr),
        final_div_factor=(max_lr/final_lr)
    )

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"[TRAIN]", unit="batch"):
            anchor, positive, negative = [b.to(device) for b in batch]

            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"avg_loss:{avg_loss:.4f}")

    torch.save(model.state_dict(), args.ckpt)
    print(f"Model saved to {args.ckpt}")

# Example usage
if __name__ == "__main__":
    train_facenet()