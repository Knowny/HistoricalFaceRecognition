#!/usr/bin/env python3
"""
Finetuning of the FaceNet baseline model

code is inspired by https://github.com/timesler/facenet-pytorch/blob/master/examples/finetune.ipynb

file: finetune_facenet.py
project: KNN Face Recognition
version: 1.0
author: Tereza Magerkova, xmager00
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1, training, fixed_image_standardization
import numpy as np

DATA_DIR = "../../data/casia_stylized"
BATCH_SIZE = 32
HEAD_ONLY = 0.3
EPOCHS = 10
LR = 0.001
NUM_CLASSES = 1000  # set to 1000 from style transfer generation
CKPT_DIR = "../models/facenet"
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        DataLoader
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CasiaStylizedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = []
        self.images = []
        # load images and labels
        # file structure: root_dir/<identity>/<sample_id>.jpg
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(img_path)
                        self.labels.append(label)

        uniq_labels = sorted(set(self.labels))
        self.label_to_idx = {lab: idx for idx, lab in enumerate(uniq_labels)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')

        if self.transform:
            sample = self.transform(img)

        label_str = self.labels[idx]
        label = self.label_to_idx[label_str]

        return sample, torch.tensor(label, dtype=torch.long)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        Run parameters
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}, for {EPOCHS} epochs")

# ----------------------------- data ------------------------------
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = CasiaStylizedDataset(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_idx = img_inds[:int(0.8 * len(img_inds))]
val_idx = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx), pin_memory=True)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_idx), pin_memory=True)

# ---------------------------- model ----------------------------- 
model = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=NUM_CLASSES
).to(device)

# # freeze all backbone weights
# for param in model.parameters():
#     param.requires_grad = False
# # unfreeze only the final fc + BN
# for param in model.last_linear.parameters():
#     param.requires_grad = True
# for param in model.last_bn.parameters():
#     param.requires_grad = True

# possible epoch split for multiple stages of finetuning
head_only_epochs = int(EPOCHS * HEAD_ONLY)
print(f"Head-only portion cutoff at epoch {head_only_epochs}/{EPOCHS}")

optimizer = torch.optim.AdamW(
    list(model.last_linear.parameters()) +
    list(model.last_bn.parameters()),
    lr=3e-4,
    weight_decay=1e-2
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

loss = torch.nn.CrossEntropyLoss()
metrics = {'accuracy': training.accuracy}
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        Training loop
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This part uses facenet_pytorch.training which can be found at https://github.com/timesler/facenet-pytorch/blob/master/models/utils/training.py

writer = SummaryWriter(log_dir="runs/facenet_finetune")
writer.iteration, writer.interval = 0, 10

print("Validating before any training")
model.eval()
training.pass_epoch(model, loss, val_loader, batch_metrics=metrics, device=device, writer=writer)

print(f"Starting training for {EPOCHS} epochs")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    model.train()
    training.pass_epoch(model, loss, train_loader, optimizer=optimizer, scheduler=scheduler, device=device, writer=writer, batch_metrics=metrics)

    model.eval()
    training.pass_epoch(model, loss, val_loader, batch_metrics=metrics, device=device, writer=writer)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        },
        os.path.join(CKPT_DIR, f"facenet_epoch_{epoch:02d}.pth"),
    )

writer.close()