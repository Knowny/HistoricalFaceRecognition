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

DATA_DIR = "../data/casia_stylized"
BATCH_SIZE = 32
EPOCHS = 10 # set to 10 for testing
LR = 0.001
NUM_CLASSES = 1000  # set to 1000 from style transfer generation
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample_path = self.images[idx]
        label = self.labels[idx]

        img = Image.open(sample_path).convert('RGB')

        if self.transform:
            sample = self.transform(img)

        return sample, label
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        Run parameters
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}, for {EPOCHS} epochs")

# ----------------------------- data ------------------------------
transform = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = CasiaStylizedDataset(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_idx = img_inds[:int(0.8 * len(img_inds))]
val_idx = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))

# ---------------------------- model ----------------------------- 
model = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=NUM_CLASSES
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = MultiStepLR(optimizer, [5, 10])
loss = torch.nn.CrossEntropyLoss()
# metrics followed during training
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        Training loop
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This part uses facenet_pytorch.training which can be found at https://github.com/timesler/facenet-pytorch/blob/master/models/utils/training.py

writer = SummaryWriter(log_dir='logs')
writer.iteration, writer.interval = 0, 10

print("Validating before any training")
model.eval()
training.pass_epoch(model, loss, val_loader, batch_metrics=metrics, device=device, writer=writer)
print("Starting training")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    
    model.train()
    training.pass_epoch(model, loss, train_loader, optimizer=optimizer, scheduler=scheduler, device=device, writer=writer, batch_metrics=metrics)

    model.eval()
    training.pass_epoch(model, loss, val_loader, batch_metrics=metrics, device=device, writer=writer)

writer.close()