#!/usr/bin/env python3
"""
Finetuning of the AdaFace ir50_ms1mv2 model

file: finetune_adaface.py
project: KNN Face Recognition
version: 1.0
author: xjanos19
"""

import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from net import build_model
from head import build_head
import numpy as np

DATA_DIR = "../datasets_examples/casia_stylized"
PRETRAINED_CKPT = "../models/adaface_ir50_ms1mv2.ckpt"
OUTPUT_DIR = "../models/adaface_checkpoints"
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.01
milestone1 = EPOCHS // 2
milestone2 = int(0.75 * EPOCHS)
MILESTONES = [milestone1, milestone2]
NUM_WORKERS = 4
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def rgb_to_bgr(x: torch.Tensor) -> torch.Tensor:
    return x[[2, 1, 0], ...]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}, fine-tuning AdaFace for {EPOCHS} epochs")

    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Lambda(rgb_to_bgr),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    dataset = ImageFolder(DATA_DIR, transform=transform)
    num_classes = len(dataset.classes)
    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    backbone = build_model("ir_50").to(device)
    head = build_head(
        head_type="adaface",
        embedding_size=512,
        class_num=num_classes,
        m=0.4,
        t_alpha=1.0,
        h=0.333,
        s=64.0,
    ).to(device)

    ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    backbone.load_state_dict(sd, strict=False)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(head.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)

    writer = SummaryWriter(log_dir="runs/adaface_finetune")

    for epoch in range(1, EPOCHS + 1):
        # ——— TRAIN ———
        backbone.train()
        head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            embeddings, norms = backbone(imgs)
            logits = head(embeddings, norms, labels)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total * 100

        # ——— VALID ———
        backbone.eval()
        head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                embeddings, norms = backbone(imgs)
                logits = head(embeddings, norms, labels)
                loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total * 100

        # ——— LOG & SAVE ———
        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%  |  "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # checkpoint save
        torch.save(
            {
                "epoch": epoch,
                "backbone_state_dict": backbone.state_dict(),
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(OUTPUT_DIR, f"adaface_epoch_{epoch:02d}.pt"),
        )

    writer.close()


if __name__ == "__main__":
    main()
