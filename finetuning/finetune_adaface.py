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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        Run parameters
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
DATA_DIR = (
    "../datasets_examples/casia_stylized"  # root folder with subfolders per identity
)
PRETRAINED_CKPT = (
    "../models/adaface_ir50_ms1mv2.ckpt"  # path to pretrained AdaFace checkpoint
)
OUTPUT_DIR = "../models/adaface_checkpoints"  # where to save fine-tuned checkpoints
BATCH_SIZE = 32  # number of samples per batch
EPOCHS = 10  # total number of training epochs
LR = 0.01  # initial learning rate

# dynamically compute milestones at 50% and 75% of total epochs
milestone1 = EPOCHS // 2
milestone2 = int(0.75 * EPOCHS)
MILESTONES = [milestone1, milestone2]  # epochs at which to decay LR

NUM_WORKERS = 4  # number of DataLoader workers
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        RGB→BGR Conversion
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def rgb_to_bgr(x: torch.Tensor) -> torch.Tensor:
    """
    Swap image channels from RGB to BGR.
    Required because AdaFace was pretrained on BGR inputs.
    Must be a top-level function for Windows DataLoader pickling.
    """
    return x[[2, 1, 0], ...]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        Main training script
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}, fine-tuning AdaFace for {EPOCHS} epochs")

    # ----------------------------- data transforms -----------------------------
    # resize → to tensor → RGB→BGR → normalize to [-1,1]
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),  # resize to 112×112 pixels
            transforms.ToTensor(),  # convert PIL Image to [0,1] tensor
            transforms.Lambda(rgb_to_bgr),  # swap channels to BGR
            transforms.Normalize(  # normalize to ~[-1,1]
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    # ------------------------- dataset and dataloaders -------------------------
    dataset = ImageFolder(DATA_DIR, transform=transform)
    num_classes = len(dataset.classes)

    # create train/validation split (80/20)
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

    # ------------------------ model initialization ----------------------------
    # build backbone and AdaFace head
    backbone = build_model("ir_50").to(device)
    head = build_head(
        head_type="adaface",
        embedding_size=512,
        class_num=num_classes,
        m=0.4,  # angular margin
        t_alpha=1.0,  # temperature for adaptive margin
        h=0.333,  # adaptive margin scale
        s=64.0,  # feature scale
    ).to(device)

    # load pretrained backbone weights (ignore head mismatch)
    ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    backbone.load_state_dict(state_dict, strict=False)

    # ---------------------- loss, optimizer, scheduler ------------------------
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(head.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = MultiStepLR(
        optimizer, milestones=MILESTONES, gamma=0.1  # decay LR at specified epochs
    )

    # ---------------------------- TensorBoard logger --------------------------
    writer = SummaryWriter(log_dir="runs/adaface_finetune")

    # ---------------------------- training loop -------------------------------
    for epoch in range(1, EPOCHS + 1):
        # ----- training phase -----
        backbone.train()
        head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # forward pass through backbone and head
            embeddings, norms = backbone(imgs)
            logits = head(embeddings, norms, labels)
            loss = criterion(logits, labels)

            # backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate training metrics
            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # step the scheduler after each epoch
        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total * 100

        # ----- validation phase -----
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

                # accumulate validation metrics
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total * 100

        # ----- log results and save checkpoint -----
        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%  |  "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # save model checkpoint for this epoch
        torch.save(
            {
                "epoch": epoch,
                "backbone_state_dict": backbone.state_dict(),
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(OUTPUT_DIR, f"adaface_epoch_{epoch:02d}.pt"),
        )

    # close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
