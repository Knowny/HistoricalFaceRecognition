#!/usr/bin/env python3
"""
No good results just a proof of earlier attempts at finetuning

file: experiment1.py
project: KNN Face Recognition
author: Tereza Magerkova, xmager00
"""
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for face recognition fine-tuning using classification head")
    parser.add_argument('--data_dir', type=str, default="../datasets/stylized_images_112_fin", help="Path to training dataset")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and validation")
    parser.add_argument('--epochs', type=int, default=15, help="Total number of training epochs")
    parser.add_argument('--lr', type=float, default=2e-4, help="Initial learning rate for optimizer")
    parser.add_argument('--num_classes', type=int, default=999, help="Number of identity classes classification head")
    return parser.parse_args()

CKPT_DIR = "../models/facenet"
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    print(f"Running on {device}, classify {model.classify}")

    # freezing model parts
    modules_to_freeze = ['conv2d_1a', 'conv2d_2a', 'conv2d_2b', 'conv2d_3b', 'conv2d_4a', 'conv2d_4b',
                     'repeat_1', 'mixed_6a', 'repeat_2', 'mixed_7a']
    
    for name, param in model.named_parameters():
        if any(module_name in name for module_name in modules_to_freeze):
            param.requires_grad = False
            # print(f"Freezing param: {name}")
        else:
            param.requires_grad = True
            # print(f"Training param: {name}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Portion of network training: {trainable_params / total_params:.4f}")
    # classifier, scheduler, loss
    model.classifier = torch.nn.Linear(512, args.num_classes).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # data
    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data = datasets.ImageFolder(args.data_dir, transform=None)
    labels = np.array(data.targets)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, val_indices = next(splitter.split(np.zeros(len(labels)), labels))

    train_dataset = Subset(data, train_indices)
    val_dataset = Subset(data, val_indices)

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    

    best_val_loss = float('inf')
    val_loss = 0.0
    for epoch in range(args.epochs):
        # train
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"[TRAIN]\t{epoch+1}/{args.epochs}", unit="batch")
        for inputs, targets in train_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            embeddings = model(inputs) 
            outputs = model.classifier(embeddings)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            print(f"[VAL] batches: {len(val_loader)}")
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                embeddings = model(inputs)
                outputs = model.classifier(embeddings)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"train loss: {train_loss} | val_loss: {val_loss}")
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_e1.pth')
    # save last model
    torch.save(model.state_dict(), 'last_e1.pth')

if __name__ == "__main__":
    main()