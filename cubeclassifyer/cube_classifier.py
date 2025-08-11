import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from PIL import Image
import json
import random
from collections import defaultdict

# Custom dataset class for cube detection with grayscale images
class CubeDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(240, 320)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        
        # Load annotations
        self.annotations = []
        annotations_path = os.path.join(root_dir, "annotations.json")
        if os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                self.annotations = json.load(f)
        else:
            # If no annotations file, assume all images are in good/defective folders
            self.annotations = self._create_annotations_from_folders()
    
    def _create_annotations_from_folders(self):
        annotations = []
        for class_name in ['good', 'defective']:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        annotations.append({
                            'image_path': os.path.join(class_name, filename),
                            'label': 0 if class_name == 'good' else 1  # 0: good, 1: defective
                        })
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.root_dir, ann['image_path'])
        # Load as grayscale
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(ann['label'], dtype=torch.long)
        return image, label

# Lightweight CNN for grayscale images
class LightweightCubeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(LightweightCubeClassifier, self).__init__()
        
        # Custom lightweight CNN for grayscale images
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to fixed size
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Data augmentation transforms for grayscale images
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((240, 320)),  # Resize to 240x320
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            # Normalize grayscale images
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((240, 320)),  # Resize to 240x320
            transforms.ToTensor(),
            # Normalize grayscale images
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

# Training function
def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
        # Save best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_cube_classifier.pth')
    
    print(f'Best val Acc: {best_acc:.4f}')
    return model

# Function to convert model for Raspberry Pi deployment
def convert_model_for_rpi(model_path):
    model = LightweightCubeClassifier()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Convert to TorchScript for easier deployment
    example_input = torch.rand(1, 1, 240, 320)  # Grayscale input
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("cube_classifier_rpi.pt")
    
    print("Model converted for Raspberry Pi deployment!")

