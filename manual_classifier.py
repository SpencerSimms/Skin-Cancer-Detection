import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter

# Paths
DATA_PATH = "HAM10000"
IMAGE_PATH = os.path.join(DATA_PATH, "images")
METADATA_PATH = os.path.join(DATA_PATH, "HAM10000_metadata.csv")

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load metadata
metadata = pd.read_csv(METADATA_PATH)

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomVerticalFlip(),  # Added vertical flip
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Added affine transformations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Custom Dataset Class
class SkinCancerDataset(Dataset):
    def __init__(self, image_dir, metadata, transform=None):
        self.image_dir = image_dir
        self.metadata = metadata
        self.transform = transform
        self.image_ids = metadata["image_id"].values
        self.labels = metadata["dx"].values
        self.label_map = {label: idx for idx, label in enumerate(np.unique(self.labels))}
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        label_idx = self.label_map[label]
        image_path = os.path.join(self.image_dir, image_id + ".jpg")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label_idx, image_id

# Load dataset
dataset = SkinCancerDataset(IMAGE_PATH, metadata, transform)

# Use a smaller subset (1000 images) for faster training
subset_size = 10000
dataset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

# Train/Test Split (80% Train, 20% Test)
train_size = int(0.8 * subset_size)
test_size = subset_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Handle Class Imbalance with Weighted Sampling
labels_list = [label for _, label, _ in train_dataset]
class_counts = Counter(labels_list)
class_weights = {k: 1.0 / v for k, v in class_counts.items()}
sample_weights = [class_weights[label] for _, label, _ in train_dataset]
sampler = WeightedRandomSampler(sample_weights, len(train_dataset))

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Pretrained ResNet Model
class SkinCancerResNet(nn.Module):
    def __init__(self, num_classes):
        super(SkinCancerResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # Use ResNet18 as a base
        # Modify the final layer to match the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Initialize Model
num_classes = len(metadata["dx"].unique())
model = SkinCancerResNet(num_classes).to(device)

# Loss Function (Weighted)
class_weights_tensor = torch.tensor(list(class_weights.values())).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate

# Training Function
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

# Testing Function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    actuals = []
    image_ids = []
    
    with torch.no_grad():
        for images, labels, img_ids in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Collect predictions
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
            image_ids.extend(img_ids)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return accuracy, predictions, actuals, image_ids

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=5)

# Evaluate the model
accuracy, predictions, actuals, image_ids = test_model(model, test_loader)
print(f"Test Accuracy: {accuracy:.4f}")

# Convert predictions to labels
label_map = {v: k for k, v in dataset.dataset.label_map.items()}
predicted_labels = [label_map[p] for p in predictions]
actual_labels = [label_map[a] for a in actuals]

# Save Predictions to CSV
output_df = pd.DataFrame({
    "image_id": image_ids,
    "actual_label": actual_labels,
    "predicted_label": predicted_labels
})
output_df.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")
