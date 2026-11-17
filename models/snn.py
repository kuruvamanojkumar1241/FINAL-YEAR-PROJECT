import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SpikeActivation(nn.Module):
    def forward(self, x):
        return torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))

class SNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = SpikeActivation()(x)
        x = self.pool(self.conv2(x))
        x = SpikeActivation()(x)
        x = x.view(x.size(0), -1)
        x = SpikeActivation()(self.fc1(x))
        return self.fc2(x)

def load_data(data_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train_snn(data_dir, save_path, epochs=5): # Reduced default epochs for speed
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found.")
        return None
        
    loader = load_data(data_dir)
    model = SNNModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        acc = correct / total * 100 if total > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f} Accuracy: {acc:.2f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return model
