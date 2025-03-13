import os
import csv
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

class myDataset(Dataset):
    def __init__(self, root_dir, transform=None, labeled=True):
        self.root_dir = root_dir
        self.transform = transform
        self.labeled = labeled
        self.label_map = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 
                          'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}

        if labeled:
            self.image_paths = []
            self.labels = []

            for class_name in os.listdir(root_dir):
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    for image_name in os.listdir(class_dir):
                        self.image_paths.append(os.path.join(class_dir, image_name))
                        self.labels.append(self.label_map[class_name])
        else:
            self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image and convert to grayscale
        image = Image.open(self.image_paths[idx]).convert("L")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        if self.labeled:
            label = self.labels[idx]
            return image, label
        file_name = os.path.basename(self.image_paths[idx])
        return image, os.path.splitext(file_name)[0]

class AugmentedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = []
        self.transform = transform

        # Collect all image paths and labels from the root directory
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.image_paths.append((os.path.join(class_dir, img_name), class_name))

        self.label_map = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 
                          'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}

    def __len__(self):
        # There will be 6 augmented images for each original image
        return len(self.image_paths) * 6

    def __getitem__(self, idx):
        # Get the index of the original image (every 6th image)
        img_index = idx // 6
        aug_type = idx % 6  # Determine the augmentation type

        img_path, label_name = self.image_paths[img_index]
        label = self.label_map[label_name]

        # Load image
        image = Image.open(img_path).convert("L")

        # Define augmentations
        if aug_type == 0:  # Original image
            aug = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5], std=[0.5])])
        elif aug_type == 1:  # Rotate left 15 degrees
            aug = transforms.Compose([transforms.RandomRotation(degrees=(-15, -15)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5], std=[0.5])])
        elif aug_type == 2:  # Rotate right 15 degrees
            aug = transforms.Compose([transforms.RandomRotation(degrees=(15, 15)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5], std=[0.5])])
        elif aug_type == 3:  # Horizontal Flip
            aug = transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5], std=[0.5])])
        elif aug_type == 4:  # Horizontal Flip + Rotate left 15 degrees
            aug = transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                      transforms.RandomRotation(degrees=(-15, -15)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5], std=[0.5])])
        elif aug_type == 5:  # Horizontal Flip + Rotate right 15 degrees
            aug = transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                      transforms.RandomRotation(degrees=(15, 15)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5], std=[0.5])])

        # Apply the chosen augmentation
        augmented_image = aug(image)

        return augmented_image, label

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),           # Randomly flip the image horizontally
    transforms.ToTensor(),                      # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
])

train_dataset = AugmentedDataset(root_dir='data/Images/train', transform=None)
test_dataset = myDataset(root_dir='data/Images/test', transform=test_transforms, labeled=False)
batch = 32

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN10(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN10, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Conv Layer 1
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Conv Layer 2
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.5),
            nn.MaxPool2d(kernel_size=2),  # Pooling Layer
            nn.Dropout(0.25),  # Dropout 1
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Conv Layer 3
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Conv Layer 4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.5),
            nn.MaxPool2d(kernel_size=2),  # Pooling Layer
            nn.Dropout(0.25),  # Dropout 2
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv Layer 5
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Conv Layer 6
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.5),
            nn.MaxPool2d(kernel_size=2),  # Pooling Layer
            nn.Dropout(0.25),  # Dropout 3
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Conv Layer 7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv Layer 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.5),
            nn.MaxPool2d(kernel_size=2),  # Pooling Layer
            nn.Dropout(0.25),  # Dropout 4
            
            # Block 5
            nn.Conv2d(256, 32, kernel_size=3, padding=1),  # Conv Layer 9
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # Conv Layer 10
            nn.LeakyReLU(negative_slope=0.5),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 1 * 1, 100)  # Assuming input size is 48x48
        self.fc2 = nn.Linear(100, num_classes)  # Final classification layer

    def forward(self, x):
        x = self.features(x)  # Feature extraction
        x = self.flatten(x)   # Flatten the output
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = F.dropout(x, 0.2, training=self.training)  # Dropout before output layer
        x = self.fc2(x)  # Final output layer
        return F.log_softmax(x, dim=1)  # Log probabilities for CrossEntropyLoss

# Instantiate the model
model = CNN10(num_classes=7)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 25

for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for label, pred in zip(labels, preds):
            class_total[label.item()] += 1
            if label == pred:
                class_correct[label.item()] += 1

    train_acc = correct / total
    print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

    print("Per-Class Accuracy:")
    for class_name, class_idx in train_dataset.label_map.items():
        accuracy = (
            class_correct[class_idx] / class_total[class_idx] if class_total[class_idx] > 0 else 0
        )
        print(f"  {class_name}: {accuracy:.4f}")

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("model/cnn", exist_ok=True)
torch.save(model.state_dict(), f'model/cnn/{current_time}.pth')

model.eval()
predictions = []
with torch.no_grad():
    for images, filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(zip(filenames, preds.cpu().numpy()))

output_file = '111550151.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label'])
    writer.writerows(predictions)

valid_dataset = myDataset(root_dir='data/Images/validation', transform=test_transforms, labeled=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch, shuffle=False)

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    valid_acc = correct / total
    print(f"Validation Acc: {valid_acc:.4f}")