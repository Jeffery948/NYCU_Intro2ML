import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

class AugmentedDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []

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
        image = image.convert("RGB")

        # Define augmentations
        if aug_type == 0:  # Original image
            aug = transforms.Compose([transforms.RandomResizedCrop(224), 
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif aug_type == 1:  # Rotate left 15 degrees
            aug = transforms.Compose([transforms.RandomResizedCrop(224), 
                                      transforms.RandomRotation(degrees=(-15, -15)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif aug_type == 2:  # Rotate right 15 degrees
            aug = transforms.Compose([transforms.RandomResizedCrop(224), 
                                      transforms.RandomRotation(degrees=(15, 15)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif aug_type == 3:  # Horizontal Flip
            aug = transforms.Compose([transforms.RandomResizedCrop(224), 
                                      transforms.RandomHorizontalFlip(p=1),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif aug_type == 4:  # Horizontal Flip + Rotate left 15 degrees
            aug = transforms.Compose([transforms.RandomResizedCrop(224), 
                                      transforms.RandomHorizontalFlip(p=1),
                                      transforms.RandomRotation(degrees=(-15, -15)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif aug_type == 5:  # Horizontal Flip + Rotate right 15 degrees
            aug = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(p=1),
                                      transforms.RandomRotation(degrees=(15, 15)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # Apply the chosen augmentation
        augmented_image = aug(image)

        return augmented_image, label

train_dataset = AugmentedDataset(root_dir='data/Images/train')
batch = 32

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

fc_features = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(fc_features, 7),
                         nn.LogSoftmax(dim=1))
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
params = [
    {"params": model.fc.parameters(), "lr": 1e-3},
    {"params": [p for name, p in model.named_parameters() if not name.startswith("fc")], "lr": 1e-4},
]

optimizer = optim.Adam(params)
epochs = 15

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
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), f'model/{current_time}.pth')
