import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import datetime
import numpy as np

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
        image = Image.open(self.image_paths[idx]).convert("L")
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.labeled:
            label = self.labels[idx]
            return image, label
        file_name = os.path.basename(self.image_paths[idx])
        return image, os.path.splitext(file_name)[0]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = myDataset(root_dir='data/Images/train', transform=train_transforms, labeled=True)
test_dataset = myDataset(root_dir='data/Images/test', transform=test_transforms, labeled=False)
batch = 32

#train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

n_samples = len(train_dataset)
num_models = 10
Models = []

for i in range(num_models):
    print(f"Training model {i+1}")
    
    bootstrap_indices = torch.randint(0, n_samples, (n_samples,))
    bootstrap_dataset = Subset(train_dataset, bootstrap_indices)
    bootstrap_loader = DataLoader(bootstrap_dataset, batch_size=batch, shuffle=True)
    
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    fc_features = model.fc.in_features
    #for param in model.parameters():
    #    param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(fc_features, 7),
                             nn.LogSoftmax(dim=1))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    params = [
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("fc")], "lr": 1e-4},
    ]
    #optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = optim.Adam(params)
    #scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.9)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1)
    epochs = 15

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(bootstrap_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
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

        train_acc = correct / total
        #print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

        #scheduler.step()
        #print(f"Learning rate for epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}")

    #current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), f'model/bag/bag_{i+31}.pth')

    Models.append(model)

all_predictions = []

for model in Models:
    model.eval()
    prediction = []
    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            prediction.append(preds.cpu().numpy())
    all_predictions.append(np.concatenate(prediction))

# Majority voting
all_predictions = np.array(all_predictions)
final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_predictions)

predictions = []
c = 0
with torch.no_grad():
    for images, filenames in test_loader:
        predictions.extend(zip(filenames, final_predictions[c:min(c+batch, len(final_predictions))]))
        c += batch

output_file = '111550151.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label'])
    writer.writerows(predictions)

valid_dataset = myDataset(root_dir='data/Images/validation', transform=test_transforms, labeled=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch, shuffle=False)

all_predictions = []

for model in Models:
    model.eval()
    prediction = []
    with torch.no_grad():
        for images, filenames in valid_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            prediction.append(preds.cpu().numpy())
    all_predictions.append(np.concatenate(prediction))

# Majority voting
all_predictions = np.array(all_predictions)
final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_predictions)

correct, total, c = 0, 0, 0
with torch.no_grad():
    for images, labels in valid_loader:
        preds = final_predictions[c:min(c+batch, len(final_predictions))]
        c += batch
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    valid_acc = correct / total
    print(f"Validation Acc: {valid_acc:.4f}")