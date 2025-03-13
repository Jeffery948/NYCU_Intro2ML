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
from sklearn.metrics import precision_score, recall_score, f1_score

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
    
tta_transforms = [
    transforms.Compose([#transforms.RandomResizedCrop(224), 
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([#transforms.RandomResizedCrop(224), 
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomRotation(degrees=(-15, -15)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([#transforms.RandomResizedCrop(224), 
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomRotation(degrees=(15, 15)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([#transforms.RandomResizedCrop(224), 
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(p=1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([#transforms.RandomResizedCrop(224), 
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(p=1),
                        transforms.RandomRotation(degrees=(-15, -15)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([#transforms.RandomResizedCrop(224), 
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(p=1),
                        transforms.RandomRotation(degrees=(15, 15)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
]

def tta_predict_with_probabilities(model, image, tta_transforms, device):
    model.eval()
    augmented_probs = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            augmented_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
            outputs = model(augmented_image)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            augmented_probs.append(probs)
    
    # Average probabilities across all augmentations
    averaged_probs = np.mean(augmented_probs, axis=0)
    final_prediction = np.argmax(averaged_probs)
    return final_prediction

test_dataset = myDataset(root_dir='data/Images/test', transform=None, labeled=False)
batch = 32
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

num_models = 3
Models = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in range(num_models):
    Models.append(torch.load(f'model/res_newaug_15_{i+1}.pth', weights_only=True))

average_weights = Models[0].copy()

for key in average_weights.keys():
    average_weights[key] = torch.stack([weight[key].float() for weight in Models]).mean(dim=0)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
fc_features = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(fc_features, 7),
                        nn.Softmax(dim=1))
model.load_state_dict(average_weights)

model.to(device)
model.eval()

predictions = []
for image_path in test_dataset.image_paths:
    # Open image
    image = Image.open(image_path).convert("L")
    image = image.convert("RGB")
    
    # Predict using TTA
    final_pred = tta_predict_with_probabilities(model, image, tta_transforms, device)
    
    # Store the filename and prediction
    file_name = os.path.basename(image_path)
    predictions.append((os.path.splitext(file_name)[0], final_pred))

output_file = '111550151.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label'])
    writer.writerows(predictions)