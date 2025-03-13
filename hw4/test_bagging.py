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

class EqualizeTransform:
    def __call__(self, image):
        return transforms.functional.equalize(image)

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

test_dataset = myDataset(root_dir='data/Images/test', transform=test_transforms, labeled=False)
batch = 32

#train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

#model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

num_models = 25
Models = []
model1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
fc_features = model1.fc.in_features
model1.fc = nn.Sequential(nn.Linear(fc_features, 7),
                        nn.Softmax(dim=1))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1.load_state_dict(torch.load(f'model/new_lr_15.pth', weights_only=True))

Models.append(model1)

Models2 = []
model2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
fc_features = model2.fc.in_features
model2.fc = nn.Sequential(nn.Linear(fc_features, 7),
                        nn.Softmax(dim=1))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model2.load_state_dict(torch.load(f'model/res_newaug_15_1.pth', weights_only=True))

Models2.append(model2)

model3 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
fc_features = model3.fc.in_features
model3.fc = nn.Sequential(nn.Linear(fc_features, 7),
                        nn.Softmax(dim=1))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model3.load_state_dict(torch.load(f'model/res_newaug_15_2.pth', weights_only=True))

Models2.append(model3)

model4 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
fc_features = model4.fc.in_features
model4.fc = nn.Sequential(nn.Linear(fc_features, 7),
                        nn.Softmax(dim=1))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model4.load_state_dict(torch.load(f'model/res_newaug_15_3.pth', weights_only=True))

Models2.append(model4)

for i in range(num_models):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    fc_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(fc_features, 7),
                            nn.Softmax(dim=1))
    model.load_state_dict(torch.load(f'model/bag/bag_{i+1}.pth', weights_only=True))

    Models.append(model)

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
    #final_prediction = np.argmax(averaged_probs)
    return averaged_probs

all_predictions = []

for model in Models:
    model = model.to(device)
    model.eval()
    prediction = []
    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.softmax(outputs, dim=1)
            prediction.append(preds.cpu().numpy())
        all_predictions.append(np.concatenate(prediction, axis=0))

for model in Models2:
    model = model.to(device)
    model.eval()
    predictions = []
    for image_path in test_dataset.image_paths:
        # Open image
        image = Image.open(image_path).convert("L")
        image = image.convert("RGB")
        
        # Predict using TTA
        final_pred = tta_predict_with_probabilities(model, image, tta_transforms, device)
        
        # Store the filename and prediction
        predictions.append(final_pred)
    all_predictions.append(np.concatenate(predictions, axis=0))

# Majority voting
all_predictions = np.array(all_predictions)
average_probabilities = np.mean(all_predictions, axis=0)
final_predictions = np.argmax(average_probabilities, axis=-1)

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
