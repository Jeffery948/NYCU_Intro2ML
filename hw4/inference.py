import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

class myDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 
                          'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}

        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        file_name = os.path.basename(self.image_paths[idx])
        return image, os.path.splitext(file_name)[0]

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = myDataset(root_dir='data/Images/test', transform=test_transforms)
batch = 32
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tta_transforms = [
    transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomRotation(degrees=(-15, -15)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomRotation(degrees=(15, 15)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(p=1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(p=1),
                        transforms.RandomRotation(degrees=(-15, -15)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(p=1),
                        transforms.RandomRotation(degrees=(15, 15)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
]

num_models = 4
Models = []

for i in range(num_models):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    fc_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(fc_features, 7),
                            nn.Softmax(dim=1))
    model.load_state_dict(torch.load(f'model/model{i+1}.pth', weights_only=True))

    Models.append(model)

def tta_predict_with_probabilities(model, image, tta_transforms, device):
    model.eval()
    augmented_probs = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            augmented_image = transform(image).unsqueeze(0).to(device)
            outputs = model(augmented_image)
            probs = outputs.cpu().numpy()
            augmented_probs.append(probs)
    
    # Average probabilities across all augmentations
    averaged_probs = np.mean(augmented_probs, axis=0)

    return averaged_probs

all_predictions = []

for model in Models:
    model = model.to(device)
    model.eval()
    predictions = []
    for image_path in test_dataset.image_paths:
        # Open image
        image = Image.open(image_path).convert("L")
        image = image.convert("RGB")
        
        # Predict using TTA
        final_pred = tta_predict_with_probabilities(model, image, tta_transforms, device)
        
        predictions.append(final_pred)
    all_predictions.append(np.concatenate(predictions, axis=0))

all_predictions = np.array(all_predictions)
average_probabilities = np.mean(all_predictions, axis=0)
final_predictions = np.argmax(average_probabilities, axis=-1)

predictions = []
c = 0

for images, filenames in test_loader:
    predictions.extend(zip(filenames, final_predictions[c:min(c+batch, len(final_predictions))]))
    c += batch

output_file = '111550151.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label'])
    writer.writerows(predictions)
