import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

class MoviePosterDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'image_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        labels = torch.tensor(self.dataframe.loc[idx, 'labels'], dtype=torch.float32)
        return image, labels

def run_resnet(datasets, num_classes, num_epochs=10):
    df_train = pd.concat(datasets[:4]).reset_index(drop=True)
    df_test = datasets[4].reset_index(drop=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(MoviePosterDataset(df_train, transform), batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(MoviePosterDataset(df_test, transform), batch_size=64, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Starting ResNet Training...")
    for epoch in range(num_epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print("Evaluating ResNet...")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing ResNet"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    metrics = {
        "Micro F1": f1_score(all_targets, all_preds, average='micro', zero_division=0),
        "Exact Match": accuracy_score(all_targets, all_preds)
    }
    return metrics