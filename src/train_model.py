# [Imports]
import os
import time
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# ----- Debug Toggle -----
DEBUG = True  # Set to False when training full-scale

# ----- Dataset -----
class RetinalFundusDataset(Dataset):
    def __init__(self, img_folder, csv_path, transform=None):
        print("🔄 Loading dataset...")
        self.img_folder = img_folder
        self.labels_df = pd.read_csv(csv_path)
        self.transform = transform
        self.disease_cols = [col for col in self.labels_df.columns if col not in ['ID', 'Disease_Risk']]
        print(f"🦠 Number of disease labels: {len(self.disease_cols)}")

        if DEBUG:
            print(f"🔍 Sample disease columns: {self.disease_cols[:5]}")
            print(f"📊 First 2 label rows:\n{self.labels_df[self.disease_cols].head(2)}")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        image_id = row['ID']
        image_path = os.path.join(self.img_folder, f"{image_id}.png")
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row[self.disease_cols].values.astype(float), dtype=torch.float32)

        if DEBUG and idx == 0:
            print(f"🖼️ Loaded image: {image_id} | Image shape: {image.shape} | Label shape: {label.shape}")
        return image, label

# ----- Transforms -----
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----- Model Import -----
from model import RetinalDiseaseClassifier

# ----- Training Loop -----
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_roc_auc': [], 'epoch_time': []}

    output_dir = "../outputs/Training"
    os.makedirs(output_dir, exist_ok=True)

    total_start_time = time.time()
    print("🚀 Starting training...")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            if DEBUG and batch_idx == 0:
                print(f"🔧 Epoch {epoch+1} | Output shape: {outputs.shape} | Label shape: {labels.shape}")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                all_labels.append(labels.cpu())
                all_outputs.append(outputs.cpu())

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        y_true = torch.cat(all_labels)
        y_pred = torch.cat(all_outputs).sigmoid()

        try:
            roc_auc = roc_auc_score(y_true, y_pred, average='macro')
        except ValueError:
            roc_auc = 0.0

        epoch_time = time.time() - epoch_start
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_roc_auc'].append(roc_auc)
        history['epoch_time'].append(epoch_time)

        print(f"📈 Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ROC AUC: {roc_auc:.4f} | Time: {epoch_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, '../outputs/Training/best_fundus_model.pth'))
            print("✅ Saved best model!")

    total_time = time.time() - total_start_time
    print(f"✅ Training complete. Total time: {total_time:.2f}s")

    # ----- Plot and Save Loss Curve -----
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()
    print(f"📉 Loss curve saved to: {os.path.join(output_dir, 'loss_curve.png')}")

    return model, history

# ----- Run Training -----
if __name__ == "__main__":
    image_folder = "../data/Training_Set/training_cropped"
    csv_path = "../data/Training_Set/RFMiD_Training_Labels.csv"

    dataset = RetinalFundusDataset(img_folder=image_folder, csv_path=csv_path, transform=train_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)

    model = RetinalDiseaseClassifier(num_classes=45, pretrained=True)

    try:
        trained_model, training_history = train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4)
    except Exception as e:
        print(f"❌ Training failed: {e}")
