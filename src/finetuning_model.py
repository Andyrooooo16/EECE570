import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt  # ← Added for plotting
from model import RetinalDiseaseClassifier

DEBUG = True  # Enable debug for step-by-step feedback

# Dataset class remains the same
class RetinalFundusDataset(Dataset):
    def __init__(self, img_folder, csv_path, transform=None):
        print("🔄 Loading dataset...")
        self.img_folder = img_folder
        self.labels_df = pd.read_csv(csv_path)
        self.transform = transform
        self.disease_cols = [col for col in self.labels_df.columns if col not in ['ID', 'Disease_Risk']]
        print(f"✅ Dataset loaded. Number of samples: {len(self.labels_df)}")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        image_path = os.path.join(self.img_folder, f"{row['ID']}.png")
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row[self.disease_cols].values.astype(float), dtype=torch.float32)
        return image, label

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Plot training curves
def plot_training_curves(history, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='o')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # ROC AUC plot
    plt.figure(figsize=(8, 5))
    plt.plot(history['val_roc_auc'], label='Val ROC AUC', marker='o', color='green')
    plt.title('Validation ROC AUC Curve')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_auc_curve.png'))
    plt.close()

    print("📊 Training curves saved to 'plots/' folder.")

# Fine-tuning loop
def fine_tune_model(model, train_loader, val_loader, num_epochs=5, lr=1e-5, freeze_backbone=True):
    print("🔄 Starting fine-tuning...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if freeze_backbone:
        print("🔒 Freezing the backbone...")
        for name, param in model.model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False
        print("✅ Backbone frozen.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_roc_auc': []}

    for epoch in range(num_epochs):
        print(f"🔄 Starting Epoch {epoch+1}/{num_epochs}...")
        model.train()
        train_loss = 0.0

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        all_labels, all_outputs = [], []
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
        except:
            roc_auc = 0.0

        print(f"🔁 Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ROC AUC: {roc_auc:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'fine_tuned_model.pth')
            print("✅ Fine-tuned model saved!")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_roc_auc'].append(roc_auc)

    plot_training_curves(history)
    return model, history

# Run fine-tuning
if __name__ == "__main__":
    print("🔄 Initializing dataset and data loaders...")
    image_folder = "../data/Training_Set/training_cropped"
    csv_path = "../data/Training_Set/RFMiD_Training_Labels.csv"

    dataset = RetinalFundusDataset(image_folder, csv_path, transform=train_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)

    print("✅ Data loaders initialized.")

    # Load the pre-trained model
    model = RetinalDiseaseClassifier(num_classes=45, pretrained=True)
    model.load_state_dict(torch.load("../outputs/Traning/best_fundus_model.pth"))
    print("✅ Model loaded.")

    try:
        fine_tuned_model, fine_tune_history = fine_tune_model(model, train_loader, val_loader)
        print("✅ Fine-tuning complete.")
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
