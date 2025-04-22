import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# ----- Dataset Class -----
class RetinalFundusDataset(Dataset):
    def __init__(self, img_folder, csv_path, transform=None):
        self.img_folder = img_folder
        self.labels_df = pd.read_csv(csv_path)
        self.transform = transform
        self.disease_cols = [col for col in self.labels_df.columns if col not in ['ID', 'Disease_Risk']]
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        image_id = row['ID']
        image_path = os.path.join(self.img_folder, f"{image_id}.png")
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Create multi-hot encoded label vector for all 45 diseases
        label = torch.tensor(row[self.disease_cols].values.astype(float), dtype=torch.float32)
        
        return image, label

# ----- Image Transforms -----
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----- Load Training Dataset -----
train_dataset = RetinalFundusDataset(
    img_folder="../data/Training_Set/training_cropped",
    csv_path="../data/Training_Set/RFMiD_Training_Labels.csv",
    transform=train_transform
)

disease_names = train_dataset.disease_cols
print("✅ Training dataset loaded and transformations applied successfully.")
print(f"Total disease classes: {len(disease_names)}")

# ----- Load Evaluation Dataset -----
eval_dataset = RetinalFundusDataset(
    img_folder="../data/Evaluation_Set/evaluation_cropped",
    csv_path="../data/Evaluation_Set/RFMiD_Validation_Labels.csv",
    transform=eval_transform
)

eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
print("✅ Evaluation dataset loaded and ready.")

# ----- Load Test Dataset -----
test_dataset = RetinalFundusDataset(
    img_folder="../data/Test_Set/test_cropped",  
    csv_path="../data/Test_Set/RFMiD_Testing_Labels.csv",  
    transform=eval_transform 
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("✅ Test dataset loaded and ready.")