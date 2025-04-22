import torch.nn as nn
from torchvision import models

class RetinalDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=45, pretrained=True):
        super(RetinalDiseaseClassifier, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)  # Output logits for 45 disease classes
        )

    def forward(self, x):
        return self.model(x)

