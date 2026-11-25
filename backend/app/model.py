import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["control", "alzheimer_pattern"]

class ResearchCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = models.resnet18(weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        return self.net(x)

class ModelService:
    def __init__(self):
        self.path = os.getenv("MODEL_PATH", "backend/models/research_model.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])
        self.load()

    def load(self):
        p = Path(self.path)
        if not p.exists():
            self.model = None
            return
        model = ResearchCNN()
        state = torch.load(p, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device).eval()
        self.model = model

    def predict(self, image: Image.Image):
        if self.model is None:
            return None
        x = self.transform(image.convert("L")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].cpu().tolist()
        return {CLASS_NAMES[i]: float(probs[i]) for i in range(2)}
