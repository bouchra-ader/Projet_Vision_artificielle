import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class Predictor:
    def __init__(self, model_path, img_size=(150, 150)):
        self.model_path = model_path
        self.img_size = img_size
        self.model = self.load_model()

    def load_model(self):
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def predict(self, img_path):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = Image.open(img_path)
        img = transform(img).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        with torch.no_grad():
            output = self.model(img)
            prediction = torch.sigmoid(output).item()

        return "Chien" if prediction > 0.5 else "Chat"
