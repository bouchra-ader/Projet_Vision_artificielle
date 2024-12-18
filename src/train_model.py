import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class ModelTrainer:
    def __init__(self, img_size=(3, 150, 150)):
        self.img_size = img_size
        self.model = None

    def build_model(self):
        # Utiliser ResNet pré-entraîné
        self.model = models.resnet18(pretrained=True)
        # Modifier la dernière couche pour la classification binaire
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.model = self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def train(self, train_loader, validation_loader, epochs=10):
        if self.model is None:
            raise ValueError("Le modèle doit être construit avant l'entraînement.")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                optimizer.zero_grad()

                # Propagation avant
                outputs = self.model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()

                optimizer.step()
                running_loss += loss.item()

                predicted = torch.round(torch.sigmoid(outputs))
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")

    def save_model(self, file_path):
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder.")
        torch.save(self.model.state_dict(), file_path)
