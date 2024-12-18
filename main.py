from src.data_preprocessing import DataPreprocessor
from src.train_model import ModelTrainer
from src.predict import Predictor

# Prétraitement des données
preprocessor = DataPreprocessor("data/raw/train", "data/raw/validation")
train_loader, val_loader = preprocessor.get_loaders()

# Entraînement du modèle
trainer = ModelTrainer()
trainer.build_model()
trainer.train(train_loader, val_loader, epochs=5)
trainer.save_model("models/cat_dog_model.pth")

# Prédiction avec le modèle
predictor = Predictor("models/cat_dog_model.pth")
result = predictor.predict("data/test_image.jpg")
print("Résultat :", result)
