from src.data_preprocessing import DataPreprocessor
from src.train_model import ModelTrainer

# Initialize the data preprocessor
preprocessor = DataPreprocessor("data/raw/train", "data/raw/validation")
train_gen, val_gen = preprocessor.get_generators()

# Initialize and train the model
trainer = ModelTrainer()
trainer.build_model()
trainer.train(train_gen, val_gen, epochs=5)
trainer.save_model("models/cat_dog_model.h5")
