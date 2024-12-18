import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreprocessor:
    def __init__(self, train_dir, validation_dir):
        self.train_dir = train_dir
        self.validation_dir = validation_dir

    def get_generators(self):
        # Use ImageDataGenerator to preprocess the images
        train_datagen = ImageDataGenerator(rescale=1./255)
        validation_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(150, 150),  # Resize images
            batch_size=32,
            class_mode='binary'  # For binary classification (cat vs dog)
        )

        validation_generator = validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary'
        )

        return train_generator, validation_generator
