from tensorflow.keras import layers, models

class ModelTrainer:
    def __init__(self):
        self.model = None

    def build_model(self):
        # Build a simple CNN model
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification (cat or dog)
        ])

        # Compile the model
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train(self, train_generator, validation_generator, epochs=5):
        # Train the model
        self.model.fit(
            train_generator,
            steps_per_epoch=100,  # Depends on the number of images
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=50   # Depends on the number of images
        )

    def save_model(self, model_path):
        # Save the trained model
        self.model.save(model_path)
