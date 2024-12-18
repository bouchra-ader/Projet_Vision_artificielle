import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

class Predictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, img):
        # Load and preprocess the image
        img = image.load_img(img, target_size=(150, 150))  # Match the model input size
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch size 1
        prediction = self.model.predict(img_array)
        
        # Return result based on the prediction
        if prediction[0] > 0.5:
            return 'Dog'
        else:
            return 'Cat'
