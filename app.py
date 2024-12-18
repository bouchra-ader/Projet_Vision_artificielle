from flask import Flask, render_template, request
from src.predict import Predictor

app = Flask(__name__)

# Initialize the model for predictions
predictor = Predictor('models/cat_dog_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file part', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400
    result = predictor.predict(file)
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
