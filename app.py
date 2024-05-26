# app.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "Models")
upload_dir = os.path.join(base_dir, "static/uploads")

# Create uploads directory if it doesn't exist
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Load class names
class_names = ['Potato_Early_blight', 'Potato_Healthy', 'Potato_Late_blight',
                'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Healthy',
                'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                'Tomato_Target_Spot', 'Tomato_Tomato_mosaic_virus', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato_Two-spotted_spider_mites']

# Function to load model
def load_model(model_name, num_classes):
    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(os.path.join(models_dir, 'resnet50_best_model.pth')))
    elif model_name == 'vgg16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        model.load_state_dict(torch.load(os.path.join(models_dir, 'vgg16_best_model.pth')))
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(torch.load(os.path.join(models_dir, 'efficientnet_b0_best_model.pth')))
    model.eval()
    return model

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Route to handle home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and model prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the model name from the form
        model_name = request.form['model']
        # Load the selected model
        model = load_model(model_name, len(class_names))
        
        # Get the uploaded image
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)

            # Preprocess the image
            image = Image.open(file_path)
            image = data_transforms(image).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                predicted_class = class_names[preds[0]]

            # Determine the health condition
            if 'Healthy' in predicted_class:
                condition = '100% Healthy'
            else:
                condition = 'Unhealthy'

            # Determine the background image
            if 'Potato' in predicted_class:
                background_image = url_for('static', filename='backgrounds/BG-Result_HTML-Potato.jpg')
            elif 'Tomato' in predicted_class:
                background_image = url_for('static', filename='backgrounds/BG-Result_HTML-Tomato.jpg')
            else:
                background_image = url_for('static', filename='backgrounds/BG-Index-HTML.jpg')

            # Render the result page
            return render_template('result.html', image_url=url_for('static', filename='uploads/' + filename), predicted_class=predicted_class, condition=condition, background_image=background_image)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)
