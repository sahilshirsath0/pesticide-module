import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from flask import Blueprint, request, jsonify
from PIL import Image
from class_labels import disease_pesticide_mapping  # Import mapping from class_label.py

# Blueprint Setup
disease_bp = Blueprint("disease", __name__)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "crop_disease_model.pth"
CLASS_PATH = BASE_DIR / "classes.pth"

DEVICE = torch.device("cpu")

# Load class names
class_names = torch.load(
    CLASS_PATH,
    map_location="cpu"
)

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval() # Adjust for output classes
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@disease_bp.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the uploaded file
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        
        # Convert image to correct format
        image = Image.open(file).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            disease_name = class_names[predicted.item()]

        # Get recommended pesticides from class_label.py
        recommended_pesticides = disease_pesticide_mapping.get(disease_name, [])

        return jsonify({
            "disease": disease_name,
            "recommended_pesticides": recommended_pesticides if recommended_pesticides else ["No pesticides found"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
