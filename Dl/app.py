import json
import os

from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

app = Flask(__name__)

# Ordre des indices = celui du dataset / entraînement (voir class_names.json)
_DEFAULT_CLASSES = [
    "Actinic keratoses and intraepithelial carcinoma",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]

_class_path = os.path.join(os.path.dirname(__file__), "class_names.json")
if os.path.isfile(_class_path):
    with open(_class_path, encoding="utf-8") as f:
        classes_list = json.load(f)["classes"]
else:
    classes_list = _DEFAULT_CLASSES

device = torch.device("cpu")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes_list))

checkpoint = torch.load(
    os.path.join(os.path.dirname(__file__), "best_model.pth"),
    map_location=device,
    weights_only=True,
)
model.load_state_dict(checkpoint)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict/", methods=["POST"])
def predict():
    if "file" not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["file"]
    if file.filename == "":
        return {"error": "No file selected"}, 400
        
    ext = file.filename.split('.')[-1].lower()
    if ext not in ['jpg', 'jpeg', 'png', 'webp']:
        return {"error": "Invalid file format. Please upload an image (jpg, png)."}, 400
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    top3_prob, top3_idx = torch.topk(probs, 3)

    results = []
    for i in range(3):
        idx = top3_idx[0][i].item()
        prob = top3_prob[0][i].item()

        results.append({
            "label": classes_list[idx],
            "confidence": round(prob * 100, 2),
        })

    return {"top3": results}


if __name__ == "__main__":
    app.run(debug=True)
