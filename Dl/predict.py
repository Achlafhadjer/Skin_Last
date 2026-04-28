import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

_BASE = os.path.dirname(os.path.abspath(__file__))


def load_class_names():
    path = os.path.join(_BASE, "class_names.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["classes"]
    return [
        "Actinic keratoses and intraepithelial carcinoma",
        "Basal cell carcinoma",
        "Benign keratosis-like lesions",
        "Dermatofibroma",
        "Melanoma",
        "Melanocytic nevi",
        "Vascular lesions",
    ]


classes = load_class_names()

checkpoint = torch.load(
    os.path.join(_BASE, "best_model.pth"),
    map_location=torch.device("cpu"),
    weights_only=True,
)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(checkpoint)
model.eval()

print("Model loaded ✅")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image_path = os.path.join(_BASE, "test.jpg")

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(image)

_, predicted = torch.max(outputs, 1)
pred_idx = predicted.item()
print("Prediction index:", pred_idx)
print("Prediction disease:", classes[pred_idx])

probs = F.softmax(outputs, dim=1)
top3_prob, top3_idx = torch.topk(probs, 3)

print("\nTop 3 predictions:")
for i in range(3):
    idx = top3_idx[0][i].item()
    prob = top3_prob[0][i].item()
    print(f"{classes[idx]}: {prob:.4f}")
