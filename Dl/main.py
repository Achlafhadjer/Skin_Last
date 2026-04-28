from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
import json
import os
import time

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_BASE_DIR = os.path.dirname(__file__)

_DEFAULT_CLASSES = [
    "Actinic keratoses and intraepithelial carcinoma",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]

_class_path = os.path.join(_BASE_DIR, "class_names.json")
if os.path.isfile(_class_path):
    with open(_class_path, encoding="utf-8") as f:
        data = json.load(f)
        classes_list = data.get("classes", _DEFAULT_CLASSES)
else:
    classes_list = _DEFAULT_CLASSES

device = torch.device("cpu")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes_list))

checkpoint_path = os.path.join(_BASE_DIR, "best_model.pth")
USE_MOCK = False

if os.path.isfile(checkpoint_path):
    print(f"Loading PyTorch model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    print("✅ PyTorch Model Loaded successfully!")
else:
    print("⚠️ Model file not found. Falling back to MOCK mode.")
    USE_MOCK = True

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.on_event("startup")
def register_consul():
    try:
        import consul
        c = consul.Consul(host='consul', port=8500)
        c.agent.service.register(
            name='ml-service',
            service_id='ml-service-1',
            address='ml-service',
            port=8000,
            tags=['fastapi', 'ml']
        )
        print("ML Service successfully registered to Consul!")
    except Exception as e:
        print(f"Failed to register to Consul: {e}")

@app.get("/api/health")
def health():
    mode = "mock_mode" if USE_MOCK else "production_mode"
    return {"status": mode, "message": f"DL Service running in {mode} with PyTorch"}

@app.post("/api/predict/")
async def predict(image: UploadFile = File(...)):
    # Note: Using 'image' parameter name as expected by the prediction service caller
    image_bytes = await image.read()
    
    if USE_MOCK:
        import random
        time.sleep(1.5)
        main_choice = random.choice(classes_list)
        confidence = round(random.uniform(85, 99), 2)
        return {
            "top3": [
                {"label": main_choice, "confidence": confidence},
                {"label": "Other", "confidence": round(100 - confidence, 2)}
            ],
            "model": "Mock-PyTorch-Model"
        }

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)

        top3_prob, top3_idx = torch.topk(probs, min(3, len(classes_list)))

        results = []
        for i in range(min(3, len(classes_list))):
            idx = top3_idx[0][i].item()
            prob = top3_prob[0][i].item()
            results.append({
                "label": classes_list[idx],
                "confidence": round(prob * 100, 2),
            })

        return {
            "top3": results,
            "model": "PyTorch-Skin-EfficientNet"
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": "Prediction failed", "details": str(e)}

class ChatMessage(BaseModel):
    sender: str
    text: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

@app.post("/api/chat/")
async def chat(request: ChatRequest):
    user_msg = request.message
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            genai_model = genai.GenerativeModel('gemini-2.5-flash')
            
            gemini_history = [
                {"role": "user", "parts": ["Tu es SkinBot, un assistant dermatologique IA créé pour le projet Skin AI. Réponds toujours en français, sois très organisé dans tes réponses. Utilise des paragraphes courts, des listes à puces avec des tirets, et du texte en gras (avec **) pour mettre en évidence les informations importantes. Utilise des emojis appropriés (comme 👨‍⚕️, 🩺, 💡, ⚠️, 🌿) pour rendre la lecture plus agréable. Sois professionnel mais empathique. Rappelle TOUJOURS que tu ne remplaces pas un avis médical."]},
                {"role": "model", "parts": ["Compris. Je suis SkinBot et je suis prêt à vous aider en français."]}
            ]
            
            if request.history:
                for msg in request.history:
                    role = "user" if msg.sender == "user" else "model"
                    gemini_history.append({"role": role, "parts": [msg.text]})
            
            chat_session = genai_model.start_chat(history=gemini_history)
            response = chat_session.send_message(user_msg)
            return {"reply": response.text}
        except Exception as e:
            print("Gemini error:", str(e))
            pass

    time.sleep(1)
    return {
        "reply": "⚠️ L'API Gemini n'est pas encore configurée (Clé API manquante dans le fichier .env). Pour rendre ce chatbot intelligent et permanent, veuillez ajouter `GEMINI_API_KEY=votre_cle` dans le fichier .env de votre projet."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
