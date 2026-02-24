from fastapi import FastAPI,File,UploadFile
from PIL import Image
import io
import torch
from predictions.pred import load_model,transform
from predictions.categories import coco_classes
from api.schemas import PredictionResponse

app = FastAPI()

model = load_model()

@app.get("/")
def health():
    return {"status": "ok"}

# @app.health("/health")
# def health_check():
#     return {
#         'status': 'OK',

#     }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert bytes → PIL → Tensor
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img)
        probs = torch.sigmoid(logits)[0]   # shape: (num_classes,)

    threshold = 0.5
    indices = (probs > threshold).nonzero(as_tuple=True)[0]

    # fallback if nothing crosses threshold
    if len(indices) == 0:
        idx = probs.argmax().item()
        return {
            "class_name": coco_classes[idx],
            "confidence": float(probs[idx].item())
        }

    # return highest confidence class
    idx = indices[probs[indices].argmax()].item()

    return {
        "class_name": [coco_classes[idx]],
        "confidence": float(probs[idx].item())
    }
