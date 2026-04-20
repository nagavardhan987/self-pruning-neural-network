import io
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms
from model import SelfPruningNet
import uvicorn

app = FastAPI(title="Self-Pruning NN Inference API")

# Use the lambda=0.001 model as default
MODEL_PATH = "model_lambda_0.001.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SelfPruningNet()

@app.on_event("startup")
def load_model():
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Warning: {MODEL_PATH} not found. Please train the model first.")

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        
    class_idx = predicted.item()
    class_name = CIFAR10_CLASSES[class_idx]
    conf = confidence.item()
    
    sparsity = model.get_sparsity()
    
    return {
        "class": class_name,
        "class_id": class_idx,
        "confidence": conf,
        "model_sparsity_percent": sparsity
    }

@app.get("/stats")
def get_stats():
    sparsity = model.get_sparsity()
    layer_sparsity = model.get_layer_sparsity()
    return {
        "overall_sparsity_percent": sparsity,
        "layer_sparsity_percent": layer_sparsity
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
