from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
from CNN import CNN, idx_to_classes

app = FastAPI()

# Load the model
model = CNN(K=len(idx_to_classes))
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Perform prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_name = idx_to_classes[predicted.item()]

    return {"class": class_name}
