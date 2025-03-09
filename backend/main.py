import torch
import torchvision.transforms as transforms
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np

MODEL_PATH = "model/skin_cancer_model.pth"

# Initialize FastAPI
app = FastAPI()

# Configure CORS middleware (Put this right after app initialization)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load class labels (must match your original dataset)
class_labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Load the trained model
class SkinCancerResNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(SkinCancerResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)  # Set to False to avoid downloading weights
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Initialize model and load weights
num_classes = len(class_labels)
model = SkinCancerResNet(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
# Set model to evaluation mode
model.eval()

# Image transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.get("/")
def home():
    return {"message": "Skin Cancer Detection API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    
    # Preprocess
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    # Get class label
    predicted_label = class_labels[predicted_class.item()]
    
    return {"predicted_label": predicted_label}
