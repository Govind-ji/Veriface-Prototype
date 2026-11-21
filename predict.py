import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("veriface_model.pth", map_location=device)

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_features, 2)
)

model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_path = sys.argv[1]

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()

classes = {v: k for k, v in checkpoint["class_to_idx"].items()}

print("Prediction:", classes[prediction])
print("Confidence:", round(probabilities[0][prediction].item() * 100, 2), "%")
