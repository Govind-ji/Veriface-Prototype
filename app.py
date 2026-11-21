import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

st.set_page_config(page_title="VeriFace", page_icon="🔍")

st.title("VeriFace - Deepfake Detection")
st.write("Upload an image to check whether it is Real or Fake.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    checkpoint = torch.load("veriface_model.pth", map_location=device)

    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.fc.in_features, 2)
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, checkpoint["class_to_idx"]

model, class_to_idx = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect"):

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        prediction = torch.argmax(probabilities, dim=1).item()

        classes = {v: k for k, v in class_to_idx.items()}

        result = classes[prediction]
        confidence = probabilities[0][prediction].item() * 100

        st.subheader(f"Prediction: {result.upper()}")
        st.write(f"Confidence: {confidence:.2f}%")
