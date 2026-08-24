import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Charger le modèle une seule fois
@st.cache_resource
def load_model():
    model = torch.load("data_processed/best_model.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()

st.title("Détection des maladies bovines")

uploaded_file = st.file_uploader(
    "Choisissez une image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image sélectionnée")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        prediction = output.argmax(1).item()

    st.write("Classe prédite :", prediction)