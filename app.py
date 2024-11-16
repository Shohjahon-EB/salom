import streamlit as st
import pathlib
import plotly.express as px

# Try to import fastai
try:
    from fastai.vision.all import *
except ImportError:
    st.error("The 'fastai' library is not installed. Please install it using 'pip install fastai'.")
    st.stop()

# Fix for Windows path issue
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Add instructions for running the app
st.set_page_config(page_title="Transport Image Classifier")
st.title("Transport Image Classifier")
st.write("To run this app, use the command: streamlit run app.py in your terminal.")

# Modelni yuklash (Load the model)
@st.cache_resource
def load_model():
    return load_learner("model.pkl")

try:
    learner = load_model()
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please make sure it's in the same directory as this script.")
    st.stop()

# Rasm yuklash (Upload image)
uploaded_file = st.file_uploader("Rasm yuklang", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Yuklangan rasmni o'qish (Read uploaded image)
    img = PILImage.create(uploaded_file)
    # Rasmni aniqlash (Predict image)
    try:
        pred, pred_idx, probs = learner.predict(img)
        
        # Natijani ko'rsatish (Show results)
        st.image(img, caption='Yuklangan rasm', use_column_width=True)
        st.write(f"Bu rasm: {pred} (Ishonch: {probs[pred_idx]:.2f})")
    except Exception as e:
        st.error(f"Rasmni aniqlashda xato: {e}")








































