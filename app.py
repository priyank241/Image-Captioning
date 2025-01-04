import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from utils import generate_caption,load_models



model,fe,max_length,tokenizer=load_models()


# Set up the side menu
st.sidebar.title("Menu")
st.sidebar.write("Use The Menu To Navigate")

# Image uploader in the sidebar
st.sidebar.title("Image Upload")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

st.title("Image Upload Example")

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    img = load_img((uploaded_file), target_size=(224, 224))
    predicted_caption = generate_caption(model, tokenizer, fe, img, max_length)
    # Resize the image to a maximum width and height
    max_size = (300,400)  # Set the maximum dimensions (width, height)
    image.thumbnail(max_size)

    st.image(image, caption="Uploaded Image (Resized)")
    st.write(predicted_caption)
else:
    st.write("Please upload an image using the sidebar.")