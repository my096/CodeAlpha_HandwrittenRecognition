import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("handwritten_model.h5")

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert('L')

    image = image.resize((28, 28))

    img_array = np.array(image) / 255.0

    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)

    predicted_digit = np.argmax(prediction)

    st.image(image, caption="Uploaded Image")

    st.success(f"Predicted Digit: {predicted_digit}")
