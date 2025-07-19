import streamlit as st
from PIL import Image
from utils import predict_image

st.set_page_config(page_title="PPE Detection", layout="centered")
st.title("🧠 PPE Detection System (Mask + Helmet)")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict_image(image)

    emoji_map = {
        "mask": "😷 Mask",
        "nomask": "🚫 No Mask",
        "helmet": "🧢 Helmet",
        "nohelmet": "❌ No Helmet"
    }

    st.success(f"**Prediction:** {emoji_map.get(label, label)}  \n**Confidence:** {confidence:.2f}%")
