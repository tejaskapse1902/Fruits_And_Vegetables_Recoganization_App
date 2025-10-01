import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# --- Load model and labels once ---
@st.cache_resource
def load_model_and_labels(model_path="trained_model.h5", labels_path="labels.json"):
    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r") as f:
        labels_dict = json.load(f)
    labels = list(labels_dict.keys())
    return model, labels, labels_dict

model, labels, labels_dict = load_model_and_labels()

# --- Load example images ---
EXAMPLES_DIR = "example_images"
example_images = {}
for label in labels:
    img_path = os.path.join(EXAMPLES_DIR, f"{label}.jpg")
    if os.path.exists(img_path):
        example_images[label] = Image.open(img_path)

# --- Preprocess function ---
def preprocess_image(file_obj, target_size=(64, 64)):
    img = Image.open(file_obj).convert("RGB")
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# --- Prediction functions ---
def predict_image(uploaded_file, threshold=0.1, top_k=3):
    input_arr = preprocess_image(uploaded_file)
    preds = model.predict(input_arr)
    probs = tf.nn.softmax(preds[0]).numpy()
    top_indices = probs.argsort()[-top_k:][::-1]
    top = [(labels[i], float(probs[i])) for i in top_indices]
    best_idx = int(top_indices[0])
    best_conf = float(probs[best_idx])
    if best_conf < threshold:
        return {"label": "Unknown", "confidence": best_conf, "top_k": top}
    return {"label": labels[best_idx], "confidence": best_conf, "top_k": top}

def scale_topk_probs(top_k):
    max_prob = max(prob for label, prob in top_k)
    if max_prob == 0:
        return top_k
    return [(label, prob / max_prob * 100) for label, prob in top_k]

# --- Streamlit UI ---
st.title("Fruits and Vegetables Recognition App")
st.image("home_img.jpg", use_column_width=True)

# --- Switch between upload and camera ---
input_method = st.radio("Choose Input Method:", ("Upload Image", "Live Camera Capture"))

if input_method == "Upload Image":
    image_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("Capture image using your camera")

if image_file is not None:
    st.image(image_file, width=250, caption="Selected Image")
    
    if st.button("Predict"):
        st.write("Predicting...")
        
        # --- Prediction ---
        result = predict_image(image_file, top_k=3)
        scaled_topk = scale_topk_probs(result["top_k"])
        top_label, top_prob = scaled_topk[0]
        info = labels_dict.get(top_label, "No health info available.")

        # --- Top-1 Prediction Card ---
        st.markdown("## Top Prediction")
        st.markdown(f"""
        <div style='border:2px solid rgb(28, 131, 225); padding:15px; border-radius:10px; background-color:rgb(255 255 255 / 0%)'>
            <h3 style='color:rgb(28, 131, 225)'>{top_label} ({top_prob:.1f}%)</h3>
            <p>{info}</p>
        </div>
        """, unsafe_allow_html=True)

        # --- Top-3 Predictions with Example Images ---
        st.markdown("## Top Predictions with Example Images")
        for label, prob in scaled_topk:
            cols = st.columns([1, 3])
            with cols[0]:
                if label in example_images:
                    st.image(example_images[label], width=80)
            with cols[1]:
                st.write(f"**{label}**: {prob:.1f}%")
                st.progress(int(prob))
