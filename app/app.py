"""Streamlit farmer-facing disease prediction demo."""

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.predict import predict


CHECKPOINT = ROOT / "artifacts" / "efficientnet_b0" / "best.pt"


ADVICE = {
    "healthy": "The model found no visible disease pattern. Continue monitoring the crop and maintain good field hygiene.",
    "early_blight": "Remove severely affected leaves, avoid overhead irrigation, and improve airflow around plants.",
    "late_blight": "Remove affected material promptly, avoid wet foliage, and seek local agricultural guidance for treatment.",
    "bacterial_spot": "Avoid handling wet plants, remove badly affected leaves, and sanitize tools between plants.",
    "leaf_mold": "Improve ventilation and reduce prolonged leaf wetness in the crop canopy.",
}


def advice_for(label: str) -> str:
    normalized = label.lower()
    for keyword, advice in ADVICE.items():
        if keyword in normalized:
            return advice
    return "Use this result as an early warning, inspect nearby plants, and consult a qualified agricultural expert before applying treatment."


st.set_page_config(page_title="AgriSmart AI", page_icon="🌿", layout="centered")
st.title("🌿 AgriSmart AI")
st.caption("Plant disease detection powered by an EfficientNet-B0 model trained on PlantVillage.")

if not CHECKPOINT.exists():
    st.error("The trained model file is missing. Expected: artifacts/efficientnet_b0/best.pt")
    st.stop()

uploaded = st.file_uploader("Upload a clear leaf image", type=["jpg", "jpeg", "png"])
if uploaded:
    st.image(uploaded, caption="Uploaded leaf image", use_container_width=True)
    temporary_image = ROOT / "artifacts" / "uploaded_leaf.jpg"
    temporary_image.write_bytes(uploaded.getvalue())
    result = predict(temporary_image, CHECKPOINT)
    st.subheader("Prediction")
    st.success(result["class"])
    st.metric("Confidence", f"{result['confidence']:.1%}")
    st.info(advice_for(result["class"]))
    st.caption("This is a screening result, not a substitute for professional agricultural advice.")
