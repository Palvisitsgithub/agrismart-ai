"""Streamlit farmer-facing disease prediction demo."""

from pathlib import Path
import sys
import time

import requests
import folium
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.predict import predict


CHECKPOINT = ROOT / "artifacts" / "mixed_finetuned.pt"


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


@st.cache_data(ttl=1800)
def get_weather(latitude: float, longitude: float) -> dict:
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain",
            "daily": "precipitation_probability_max,precipitation_sum",
            "forecast_days": 1,
            "timezone": "auto",
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    current = payload.get("current", {})
    daily = payload.get("daily", {})
    return {
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "rain": current.get("rain", 0),
        "rain_probability": (daily.get("precipitation_probability_max") or [None])[0],
        "rain_sum": (daily.get("precipitation_sum") or [None])[0],
    }


@st.cache_data(ttl=86400)
def get_soil(latitude: float, longitude: float) -> dict:
    response = requests.get(
        "https://rest.isric.org/soilgrids/v2.0/properties/query",
        params={"lon": longitude, "lat": latitude, "property": "phh2o", "depth": "0-5cm", "value": "mean"},
        timeout=20,
    )
    response.raise_for_status()
    layers = response.json().get("properties", {}).get("layers", [])
    for layer in layers:
        if layer.get("name") == "phh2o":
            depths = layer.get("depths", [])
            if depths and depths[0].get("values", {}).get("mean") is not None:
                return {"ph": depths[0]["values"]["mean"] / 10}
    return {"ph": None}


def recommend_crops(ph: float | None, temperature: float | None, moisture: float, season: str) -> list[str]:
    recommendations = []
    if ph is None:
        return ["Enter soil pH or use the SoilGrids lookup to receive recommendations."]
    if 5.5 <= ph <= 7.0 and moisture >= 25:
        recommendations.extend(["Tomato", "Potato", "Corn"])
    if 6.0 <= ph <= 7.5 and 20 <= moisture <= 60:
        recommendations.append("Grape")
    if 5.5 <= ph <= 6.8 and (temperature is None or temperature >= 18):
        recommendations.append("Apple")
    if not recommendations:
        recommendations.append("Choose a crop after local agronomist review; the current soil inputs do not match the demo rules.")
    return list(dict.fromkeys(recommendations))


st.set_page_config(page_title="AgriSmart AI", page_icon="🌿", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    div[data-testid="stFileUploader"] {
        border: 1px dashed #79a982;
        border-radius: 16px;
        padding: 0.6rem;
        background: #f7fbf7;
    }
    div[data-testid="stMetric"] {
        background: #f7fbf7;
        border: 1px solid #e2eee3;
        border-radius: 14px;
        padding: 0.8rem;
    }
    div.stButton > button {
        border-radius: 10px;
        min-height: 2.6rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🌿 AgriSmart AI")
st.caption("Choose a feature below to scan a plant or review farm conditions.")

st.session_state.setdefault("farm_location", {"lat": 20.5937, "lon": 78.9629})
st.session_state.setdefault("location_confirmed", False)

scan_tab, farm_tab = st.tabs(["📷 Scan plant photo", "📍 Farm location & soil"])

with scan_tab:
    st.header("Scan a plant photo")
    st.write("Upload a clear leaf image to receive a disease screening result.")
    if not CHECKPOINT.exists():
        st.error("The trained model file is missing. Expected: artifacts/mixed_finetuned.pt")
    else:
        uploaded = st.file_uploader(
            "Upload a clear leaf image",
            type=["jpg", "jpeg", "png"],
            key="plant_scan_upload",
        )
        if uploaded:
            preview_col, details_col = st.columns([1.15, 1], gap="large")
            with preview_col:
                st.image(uploaded, caption="Uploaded leaf image", use_container_width=True)
            with details_col:
                st.info("The image will be checked against the trained plant-disease model.")
            temporary_image = ROOT / "artifacts" / "uploaded_leaf.jpg"
            temporary_image.write_bytes(uploaded.getvalue())
            with st.status("Analyzing your plant image...", expanded=True) as analysis_status:
                st.write("Preparing image")
                time.sleep(0.35)
                st.write("Comparing visual patterns")
                time.sleep(0.35)
                result = predict(temporary_image, CHECKPOINT)
                time.sleep(0.35)
                st.write("Preparing your result")
                analysis_status.update(label="Analysis complete", state="complete", expanded=False)
            st.subheader("Prediction")
            result_col, confidence_col = st.columns([2, 1])
            with result_col:
                st.success(result["class"])
            with confidence_col:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            st.info(advice_for(result["class"]))
            st.caption("This is a screening result, not a substitute for professional agricultural advice.")

with farm_tab:
    st.header("Farm location & soil information")
    st.write("Choose your farm location to view weather, soil, and crop recommendations.")

    location = st.session_state["farm_location"]
    latitude = location["lat"]
    longitude = location["lon"]
    confirmed = st.session_state["location_confirmed"]
    st.caption(f"Selected location: {latitude:.4f}, {longitude:.4f}")

    if confirmed:
        st.success("Location confirmed")
        if st.button("Select location again", use_container_width=False):
            st.session_state["location_confirmed"] = False
            st.rerun()
        st.caption("The map is locked. Select the button above to choose a different location.")
    else:
        st.info("Click the map to place the marker, then confirm the location.")

    farm_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=5,
        control_scale=True,
        zoom_control=not confirmed,
        dragging=not confirmed,
        scrollWheelZoom=not confirmed,
        doubleClickZoom=not confirmed,
        touchZoom=not confirmed,
    )
    folium.Marker([latitude, longitude], tooltip="Selected farm").add_to(farm_map)
    map_state = st_folium(farm_map, height=500, use_container_width=True, key="farm_location_map")
    if not confirmed and map_state and map_state.get("last_clicked"):
        clicked = map_state["last_clicked"]
        st.session_state["farm_location"] = {"lat": clicked["lat"], "lon": clicked["lng"]}
        st.rerun()

    if not confirmed:
        if st.button("Confirm location", type="primary"):
            st.session_state["location_confirmed"] = True
            st.rerun()

    controls_col, spacer_col = st.columns([1, 3])
    with controls_col:
        season = st.selectbox("Season", ["Kharif", "Rabi", "Summer", "Other"], key="farm_season")
        soil_moisture = st.slider("Soil moisture (%)", 0, 100, 35, key="farm_soil_moisture")

    st.divider()
    st.header("🌦️ Weather intelligence")
    try:
        weather = get_weather(latitude, longitude)
        weather_columns = st.columns(4)
        weather_columns[0].metric("Temperature", f"{weather['temperature']} °C")
        weather_columns[1].metric("Humidity", f"{weather['humidity']} %")
        weather_columns[2].metric("Rain probability", f"{weather['rain_probability']} %")
        weather_columns[3].metric("Rain forecast", f"{weather['rain_sum']} mm")
        if weather["rain_probability"] is not None and weather["rain_probability"] >= 60:
            st.info("Rain is likely. Consider delaying irrigation and avoid wetting leaves.")
        else:
            st.info("Rain risk is currently lower. Check soil moisture before irrigating.")
    except requests.RequestException:
        weather = {}
        st.warning("Weather service is unavailable. Location-based recommendations may be limited.")

    st.header("🪨 Soil information")
    try:
        soil = get_soil(latitude, longitude)
        soil_ph = soil.get("ph")
        if soil_ph is not None:
            st.metric("Estimated topsoil pH", f"{soil_ph:.1f}")
        else:
            st.info("Soil data did not return a pH value for this location.")
    except requests.RequestException:
        soil_ph = None
        st.warning("Soil data is unavailable. Entering local soil measurements is recommended.")

    st.header("🌾 Crop recommendation")
    temperature = weather.get("temperature")
    recommendations = recommend_crops(soil_ph, temperature, soil_moisture, season)
    st.write(f"Indicative recommendations for {season}: " + ", ".join(recommendations))
    st.caption("These are educational recommendations based on simple rules, not professional agricultural advice.")
