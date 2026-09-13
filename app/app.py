"""Streamlit farmer-facing disease prediction demo."""

from pathlib import Path
import sys

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


st.set_page_config(page_title="AgriSmart AI", page_icon="🌿", layout="centered")
st.title("🌿 AgriSmart AI")
st.caption("Plant disease detection powered by an EfficientNet-B0 model trained on PlantVillage.")

with st.sidebar:
    st.header("Farm location")
    st.session_state.setdefault("farm_location", {"lat": 20.5937, "lon": 78.9629})
    st.session_state.setdefault("show_location_map", False)
    location = st.session_state["farm_location"]
    if st.button("Select location on map", use_container_width=True):
        st.session_state["show_location_map"] = True
    latitude = location["lat"]
    longitude = location["lon"]
    st.caption(f"Selected: {latitude:.4f}, {longitude:.4f}")
    season = st.selectbox("Season", ["Kharif", "Rabi", "Summer", "Other"])
    soil_moisture = st.slider("Soil moisture (%)", 0, 100, 35)

if st.session_state.get("show_location_map"):
    st.subheader("Select your farm location")
    st.caption("Click the map to place the farm marker, then close this section.")
    location = st.session_state["farm_location"]
    farm_map = folium.Map(location=[location["lat"], location["lon"]], zoom_start=5, control_scale=True)
    folium.Marker([location["lat"], location["lon"]], tooltip="Selected farm").add_to(farm_map)
    map_state = st_folium(farm_map, height=500, use_container_width=True, key="farm_location_map")
    if map_state and map_state.get("last_clicked"):
        clicked = map_state["last_clicked"]
        st.session_state["farm_location"] = {"lat": clicked["lat"], "lon": clicked["lng"]}
        st.success(f"Location selected: {clicked['lat']:.4f}, {clicked['lng']:.4f}")
    if st.button("Done selecting location"):
        st.session_state["show_location_map"] = False
        st.rerun()
    location = st.session_state["farm_location"]
    latitude = location["lat"]
    longitude = location["lon"]

if not CHECKPOINT.exists():
    st.error("The trained model file is missing. Expected: artifacts/mixed_finetuned.pt")
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
    st.warning("Weather service is unavailable. The disease classifier still works.")

st.header("🪨 Soil information")
try:
    soil = get_soil(latitude, longitude)
    soil_ph = soil.get("ph")
    if soil_ph is not None:
        st.metric("Estimated topsoil pH", f"{soil_ph:.1f}")
    else:
        st.info("SoilGrids did not return a pH value for this location.")
except requests.RequestException:
    soil_ph = None
    st.warning("SoilGrids is unavailable. Entering local soil measurements is recommended.")

st.header("🌾 Crop recommendation")
temperature = weather.get("temperature") if "weather" in locals() else None
recommendations = recommend_crops(soil_ph, temperature, soil_moisture, season)
st.write(f"Indicative recommendations for {season}: " + ", ".join(recommendations))
st.caption("These are educational recommendations based on simple rules, not professional agricultural advice.")
