import streamlit as st
import pandas as pd
import joblib
import base64
import os

# ----------------------------------------
# Load your ML model
# ----------------------------------------
model = joblib.load('breast_cancer_model.pkl')

# ----------------------------------------
# Page configuration
# ----------------------------------------
st.set_page_config(page_title="💜 Breast Cancer Survival Prediction", layout="centered")

# ----------------------------------------
# Load video file (MP4)
# ----------------------------------------
video_path = os.path.abspath(r"C:\Users\Lenovo\Desktop\breast_cancer_model\bgvdo.mp4")
with open(video_path, "rb") as file:
    video_bytes = file.read()
video_base64 = base64.b64encode(video_bytes).decode("utf-8")

# ----------------------------------------
# Inject video once (no reload = no flicker)
# ----------------------------------------
if "background_video" not in st.session_state:
    st.session_state["background_video"] = f"""
    <video autoplay loop muted playsinline id="bg-video">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    <div class="overlay"></div>
    """

st.markdown(st.session_state["background_video"], unsafe_allow_html=True)

# ----------------------------------------
# Custom CSS (video, layout, buttons, etc.)
# ----------------------------------------
st.markdown("""
<style>
/* Hide Streamlit UI */
#MainMenu, header, footer {visibility: hidden;}

/* Fullscreen background video */
#bg-video {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    object-fit: cover;
    z-index: -100;
}

/* Dark overlay for contrast */
.overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0,0,0,0.35);
    z-index: -99;
}

/* Main glass container */
.main-container {
    position: relative;
    z-index: 10;
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 40px 50px;
    max-width: 650px;
    margin: 60px auto;
    color: white;
    font-family: "Segoe UI", sans-serif;
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
}

/* Text */
h1, h4 {
    text-align: center;
    color: white;
}
label {
    color: white !important;
    font-weight: 600;
}

/* ✨ Whitish bling buttons */
.stButton>button {
    width: 100%;
    background-color: rgba(255,255,255,0.9);
    color: black;
    font-weight: 700;
    font-size: 18px;
    border-radius: 12px;
    padding: 12px;
    border: 2px solid white;
    box-shadow: 0 0 15px rgba(255,255,255,0.6);
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    background-color: white;
    color: black;
    transform: scale(1.05);
    box-shadow: 0 0 35px rgba(255,255,255,1);
}

/* Feature caption styling */
.feature-info {
    font-size: 14px;
    color: #e0e0e0;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# Feature explanations
# ----------------------------------------
feature_info = {
    "Age": "Older patients often have weaker immune response affecting recovery.",
    "Race": "Genetic differences among races may influence tumor behavior.",
    "Marital Status": "Married patients often receive better support and care.",
    "Tumor (T) Stage": "Represents how large the tumor is and how far it has grown.",
    "Node (N) Stage": "Shows if cancer has spread to lymph nodes nearby.",
    "6th Stage": "Overall stage combining tumor size and node spread.",
    "Cell Differentiation": "Describes how abnormal the cells look. Poor = faster spread.",
    "Tumor Grade": "Indicates tumor aggressiveness.",
    "Anatomic Stage": "Overall cancer stage (I–IV).",
    "Tumor Size": "Larger tumor → more advanced disease.",
    "Estrogen Status": "Positive = better response to hormone therapy.",
    "Progesterone Status": "Positive = better therapy outcome.",
    "Regional Nodes Examined": "How many nodes were tested.",
    "Reginol Nodes Positive": "How many nodes had cancer.",
    "Survival Months": "Months patient survived after treatment or diagnosis."
}

# ----------------------------------------
# Title and intro
# ----------------------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("💜 Breast Cancer Survival Prediction")
st.markdown("<h4>AI-Powered Prognosis System</h4>", unsafe_allow_html=True)
st.write("---")

# ----------------------------------------
# Helper to show description + input
# ----------------------------------------
def feature_input(label, widget):
    st.markdown(f"<p class='feature-info'>🩺 {feature_info[label]}</p>", unsafe_allow_html=True)
    return widget

# ----------------------------------------
# Input fields (vertical layout)
# ----------------------------------------
age = feature_input("Age", st.number_input("Age", 20, 100, 50, key="age"))
race = feature_input("Race", st.selectbox("Race (Encoded)", [1,2,3,4,5], key="race"))
marital_status = feature_input("Marital Status", st.selectbox("Marital Status", [1,2,3], key="marital", help="1=Married, 2=Single, 3=Divorced"))
t_stage = feature_input("Tumor (T) Stage", st.selectbox("T Stage", [1,2,3,4], key="t_stage"))
n_stage = feature_input("Node (N) Stage", st.selectbox("N Stage", [0,1,2,3], key="n_stage"))
stage6 = feature_input("6th Stage", st.selectbox("6th Stage", [1,2,3,4], key="stage6"))
differentiate = feature_input("Cell Differentiation", st.selectbox("Differentiation", [1,2,3], key="diff"))
grade = feature_input("Tumor Grade", st.selectbox("Grade", [0,1,2,3], key="grade"))
a_stage = feature_input("Anatomic Stage", st.selectbox("Anatomic Stage", [1,2,3,4], key="a_stage"))
tumor_size = feature_input("Tumor Size", st.number_input("Tumor Size (mm)", 1, 200, 30, key="tumor_size"))
estrogen_status = feature_input("Estrogen Status", st.selectbox("Estrogen Status", [0,1], key="estrogen"))
progesterone_status = feature_input("Progesterone Status", st.selectbox("Progesterone Status", [0,1], key="progesterone"))
regional_nodes_examined = feature_input("Regional Nodes Examined", st.number_input("Nodes Examined", 0, 50, 10, key="nodes_examined"))
reginol_nodes_positive = feature_input("Reginol Nodes Positive", st.number_input("Positive Nodes", 0, 50, 2, key="nodes_positive"))
survival_months = feature_input("Survival Months", st.number_input("Survival Months", 1, 120, 36, key="survival"))

# ----------------------------------------
# Prepare dataframe and predict
# ----------------------------------------
input_data = pd.DataFrame({
    'Age': [age],
    'Race': [race],
    'Marital Status': [marital_status],
    'T Stage ': [t_stage],
    'N Stage': [n_stage],
    '6th Stage': [stage6],
    'differentiate': [differentiate],
    'Grade': [grade],
    'A Stage': [a_stage],
    'Tumor Size': [tumor_size],
    'Estrogen Status': [estrogen_status],
    'Progesterone Status': [progesterone_status],
    'Regional Node Examined': [regional_nodes_examined],
    'Reginol Node Positive': [reginol_nodes_positive],
    'Survival Months': [survival_months]
})

if st.button("Predict Survival Status", key="predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Prediction: ALIVE — Strong prognosis, continue treatment and stay hopeful!")
    else:
        st.error("⚠️ Prediction: DECEASED — Seek immediate medical guidance and emotional support.")

st.markdown("<h6 style='text-align:center; color:white;'>Built with ❤️ using Streamlit & AI</h6>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
