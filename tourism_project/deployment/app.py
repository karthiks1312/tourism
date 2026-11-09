import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Prediction")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")
MODEL_REPO = f"{HF_REPO_ID}/tourism-package-model"
MODEL_FILENAME = "best_pipeline.joblib"

st.title("🧳 Tourism Package Prediction")
st.write("Predict whether a customer is likely to purchase the Wellness Tourism Package.")

if not HF_REPO_ID:
    st.error("HF_REPO_ID secret is missing in the Space settings")
    st.stop()


@st.cache_resource
def load_model():
    try:
        path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, token=HF_TOKEN)
        return joblib.load(path)
    except Exception as e:
        st.error("Failed to download model from Hugging Face")
        st.exception(e)
        st.stop()


model = load_model()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 80, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", 1000, 100000, 20000, step=500)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    own_car = st.selectbox("Owns a Car", [1, 0], format_func=lambda x: "Yes" if x else "No")
    passport = st.selectbox("Has Passport", [1, 0], format_func=lambda x: "Yes" if x else "No")
with col2:
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of Pitch (min)", 5, 130, 15)
    num_followups = st.number_input("Number of Follow-ups", 1, 6, 3)
    pitch_satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2)
    preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    num_persons = st.number_input("Number of Persons Visiting", 1, 5, 3)
    num_children = st.number_input("Number of Children Visiting", 0, 3, 0)
    num_trips = st.number_input("Number of Trips per Year", 1, 22, 3)

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict"):
    proba = float(model.predict_proba(input_data)[0][1])
    if proba >= 0.5:
        st.success(f"Likely to purchase the package (probability {proba:.1%})")
    else:
        st.warning(f"Unlikely to purchase the package (probability {proba:.1%})")
