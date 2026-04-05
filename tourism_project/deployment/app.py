import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="waseemsk22/Tourism-Purchase-Prediction", filename="best_tourism_purchase_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Prediction
st.title("Tourism Prediction App")
st.write("""
This application predicts the potential buyers of tourism package based on different parameters.
Please enter the data below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=2)
NumberOfPersonVisiting = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=2)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5)
NumberOfTrips = st.number_input("Number of Trips", min_value=0, max_value=10)
Passport = st.number_input("Passport", min_value=0, max_value=1)
OwnCar = st.number_input("Own Car", min_value=0, max_value=1)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10)
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0,
                                max_value=100000.0,
                                value=50000.0,
                                step=100.0)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score",
                                         min_value=1,
                                         max_value=5,
                                         value=3)
NumberOfFollowups = st.number_input("Number of Followups",
                                    min_value=0,
                                    max_value=10,
                                    value=2)
DurationOfPitch = st.number_input("Duration of Pitch",
                                  min_value=1,
                                  max_value=10,
                                  value=5)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Customer", "Phone Number"])
Occupation = st.selectbox("Occupation", ["Unemployed", "Student", "Working Professional"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
Designation = st.selectbox("Designation", ["Executive", "Managerial", "Professional", "Other"])
ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Standard", "Basic"])


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'City Tier': CityTier,
    'Number of Person Visiting': NumberOfPersonVisiting,
    'Preferred Property Star': PreferredPropertyStar,
    'Number of Trips': NumberOfTrips,
    'Passport': Passport,
    'Own Car': OwnCar,
    'Number of Children Visiting': NumberOfChildrenVisiting,
    'Monthly Income': MonthlyIncome,
    'Pitch Satisfaction Score': PitchSatisfactionScore,
    'Number of Followups': NumberOfFollowups,
    'Duration of Pitch': DurationOfPitch,
    'Type of Contact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'Marital Status': MaritalStatus,
    'Designation': Designation,
    'Product Pitched': ProductPitched
}])


if st.button("Predict Purchasse"):
    prediction = model.predict(input_data)[0]
    result = "Customer makes purchase" if prediction == 1 else "Customer doesn't make purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts that: **{result}**")
