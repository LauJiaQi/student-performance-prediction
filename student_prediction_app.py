import joblib
import pandas as pd
import streamlit as st
import numpy as np

education_map = {
    'No Formal quals': 0,
    'Lower Than A Level': 1,
    'A Level or Equivalent': 2,
    'HE Qualification': 3,
    'Post Graduate Qualification': 4,
}

age_map = {'0-35': 0, '35-55': 1, '55<=': 2}

imd_map = {
    '0-10%': 0, '10-20%': 1, '20-30%': 2, '30-40%': 3, '40-50%': 4,
    '50-60%': 5, '60-70%': 6, '70-80%': 7, '80-90%': 8,
    '90-100%': 9, 'Unknown': 10
}

st.title("Student Performance Prediction")
st.write("Predict student outcomes based on engagement and demographics")

@st.cache_resource
def load_artifacts():
    rf = joblib.load("best_model.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return rf, feature_cols

rf, feature_cols = load_artifacts()

# Create 4 columns for input
col1, col2, col3, col4 = st.columns(4)

with col1:
    age_band = st.selectbox("Age Band", list(age_map.keys()), index=0)
    highest_education = st.selectbox("Highest Education", list(education_map.keys()), index=2)
    disability = st.selectbox("Disability", ["No", "Yes"], index=0)

with col2:
    imd_band = st.selectbox("IMD Band", list(imd_map.keys()), index=4)
    studied_credits = st.number_input("Studied Credits", min_value=30, value=60)
    num_of_prev_attempts = st.number_input("Previous Attempts", min_value=0, value=0)

with col3:
    active_days_count = st.number_input("Number of Active Days", min_value=0, value=40)
    total_clicks = st.number_input("Total Clicks", min_value=0, value=600)

with col4:
    mean_score = st.number_input("Mean Assessment Score", min_value=0.0, max_value=100.0, value=70.57)

if st.button("Predict"):
    log_active_days = np.log1p(active_days_count)
    log_total_clicks = np.log1p(total_clicks)

    user_input = {
        'log_active_days': log_active_days,
        'mean_score': mean_score,
        'log_total_clicks': log_total_clicks,
        'studied_credits': studied_credits,
        'num_of_prev_attempts': num_of_prev_attempts,
        'imd_band': imd_map[imd_band],
        'highest_education': education_map[highest_education],
        'age_band': age_map[age_band],
        'disability': 1 if disability == "Yes" else 0,
    }

    input_df = pd.DataFrame([user_input]).reindex(columns=feature_cols)

    proba = rf.predict_proba(input_df)[0][1]
    pred = int(proba >= 0.5)

    st.subheader("Prediction Result")
    st.write(f"**Predicted class:** {pred}  (1 = Pass/Distinction, 0 = Fail/Withdrawn)")
    st.write(f"**Probability of Pass/Distinction:** {proba:.3f}")

    if proba < 0.4:
        st.error("⚠️ High risk: student may need intervention/support.")
    elif proba < 0.6:
        st.warning("⚡ Medium risk: monitor engagement and assessment progress.")
    else:
        st.success("✅ Low risk: student is likely to perform well.")
