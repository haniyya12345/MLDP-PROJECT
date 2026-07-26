import streamlit as st
import pandas as pd
import joblib
# Load the trained model
model = joblib.load("happiness_prediction_model.pkl")

st.set_page_config(
    page_title="MindBalance Analytics",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 MindBalance Analytics")
st.subheader("Lifestyle-Based Happiness Score Prediction")

st.write(
    "Enter your lifestyle information below to receive an estimated "
    "Happiness Score."
)

st.info(
    "This prediction is for general wellness awareness only and is not "
    "a medical diagnosis."
)
st.divider()

st.header("Enter Your Details")

country = st.selectbox(
    "Country",
    ["Australia", "Brazil", "Germany", "India", "Japan", "USA"]
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=65,
    value=30
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

exercise_level = st.selectbox(
    "Exercise Level",
    ["Low", "Moderate", "High"]
)

diet_type = st.selectbox(
    "Diet Type",
    ["Balanced", "Vegetarian", "Vegan", "Junk Food", "Keto"]
)

sleep_hours = st.slider(
    "Sleep Hours",
    min_value=1.0,
    max_value=12.0,
    value=7.0,
    step=0.1
)

stress_level = st.selectbox(
    "Stress Level",
    ["Low", "Moderate", "High"]
)

mental_health_condition = st.selectbox(
    "Mental Health Condition",
    [
        "None Reported",
        "Anxiety",
        "Depression",
        "PTSD"
    ]
)

work_hours = st.slider(
    "Work Hours per Week",
    min_value=0,
    max_value=80,
    value=40
)

screen_time = st.slider(
    "Screen Time per Day (Hours)",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.1
)

social_interaction = st.slider(
    "Social Interaction Score",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)
st.divider()

if st.button("Predict Happiness Score", type="primary"):

    # Create the same engineered features used during model training
    sleep_deviation = abs(sleep_hours - 8)

    work_screen_load = (
        work_hours / 7
        + screen_time
    )

    exercise_mapping = {
        "Low": 1,
        "Moderate": 2,
        "High": 3
    }

    active_social_score = (
        exercise_mapping[exercise_level]
        * social_interaction
    )

    # Create one row of input data
    input_data = pd.DataFrame({
        "Country": [country],
        "Age": [age],
        "Gender": [gender],
        "Exercise Level": [exercise_level],
        "Diet Type": [diet_type],
        "Sleep Hours": [sleep_hours],
        "Stress Level": [stress_level],
        "Mental Health Condition": [mental_health_condition],
        "Work Hours per Week": [work_hours],
        "Screen Time per Day (Hours)": [screen_time],
        "Social Interaction Score": [social_interaction],
        "Sleep Deviation": [sleep_deviation],
        "Work Screen Load": [work_screen_load],
        "Active Social Score": [active_social_score]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Happiness Score: {prediction:.2f} / 10"
    )