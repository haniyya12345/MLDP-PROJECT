import streamlit as st
import pandas as pd
import joblib


# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("happiness_prediction_model.pkl")


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "The model file could not be found. Make sure "
        "'happiness_prediction_model.pkl' is in the same folder as app.py."
    )
    st.stop()

except Exception as error:
    st.error("The model could not be loaded.")
    st.write(error)
    st.stop()


# Page settings
st.set_page_config(
    page_title="MindBalance Analytics",
    page_icon="🧠",
    layout="centered"
)


# App title and description
st.title("🧠 MindBalance Analytics")
st.subheader("Lifestyle-Based Happiness Score Prediction")

st.write(
    "Enter your lifestyle information below to receive an estimated "
    "Happiness Score."
)

st.info(
    "This prediction is for general wellness awareness only. "
    "It is not medical advice or a mental-health diagnosis."
)

st.divider()


# User input section
st.header("Enter Your Details")

country = st.selectbox(
    "Country",
    [
        "Australia",
        "Brazil",
        "Germany",
        "India",
        "Japan",
        "USA"
    ]
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=65,
    value=30
)

gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)

exercise_level = st.selectbox(
    "Exercise Level",
    [
        "Low",
        "Moderate",
        "High"
    ]
)

diet_type = st.selectbox(
    "Diet Type",
    [
        "Balanced",
        "Vegetarian",
        "Vegan",
        "Junk Food",
        "Keto"
    ]
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
    [
        "Low",
        "Moderate",
        "High"
    ]
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


# Prediction section
if st.button(
    "Predict Happiness Score",
    type="primary",
    use_container_width=True
):

    try:
        with st.spinner(
            "Analysing your lifestyle information..."
        ):

            # Create engineered features used during training
            sleep_deviation = abs(
                sleep_hours - 8
            )

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

            # Create one-row DataFrame
            input_data = pd.DataFrame({
                "Country": [country],
                "Age": [age],
                "Gender": [gender],
                "Exercise Level": [exercise_level],
                "Diet Type": [diet_type],
                "Sleep Hours": [sleep_hours],
                "Stress Level": [stress_level],
                "Mental Health Condition": [
                    mental_health_condition
                ],
                "Work Hours per Week": [
                    work_hours
                ],
                "Screen Time per Day (Hours)": [
                    screen_time
                ],
                "Social Interaction Score": [
                    social_interaction
                ],
                "Sleep Deviation": [
                    sleep_deviation
                ],
                "Work Screen Load": [
                    work_screen_load
                ],
                "Active Social Score": [
                    active_social_score
                ]
            })

            # Generate prediction
            prediction = model.predict(
                input_data
            )[0]

            # Keep displayed score within dataset range
            prediction = max(
                1.0,
                min(10.0, prediction)
            )

        st.success(
            "Prediction completed successfully."
        )

        st.metric(
            label="Estimated Happiness Score",
            value=f"{prediction:.2f} / 10"
        )

        if prediction < 4:
            st.warning(
                "The predicted score is relatively low. "
                "Consider reviewing factors such as sleep, "
                "stress, exercise and social interaction."
            )

        elif prediction < 7:
            st.info(
                "The predicted score is moderate. "
                "Small lifestyle improvements may support "
                "better general well-being."
            )

        else:
            st.success(
                "The predicted score is relatively high. "
                "Continue maintaining balanced lifestyle habits."
            )

        with st.expander(
            "View Submitted Information"
        ):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "This result should be interpreted cautiously because "
            "the model has limited predictive performance."
        )

    except Exception as error:
        st.error(
            "The prediction could not be completed. "
            "Please check your inputs and try again."
        )

        with st.expander(
            "Technical Details"
        ):
            st.write(error)


# Footer
st.divider()

st.caption(
    "MindBalance Analytics | "
    "Machine Learning for Developers Project"
)