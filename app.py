import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
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


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="MindBalance Analytics",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f4fb;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, #4b1d72, #9b59b6);
        color: white;
        margin-bottom: 24px;
    }

    .main-title h1 {
        margin: 0;
        font-size: 40px;
    }

    .main-title p {
        margin-top: 10px;
        margin-bottom: 0;
        font-size: 17px;
    }

    .section-card {
        background-color: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #e7dff0;
        margin-bottom: 20px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #ded2ea;
        padding: 20px;
        border-radius: 16px;
    }

    div[data-testid="stForm"] {
        background-color: white;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #e7dff0;
    }

    .result-box {
        background-color: white;
        border-left: 6px solid #7d3c98;
        padding: 22px;
        border-radius: 14px;
        margin-top: 18px;
    }

    .result-label {
        color: #6b6470;
        font-size: 15px;
        margin-bottom: 4px;
    }

    .result-score {
        font-size: 42px;
        font-weight: bold;
        color: #6c3483;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: bold;
        padding: 12px;
    }

    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }

    .stSlider {
        padding-top: 5px;
        padding-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="main-title">
        <h1>🧠 MindBalance Analytics</h1>
        <p>
            Discover how your demographic and lifestyle patterns may relate
            to your estimated happiness score.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "This prediction is intended for general wellness awareness only. "
    "It is not medical advice and should not be used as a mental-health diagnosis."
)


# --------------------------------------------------
# Input form
# --------------------------------------------------
st.subheader("Enter Your Details")

with st.form("prediction_form"):

    left_column, right_column = st.columns(2)

    with left_column:

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

    with right_column:

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

    submitted = st.form_submit_button(
        "Predict Happiness Score",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:

    try:
        with st.spinner(
            "Analysing your lifestyle information..."
        ):

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

            prediction = model.predict(
                input_data
            )[0]

            prediction = max(
                1.0,
                min(10.0, prediction)
            )

        st.success(
            "Prediction completed successfully."
        )

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">
                    Estimated Happiness Score
                </div>
                <div class="result-score">
                    {prediction:.2f} / 10
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            int(prediction * 10)
        )

        summary_one, summary_two, summary_three = st.columns(3)

        with summary_one:
            st.metric(
                label="Sleep Hours",
                value=f"{sleep_hours:.1f}"
            )

        with summary_two:
            st.metric(
                label="Stress Level",
                value=stress_level
            )

        with summary_three:
            st.metric(
                label="Social Interaction",
                value=f"{social_interaction:.1f} / 10"
            )

        if prediction < 4:
            st.warning(
                "The predicted score is relatively low. Consider reviewing "
                "factors such as sleep, stress, exercise and social interaction."
            )

        elif prediction < 7:
            st.info(
                "The predicted score is moderate. Small improvements to "
                "daily lifestyle habits may support better general well-being."
            )

        else:
            st.success(
                "The predicted score is relatively high. Continue maintaining "
                "balanced lifestyle habits."
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
            "This result should be interpreted cautiously because the "
            "model has limited predictive performance."
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


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "MindBalance Analytics | "
    "Machine Learning for Developers Project"
)