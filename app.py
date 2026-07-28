import os

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="MindBalance Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# File paths
# --------------------------------------------------
MODEL_PATH = "happiness_prediction_model.pkl"
BRAIN_ICON_PATH = os.path.join("images", "brain_icon.png")
BANNER_PATH = os.path.join("images", "wellness_banner.png")
MEDITATION_IMAGE_PATH = os.path.join(
    "images",
    "meditation_illustration.png"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "The trained model file could not be found. Make sure "
        "'happiness_prediction_model.pkl' is in the same folder as app.py."
    )
    st.stop()

except Exception as error:
    st.error("The trained model could not be loaded.")

    with st.expander("Technical details"):
        st.write(error)

    st.stop()


# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8f5fc;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-header {
        padding: 28px 32px;
        border-radius: 22px;
        background: linear-gradient(135deg, #4b1d72, #9b59b6);
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(75, 29, 114, 0.18);
    }

    .main-header h1 {
        margin: 0;
        font-size: 40px;
    }

    .main-header p {
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 17px;
        opacity: 0.95;
    }

    .section-heading {
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 24px;
        font-weight: 700;
        color: #4b1d72;
    }

    div[data-testid="stForm"] {
        background-color: white;
        padding: 26px;
        border-radius: 20px;
        border: 1px solid #e4d9ef;
        box-shadow: 0 5px 16px rgba(0, 0, 0, 0.04);
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e4d9ef;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }

    .result-box {
        background: linear-gradient(135deg, #f4ebfb, #ffffff);
        border-left: 7px solid #8e44ad;
        padding: 24px;
        border-radius: 18px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 5px 16px rgba(0, 0, 0, 0.05);
    }

    .result-label {
        color: #6b6470;
        font-size: 16px;
        margin-bottom: 6px;
    }

    .result-score {
        color: #6c3483;
        font-size: 46px;
        font-weight: 800;
    }

    .image-card {
        background-color: white;
        padding: 14px;
        border-radius: 18px;
        border: 1px solid #e4d9ef;
        box-shadow: 0 5px 16px rgba(0, 0, 0, 0.04);
    }

    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 46px;
    }

    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }

    .footer-text {
        text-align: center;
        color: #756d7c;
        font-size: 14px;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------
header_left, header_right = st.columns(
    [1, 6],
    vertical_alignment="center"
)

with header_left:
    if os.path.exists(BRAIN_ICON_PATH):
        st.image(
            BRAIN_ICON_PATH,
            width=110
        )
    else:
        st.markdown("## 🧠")

with header_right:
    st.markdown(
        """
        <div class="main-header">
            <h1>MindBalance Analytics</h1>
            <p>
                Discover how your demographic and lifestyle patterns may
                relate to your estimated happiness score.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Banner
# --------------------------------------------------
if os.path.exists(BANNER_PATH):
    st.image(
        BANNER_PATH,
        use_container_width=True
    )

st.info(
    "This prediction is intended for general wellness awareness only. "
    "It is not medical advice and should not be used as a mental-health diagnosis."
)


# --------------------------------------------------
# Introduction section
# --------------------------------------------------
intro_left, intro_right = st.columns(
    [3, 1],
    vertical_alignment="center"
)

with intro_left:
    st.markdown(
        """
        <div class="section-heading">
            Your Lifestyle Snapshot
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "Complete the form below using your current lifestyle information. "
        "The machine learning model will estimate your Happiness Score on a "
        "scale from 1 to 10."
    )

with intro_right:
    if os.path.exists(MEDITATION_IMAGE_PATH):
        st.image(
            MEDITATION_IMAGE_PATH,
            use_container_width=True
        )


# --------------------------------------------------
# User input form
# --------------------------------------------------
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
        "🧠 Predict Happiness Score",
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
                    Your Estimated Happiness Score
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

        st.markdown(
            """
            <div class="section-heading">
                Lifestyle Summary
            </div>
            """,
            unsafe_allow_html=True
        )

        summary_one, summary_two, summary_three, summary_four = st.columns(4)

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
                value=f"{social_interaction:.1f}/10"
            )

        with summary_four:
            st.metric(
                label="Work Hours",
                value=f"{work_hours}/week"
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

st.markdown(
    """
    <div class="footer-text">
        MindBalance Analytics |
        Machine Learning for Developers Project
    </div>
    """,
    unsafe_allow_html=True
)