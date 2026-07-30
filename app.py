import os

import joblib
import pandas as pd
import streamlit as st


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="StudentWell Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# 2. FILE PATHS
# ==================================================
MODEL_PATH = "depression_prediction_model.pkl"
PREPROCESSOR_PATH = "depression_preprocessor.pkl"

LOGO_PATH = os.path.join(
    "images",
    "logo.png"
)

BANNER_PATH = os.path.join(
    "images",
    "student_wellness_banner.png"
)

ILLUSTRATION_PATH = os.path.join(
    "images",
    "student_support_illustration.png"
)


# ==================================================
# 3. LOAD MODEL AND PREPROCESSOR
# ==================================================
@st.cache_resource
def load_prediction_files():
    """
    Load the trained model and fitted preprocessing object.
    """

    loaded_model = joblib.load(MODEL_PATH)

    loaded_preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    return loaded_model, loaded_preprocessor


try:
    model, preprocessor = load_prediction_files()

except FileNotFoundError as error:
    st.error(
        "The model or preprocessor could not be found. "
        "Make sure both .pkl files are in the same folder as app.py."
    )

    with st.expander("Technical details"):
        st.write(error)

    st.stop()

except Exception as error:
    st.error(
        "The prediction files could not be loaded. "
        "Check that they were saved correctly from the notebook."
    )

    with st.expander("Technical details"):
        st.write(error)

    st.stop()


# ==================================================
# 4. CUSTOM CSS
# ==================================================
st.markdown(
    """
<style>

.stApp {
    background-color: #f7f4fb;
}

.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1,
h2,
h3 {
    color: #4b1d72;
}

div[data-testid="stForm"] {
    background-color: white;
    padding: 27px;
    border-radius: 20px;
    border: 1px solid #e3d8ec;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e3d8ec;
    padding: 17px;
    border-radius: 15px;
}

div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    min-height: 49px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(
        135deg,
        #55247b,
        #9450ad
    );
    color: white;
    font-size: 16px;
    font-weight: 700;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(
        135deg,
        #431c61,
        #793d91
    );
    color: white;
    border: none;
}

</style>
""",
    unsafe_allow_html=True
)


# ==================================================
# 5. HEADER
# ==================================================
header_logo, header_text = st.columns(
    [0.13, 0.87],
    vertical_alignment="center"
)

with header_logo:
    if os.path.exists(LOGO_PATH):
        st.image(
            LOGO_PATH,
            use_container_width=True
        )
    else:
        st.markdown("# 🧠")

with header_text:
    st.title("StudentWell Analytics")

    st.write(
        "AI-powered student-wellness screening for early insights, "
        "better support and stronger student outcomes."
    )


# ==================================================
# 6. BANNER IMAGE
# ==================================================
if os.path.exists(BANNER_PATH):
    st.image(
        BANNER_PATH,
        use_container_width=True
    )

else:
    st.warning(
        "Banner image missing. Place "
        "`student_wellness_banner.png` inside the images folder."
    )


# ==================================================
# 7. DISCLAIMER
# ==================================================
st.info(
    "Important notice: This application provides an estimated "
    "risk classification produced by a student machine learning "
    "project. It does not diagnose depression and must not replace "
    "assessment by a qualified mental-health professional. Results "
    "should only support further screening and student-wellness "
    "discussions."
)


# ==================================================
# 8. MAIN PAGE COLUMNS
# ==================================================
input_column, result_column = st.columns(
    [1.15, 0.85],
    gap="large"
)


# ==================================================
# 9. STUDENT INPUT FORM
# ==================================================
with input_column:

    st.header("Student Information")

    st.write(
        "Enter the student's details below. "
        "All fields are required."
    )

    with st.form("student_wellness_form"):

        # ------------------------------------------
        # Personal information
        # ------------------------------------------
        st.subheader("👤 Personal Information")

        personal_left, personal_right = st.columns(2)

        with personal_left:
            gender = st.selectbox(
                "Gender",
                options=[
                    "Female",
                    "Male"
                ]
            )

        with personal_right:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=34,
                value=21,
                step=1,
                help=(
                    "The training dataset contains students "
                    "between 18 and 34 years old."
                )
            )

        st.divider()

        # ------------------------------------------
        # Academic information
        # ------------------------------------------
        st.subheader("📚 Academic Information")

        academic_left, academic_right = st.columns(2)

        with academic_left:
            academic_pressure = st.slider(
                "Academic Pressure",
                min_value=1,
                max_value=5,
                value=3,
                help=(
                    "1 means very low academic pressure and "
                    "5 means very high academic pressure."
                )
            )

            study_hours = st.slider(
                "Study Hours Per Day",
                min_value=0.0,
                max_value=12.0,
                value=6.0,
                step=0.5
            )

        with academic_right:
            study_satisfaction = st.slider(
                "Study Satisfaction",
                min_value=1,
                max_value=5,
                value=3,
                help=(
                    "1 means very low study satisfaction and "
                    "5 means very high study satisfaction."
                )
            )

            financial_stress = st.slider(
                "Financial Stress",
                min_value=1,
                max_value=5,
                value=3,
                help=(
                    "1 means very low financial stress and "
                    "5 means very high financial stress."
                )
            )

        st.divider()

        # ------------------------------------------
        # Lifestyle information
        # ------------------------------------------
        st.subheader("🌙 Lifestyle Information")

        lifestyle_left, lifestyle_right = st.columns(2)

        with lifestyle_left:
            sleep_duration = st.selectbox(
                "Sleep Duration",
                options=[
                    "Less than 5 hours",
                    "5-6 hours",
                    "7-8 hours",
                    "More than 8 hours"
                ]
            )

        with lifestyle_right:
            dietary_habits = st.selectbox(
                "Dietary Habits",
                options=[
                    "Healthy",
                    "Moderate",
                    "Unhealthy"
                ]
            )

        st.divider()

        # ------------------------------------------
        # Personal and family history
        # ------------------------------------------
        st.subheader("💬 Personal and Family History")

        history_left, history_right = st.columns(2)

        with history_left:
            suicidal_thoughts = st.selectbox(
                "Have you ever had suicidal thoughts?",
                options=[
                    "No",
                    "Yes"
                ],
                help=(
                    "This is a sensitive field and should only "
                    "be collected with appropriate privacy and "
                    "safeguarding procedures."
                )
            )

        with history_right:
            family_history = st.selectbox(
                "Family History of Mental Illness",
                options=[
                    "No",
                    "Yes"
                ]
            )

        st.write("")

        submitted = st.form_submit_button(
            "🧠 Estimate Student Risk"
        )


# ==================================================
# 10. RESULT PANEL BEFORE SUBMISSION
# ==================================================
with result_column:

    st.header("Prediction Result")

    if not submitted:

        if os.path.exists(ILLUSTRATION_PATH):
            st.image(
                ILLUSTRATION_PATH,
                use_container_width=True
            )

        else:
            st.warning(
                "Illustration missing. Place "
                "`student_support_illustration.png` "
                "inside the images folder."
            )

        st.subheader("How the screening works")

        st.write(
            "Complete the form and select "
            "**Estimate Student Risk**."
        )

        st.write(
            "1. The application recreates the engineered features."
        )

        st.write(
            "2. The saved preprocessing and One-Hot Encoding are applied."
        )

        st.write(
            "3. The trained Random Forest generates a prediction."
        )

        st.write(
            "4. The predicted class and probabilities are displayed."
        )


# ==================================================
# 11. GENERATE PREDICTION
# ==================================================
if submitted:

    try:
        # ------------------------------------------
        # Feature engineering
        # ------------------------------------------
        total_stress_score = (
            academic_pressure
            + financial_stress
        )

        pressure_satisfaction_gap = (
            academic_pressure
            - study_satisfaction
        )

        if study_hours <= 4:
            study_load_category = "Low"

        elif study_hours <= 8:
            study_load_category = "Moderate"

        else:
            study_load_category = "High"


        # ------------------------------------------
        # Create the input DataFrame
        # ------------------------------------------
        input_data = pd.DataFrame(
            {
                "Gender": [gender],
                "Age": [float(age)],
                "Academic Pressure": [
                    float(academic_pressure)
                ],
                "Study Satisfaction": [
                    float(study_satisfaction)
                ],
                "Sleep Duration": [
                    sleep_duration
                ],
                "Dietary Habits": [
                    dietary_habits
                ],
                "Suicidal Thoughts": [
                    suicidal_thoughts
                ],
                "Study Hours": [
                    float(study_hours)
                ],
                "Financial Stress": [
                    float(financial_stress)
                ],
                "Family History of Mental Illness": [
                    family_history
                ],
                "Total Stress Score": [
                    float(total_stress_score)
                ],
                "Pressure Satisfaction Gap": [
                    float(pressure_satisfaction_gap)
                ],
                "Study Load Category": [
                    study_load_category
                ]
            }
        )


        # ------------------------------------------
        # Apply saved preprocessing
        # ------------------------------------------
        encoded_input_array = preprocessor.transform(
            input_data
        )

        encoded_feature_names = (
            preprocessor.get_feature_names_out()
        )

        encoded_input = pd.DataFrame(
            encoded_input_array,
            columns=encoded_feature_names
        )


        # ------------------------------------------
        # Generate model prediction
        # ------------------------------------------
        predicted_class = int(
            model.predict(encoded_input)[0]
        )

        class_probabilities = (
            model.predict_proba(encoded_input)[0]
        )

        lower_risk_probability = float(
            class_probabilities[0]
        )

        at_risk_probability = float(
            class_probabilities[1]
        )

        probability_percentage = (
            at_risk_probability * 100
        )


        # ------------------------------------------
        # Display prediction result
        # ------------------------------------------
        with result_column:

            if predicted_class == 1:

                st.error(
                    "Further Screening Recommended"
                )

                st.subheader("At Risk")

                st.metric(
                    "Estimated At-Risk Probability",
                    f"{probability_percentage:.1f}%"
                )

                st.warning(
                    "This result is not a diagnosis. Further "
                    "discussion or screening by an appropriate "
                    "professional may be beneficial."
                )

            else:

                st.success(
                    "Lower Estimated Risk"
                )

                st.subheader("Lower Risk")

                st.metric(
                    "Estimated At-Risk Probability",
                    f"{probability_percentage:.1f}%"
                )

                st.info(
                    "A lower-risk prediction does not guarantee "
                    "that a student is not experiencing distress. "
                    "Students should still seek support whenever "
                    "they need it."
                )


            # --------------------------------------
            # Probability section
            # --------------------------------------
            st.subheader("At-Risk Probability")

            st.progress(
                min(
                    max(
                        at_risk_probability,
                        0.0
                    ),
                    1.0
                )
            )

            probability_left, probability_right = st.columns(2)

            with probability_left:
                st.metric(
                    "Lower-Risk Probability",
                    f"{lower_risk_probability * 100:.1f}%"
                )

            with probability_right:
                st.metric(
                    "At-Risk Probability",
                    f"{at_risk_probability * 100:.1f}%"
                )


            # --------------------------------------
            # Indicator summary
            # --------------------------------------
            st.subheader("Student Indicator Summary")

            metric_one, metric_two = st.columns(2)

            with metric_one:
                st.metric(
                    "Total Stress Score",
                    f"{total_stress_score:.0f} / 10"
                )

            with metric_two:
                st.metric(
                    "Pressure-Satisfaction Gap",
                    f"{pressure_satisfaction_gap:+.0f}"
                )

            metric_three, metric_four = st.columns(2)

            with metric_three:
                st.metric(
                    "Study Load",
                    study_load_category
                )

            with metric_four:
                st.metric(
                    "Study Hours",
                    f"{study_hours:.1f} hours"
                )


            # --------------------------------------
            # Screening notes
            # --------------------------------------
            st.subheader("Screening Notes")

            note_displayed = False

            if suicidal_thoughts == "Yes":
                note_displayed = True

                st.error(
                    "Immediate human follow-up is important. "
                    "A reported history of suicidal thoughts is a "
                    "serious indicator regardless of the model "
                    "prediction. The student should be supported "
                    "through established professional and "
                    "safeguarding procedures."
                )

            if academic_pressure >= 4:
                note_displayed = True

                st.warning(
                    "High academic pressure: Academic guidance, "
                    "workload planning or counselling may be useful."
                )

            if financial_stress >= 4:
                note_displayed = True

                st.warning(
                    "High financial stress: Financial-aid information, "
                    "budgeting support or suitable student services "
                    "may be relevant."
                )

            if study_satisfaction <= 2:
                note_displayed = True

                st.warning(
                    "Low study satisfaction: Academic mentoring or a "
                    "discussion about the student's learning experience "
                    "may help identify concerns."
                )

            if sleep_duration == "Less than 5 hours":
                note_displayed = True

                st.warning(
                    "Limited sleep: Sleep habits may be worth discussing "
                    "as part of a broader student-wellness conversation."
                )

            if not note_displayed:
                st.success(
                    "No additional screening notes were triggered by "
                    "the submitted values."
                )


        # ------------------------------------------
        # Submitted information
        # ------------------------------------------
        st.divider()

        st.header("Submitted Information")

        display_data = pd.DataFrame(
            {
                "Indicator": [
                    "Gender",
                    "Age",
                    "Academic Pressure",
                    "Study Satisfaction",
                    "Sleep Duration",
                    "Dietary Habits",
                    "Suicidal Thoughts",
                    "Study Hours",
                    "Financial Stress",
                    "Family History",
                    "Total Stress Score",
                    "Pressure-Satisfaction Gap",
                    "Study Load Category"
                ],
                "Submitted Value": [
                    gender,
                    int(age),
                    academic_pressure,
                    study_satisfaction,
                    sleep_duration,
                    dietary_habits,
                    suicidal_thoughts,
                    f"{study_hours:.1f}",
                    financial_stress,
                    family_history,
                    total_stress_score,
                    pressure_satisfaction_gap,
                    study_load_category
                ]
            }
        )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


        # ------------------------------------------
        # Prediction interpretation
        # ------------------------------------------
        with st.expander(
            "How should this prediction be interpreted?"
        ):

            st.write(
                "The model estimates the probability of the class "
                "labelled as depression in the training dataset."
            )

            st.write(
                "A higher probability means that the submitted "
                "combination of indicators is more similar to "
                "records labelled Yes."
            )

            st.write(
                "The result cannot confirm whether a student has "
                "depression. A qualified person should consider the "
                "output together with a confidential conversation "
                "and other relevant information."
            )

            st.write(
                "The model was developed using 502 records. Its "
                "performance may change when used with students "
                "from other institutions or backgrounds."
            )


        # ------------------------------------------
        # Model information
        # ------------------------------------------
        with st.expander(
            "Model and preprocessing information"
        ):

            st.write(
                "**Final model:** Tuned Random Forest Classifier"
            )

            st.write(
                "**Model file:** "
                "`depression_prediction_model.pkl`"
            )

            st.write(
                "**Preprocessor file:** "
                "`depression_preprocessor.pkl`"
            )

            st.write(
                "**Engineered features:** Total Stress Score, "
                "Pressure Satisfaction Gap and Study Load Category"
            )

            st.write(
                "**Model output:** Lower Risk or At Risk, "
                "together with estimated probabilities"
            )


    except ValueError as error:
        st.error(
            "The input format does not match the trained model."
        )

        with st.expander("Technical details"):
            st.write(error)

    except Exception as error:
        st.error(
            "An unexpected error occurred while generating "
            "the prediction."
        )

        with st.expander("Technical details"):
            st.write(error)


# ==================================================
# 12. FOOTER
# ==================================================
st.divider()

st.caption(
    "StudentWell Analytics — Machine Learning Student-Wellness "
    "Screening Prototype"
)

st.caption(
    "This application is intended for educational demonstration "
    "and early-screening support only. It does not provide medical "
    "advice or a diagnosis."
)