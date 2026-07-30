import os

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="StudentWell Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# File paths
# --------------------------------------------------
MODEL_PATH = "depression_prediction_model.pkl"
PREPROCESSOR_PATH = "depression_preprocessor.pkl"

BANNER_PATH = os.path.join(
    "images",
    "student_wellness_banner.png"
)

ILLUSTRATION_PATH = os.path.join(
    "images",
    "student_support_illustration.png"
)


# --------------------------------------------------
# Load trained model and preprocessor
# --------------------------------------------------
@st.cache_resource
def load_prediction_files():
    """
    Load the trained Random Forest classification model
    and the fitted preprocessing object.
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
        "The trained model or preprocessor could not be found. "
        "Make sure 'depression_prediction_model.pkl' and "
        "'depression_preprocessor.pkl' are in the same folder "
        "as app.py."
    )

    with st.expander("Technical details"):
        st.write(error)

    st.stop()

except Exception as error:
    st.error(
        "The prediction files could not be loaded. "
        "Please check that they were exported correctly "
        "from the notebook."
    )

    with st.expander("Technical details"):
        st.write(error)

    st.stop()


# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>

    /* Main page background */
    .stApp {
        background-color: #f8f5fc;
    }

    /* Main page width */
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main header */
    .main-header {
        padding: 30px 34px;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            #4b1d72,
            #8e44ad,
            #b06cc7
        );
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 26px
            rgba(75, 29, 114, 0.20);
    }

    .main-header h1 {
        margin: 0;
        font-size: 42px;
        font-weight: 750;
    }

    .main-header p {
        margin-top: 10px;
        margin-bottom: 0;
        font-size: 17px;
        line-height: 1.6;
        opacity: 0.96;
    }

    /* Section heading */
    .section-heading {
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 24px;
        font-weight: 750;
        color: #4b1d72;
    }

    /* Introduction card */
    .information-card {
        background-color: white;
        padding: 22px 24px;
        border-radius: 18px;
        border: 1px solid #e5dbee;
        box-shadow: 0 5px 16px
            rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
        line-height: 1.65;
    }

    /* Streamlit form */
    div[data-testid="stForm"] {
        background-color: white;
        padding: 27px;
        border-radius: 20px;
        border: 1px solid #e4d9ef;
        box-shadow: 0 6px 18px
            rgba(0, 0, 0, 0.05);
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e4d9ef;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 4px 12px
            rgba(0, 0, 0, 0.04);
    }

    /* Lower-risk result */
    .lower-risk-box {
        background: linear-gradient(
            135deg,
            #eefaf3,
            #ffffff
        );
        border-left: 7px solid #2e9d62;
        padding: 25px;
        border-radius: 18px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 6px 18px
            rgba(46, 157, 98, 0.10);
    }

    .lower-risk-box h2 {
        color: #23784c;
        margin-top: 0;
        margin-bottom: 8px;
    }

    /* At-risk result */
    .at-risk-box {
        background: linear-gradient(
            135deg,
            #fff1f4,
            #ffffff
        );
        border-left: 7px solid #c84665;
        padding: 25px;
        border-radius: 18px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 6px 18px
            rgba(200, 70, 101, 0.12);
    }

    .at-risk-box h2 {
        color: #a72d4b;
        margin-top: 0;
        margin-bottom: 8px;
    }

    /* Probability result */
    .probability-box {
        background-color: white;
        border: 1px solid #e4d9ef;
        padding: 18px;
        border-radius: 16px;
        margin-top: 15px;
        text-align: center;
    }

    .probability-number {
        font-size: 36px;
        font-weight: 800;
        color: #6f2c91;
        margin: 4px 0;
    }

    /* Warning card */
    .warning-card {
        background-color: #fff8e7;
        border-left: 6px solid #e0a21b;
        padding: 18px 20px;
        border-radius: 14px;
        margin-top: 18px;
        line-height: 1.6;
    }

    /* Emergency card */
    .urgent-card {
        background-color: #fff0f2;
        border-left: 6px solid #c0392b;
        padding: 18px 20px;
        border-radius: 14px;
        margin-top: 18px;
        line-height: 1.6;
    }

    /* Explanation cards */
    .explanation-card {
        background-color: white;
        border: 1px solid #e4d9ef;
        padding: 18px 20px;
        border-radius: 15px;
        margin-top: 14px;
        line-height: 1.6;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 38px;
        padding-top: 20px;
        border-top: 1px solid #ded4e7;
        color: #6d6174;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Button */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(
            135deg,
            #5e2a84,
            #9b59b6
        );
        color: white;
        font-size: 16px;
        font-weight: 700;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(
            135deg,
            #4b1d72,
            #85459d
        );
        color: white;
        border: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Main heading
# --------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🧠 StudentWell Analytics</h1>
        <p>
            A machine learning student-wellness screening prototype
            that estimates whether a student may require further
            mental-health support based on academic, lifestyle,
            financial and personal factors.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Optional banner image
# --------------------------------------------------
if os.path.exists(BANNER_PATH):
    st.image(
        BANNER_PATH,
        use_container_width=True
    )


# --------------------------------------------------
# Important disclaimer
# --------------------------------------------------
st.markdown(
    """
    <div class="information-card">
        <strong>Important notice:</strong>
        This application provides an estimated risk classification
        generated by a student machine learning project. It does not
        diagnose depression and must not replace assessment by a
        qualified mental-health professional. The result should only
        be used to support further screening and student-wellness
        discussions.
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Page columns
# --------------------------------------------------
input_column, result_column = st.columns(
    [1.15, 0.85],
    gap="large"
)


# --------------------------------------------------
# Input form
# --------------------------------------------------
with input_column:

    st.markdown(
        '<div class="section-heading">'
        'Student Information'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the student's information below. "
        "All fields are required."
    )

    with st.form(
        "student_wellness_form"
    ):

        # ------------------------------------------
        # Personal information
        # ------------------------------------------
        st.subheader(
            "👤 Personal Information"
        )

        personal_left, personal_right = st.columns(2)

        with personal_left:
            gender = st.selectbox(
                "Gender",
                options=[
                    "Female",
                    "Male"
                ],
                help=(
                    "Select the gender category used "
                    "in the training dataset."
                )
            )

        with personal_right:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=34,
                value=21,
                step=1,
                help=(
                    "The training dataset contains "
                    "students aged between 18 and 34."
                )
            )


        st.divider()


        # ------------------------------------------
        # Academic information
        # ------------------------------------------
        st.subheader(
            "📚 Academic Information"
        )

        academic_left, academic_right = st.columns(2)

        with academic_left:
            academic_pressure = st.slider(
                "Academic Pressure",
                min_value=1,
                max_value=5,
                value=3,
                help=(
                    "1 represents very low pressure "
                    "and 5 represents very high pressure."
                )
            )

            study_hours = st.slider(
                "Study Hours Per Day",
                min_value=0.0,
                max_value=12.0,
                value=6.0,
                step=0.5,
                help=(
                    "Enter the approximate number of "
                    "hours spent studying each day."
                )
            )

        with academic_right:
            study_satisfaction = st.slider(
                "Study Satisfaction",
                min_value=1,
                max_value=5,
                value=3,
                help=(
                    "1 represents very low satisfaction "
                    "and 5 represents very high satisfaction."
                )
            )

            financial_stress = st.slider(
                "Financial Stress",
                min_value=1,
                max_value=5,
                value=3,
                help=(
                    "1 represents very low financial stress "
                    "and 5 represents very high financial stress."
                )
            )


        st.divider()


        # ------------------------------------------
        # Lifestyle information
        # ------------------------------------------
        st.subheader(
            "🌙 Lifestyle Information"
        )

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
        st.subheader(
            "💬 Personal and Family History"
        )

        history_left, history_right = st.columns(2)

        with history_left:
            suicidal_thoughts = st.selectbox(
                "Have you ever had suicidal thoughts?",
                options=[
                    "No",
                    "Yes"
                ],
                help=(
                    "This is a sensitive field and should "
                    "only be collected with appropriate "
                    "privacy and safeguarding procedures."
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


# --------------------------------------------------
# Result panel before submission
# --------------------------------------------------
with result_column:

    st.markdown(
        '<div class="section-heading">'
        'Prediction Result'
        '</div>',
        unsafe_allow_html=True
    )

    if not submitted:

        if os.path.exists(ILLUSTRATION_PATH):
            st.image(
                ILLUSTRATION_PATH,
                use_container_width=True
            )

        st.markdown(
            """
            <div class="information-card">
                <h3 style="color:#4b1d72; margin-top:0;">
                    How the screening works
                </h3>

                <p>
                    Complete the form and select
                    <strong>Estimate Student Risk</strong>.
                </p>

                <p>
                    The application will:
                </p>

                <ol>
                    <li>Create the same engineered features used during training.</li>
                    <li>Apply the saved preprocessing and One-Hot Encoding.</li>
                    <li>Use the tuned Random Forest model to estimate the class.</li>
                    <li>Display the predicted probability and a summary of the input.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:

    try:
        # ------------------------------------------
        # Recreate engineered features
        # ------------------------------------------

        total_stress_score = (
            academic_pressure
            + financial_stress
        )

        pressure_satisfaction_gap = (
            academic_pressure
            - study_satisfaction
        )

        # Must match the pd.cut boundaries
        # used inside the notebook.
        if study_hours <= 4:
            study_load_category = "Low"

        elif study_hours <= 8:
            study_load_category = "Moderate"

        else:
            study_load_category = "High"


        # ------------------------------------------
        # Create raw input DataFrame
        # Column names must match X in the notebook.
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
        encoded_input_array = (
            preprocessor.transform(input_data)
        )

        encoded_feature_names = (
            preprocessor.get_feature_names_out()
        )

        encoded_input = pd.DataFrame(
            encoded_input_array,
            columns=encoded_feature_names
        )


        # ------------------------------------------
        # Generate prediction and probability
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
        # Display result
        # ------------------------------------------
        with result_column:

            if predicted_class == 1:
                st.markdown(
                    f"""
                    <div class="at-risk-box">
                        <h2>Further Screening Recommended</h2>

                        <p>
                            The model classified this input as
                            <strong>At Risk</strong>.
                        </p>

                        <div class="probability-number">
                            {probability_percentage:.1f}%
                        </div>

                        <p>
                            Estimated probability of the
                            model's at-risk class
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.warning(
                    "This result is not a diagnosis. It indicates "
                    "that further discussion or screening by an "
                    "appropriate professional may be beneficial."
                )

            else:
                st.markdown(
                    f"""
                    <div class="lower-risk-box">
                        <h2>Lower Estimated Risk</h2>

                        <p>
                            The model classified this input as
                            <strong>Lower Risk</strong>.
                        </p>

                        <div class="probability-number">
                            {probability_percentage:.1f}%
                        </div>

                        <p>
                            Estimated probability of the
                            model's at-risk class
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.info(
                    "A lower-risk prediction does not guarantee "
                    "that a student is not experiencing distress. "
                    "Students should still be encouraged to seek "
                    "support whenever they need it."
                )


            # --------------------------------------
            # Probability progress bar
            # --------------------------------------
            st.markdown(
                "#### At-Risk Probability"
            )

            st.progress(
                min(
                    max(
                        at_risk_probability,
                        0.0
                    ),
                    1.0
                )
            )

            probability_left, probability_right = (
                st.columns(2)
            )

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
            # Input summary metrics
            # --------------------------------------
            st.markdown(
                "#### Student Indicator Summary"
            )

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
            # Context-based user messages
            # These are explanatory messages only.
            # They do not replace the model output.
            # --------------------------------------
            st.markdown(
                "#### Screening Notes"
            )

            if suicidal_thoughts == "Yes":
                st.markdown(
                    """
                    <div class="urgent-card">
                        <strong>Immediate human follow-up is important.</strong>
                        A reported history of suicidal thoughts is a
                        serious indicator regardless of the machine
                        learning prediction. The student should be
                        supported through the institution's established
                        safeguarding and professional-support procedures.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if academic_pressure >= 4:
                st.markdown(
                    """
                    <div class="warning-card">
                        <strong>High academic pressure:</strong>
                        The entered academic-pressure score is high.
                        Academic guidance, workload planning or counselling
                        may be useful areas for follow-up.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if financial_stress >= 4:
                st.markdown(
                    """
                    <div class="warning-card">
                        <strong>High financial stress:</strong>
                        Consider whether financial-aid information,
                        budgeting support or suitable student services
                        may be relevant.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if study_satisfaction <= 2:
                st.markdown(
                    """
                    <div class="warning-card">
                        <strong>Low study satisfaction:</strong>
                        Academic mentoring or a discussion about the
                        student's learning experience may help identify
                        possible concerns.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if (
                sleep_duration
                == "Less than 5 hours"
            ):
                st.markdown(
                    """
                    <div class="warning-card">
                        <strong>Limited sleep:</strong>
                        The student reported fewer than five hours of
                        sleep. Sleep habits may be worth discussing as
                        part of a broader wellness conversation.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ------------------------------------------
        # Full submitted-information table
        # ------------------------------------------
        st.markdown("---")

        st.markdown(
            '<div class="section-heading">'
            'Submitted Information'
            '</div>',
            unsafe_allow_html=True
        )

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
        # Explanation
        # ------------------------------------------
        with st.expander(
            "How should this prediction be interpreted?"
        ):
            st.write(
                """
                The model estimates the probability of the class
                labelled as depression in the training dataset. A
                higher probability means that the submitted combination
                of academic, lifestyle, financial and personal factors
                resembles more of the records labelled Yes.
                """
            )

            st.write(
                """
                The prediction does not explain the student's full
                circumstances and cannot confirm whether the student
                has depression. A qualified person should consider
                the result together with a confidential conversation
                and other relevant information.
                """
            )

            st.write(
                """
                The model was developed using 502 records. Its
                performance may change when used with students from
                different institutions, age groups or backgrounds.
                """
            )


        # ------------------------------------------
        # Show model information
        # ------------------------------------------
        with st.expander(
            "Model and preprocessing information"
        ):
            st.write(
                "**Final model:** Tuned Random Forest Classifier"
            )

            st.write(
                "**Saved model file:** "
                "`depression_prediction_model.pkl`"
            )

            st.write(
                "**Saved preprocessing file:** "
                "`depression_preprocessor.pkl`"
            )

            st.write(
                "**Original input features:** 13"
            )

            st.write(
                "**Engineered features:** "
                "Total Stress Score, Pressure Satisfaction Gap "
                "and Study Load Category"
            )

            st.write(
                "**Model output:** "
                "Lower Risk or At Risk, together with "
                "estimated class probabilities"
            )


    except ValueError as error:
        st.error(
            "The input could not be processed because its "
            "format does not match the trained model."
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


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <strong>StudentWell Analytics</strong><br>
        Machine Learning Student-Wellness Screening Prototype<br><br>

        This application is intended for educational demonstration
        and early-screening support only. It does not provide medical
        advice or a diagnosis. Predictions must be reviewed responsibly
        by suitable human personnel.
    </div>
    """,
    unsafe_allow_html=True
)