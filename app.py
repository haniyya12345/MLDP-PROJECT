import os

import joblib
import pandas as pd
import streamlit as st


# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="StudentWell Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# FILE PATHS
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
# LOAD MODEL AND PREPROCESSOR
# ==================================================
@st.cache_resource
def load_prediction_files():
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
        "Check that they were exported correctly from the notebook."
    )

    with st.expander("Technical details"):
        st.write(error)

    st.stop()


# ==================================================
# CUSTOM CSS
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

.header-container {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 28px 32px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        #452067,
        #7b3fa0,
        #a967bd
    );
    color: white;
    margin-bottom: 20px;
}

.header-logo {
    width: 90px;
    height: 90px;
    object-fit: contain;
    border-radius: 18px;
}

.header-text h1 {
    margin: 0;
    font-size: 42px;
    font-weight: 750;
}

.header-text p {
    margin-top: 8px;
    margin-bottom: 0;
    font-size: 17px;
    line-height: 1.55;
}

.section-heading {
    margin-top: 8px;
    margin-bottom: 12px;
    font-size: 25px;
    font-weight: 750;
    color: #4b1d72;
}

.information-card {
    background-color: white;
    padding: 22px 24px;
    border-radius: 18px;
    border: 1px solid #e3d8ec;
    margin-bottom: 20px;
    line-height: 1.65;
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

.lower-risk-box {
    background-color: #eef9f2;
    border-left: 7px solid #2f965f;
    padding: 25px;
    border-radius: 18px;
    margin-top: 10px;
    text-align: center;
}

.lower-risk-box h2 {
    color: #24764a;
    margin-top: 0;
    margin-bottom: 8px;
}

.at-risk-box {
    background-color: #fff1f3;
    border-left: 7px solid #c74765;
    padding: 25px;
    border-radius: 18px;
    margin-top: 10px;
    text-align: center;
}

.at-risk-box h2 {
    color: #a42d49;
    margin-top: 0;
    margin-bottom: 8px;
}

.probability-number {
    font-size: 38px;
    font-weight: 800;
    color: #6f2c91;
    margin: 10px 0;
}

.warning-card {
    background-color: #fff8e7;
    border-left: 6px solid #dfa11c;
    padding: 18px 20px;
    border-radius: 14px;
    margin-top: 16px;
    line-height: 1.6;
}

.urgent-card {
    background-color: #fff0f2;
    border-left: 6px solid #bd342d;
    padding: 18px 20px;
    border-radius: 14px;
    margin-top: 16px;
    line-height: 1.6;
}

.footer {
    text-align: center;
    margin-top: 38px;
    padding-top: 20px;
    border-top: 1px solid #ddd3e6;
    color: #6d6174;
    font-size: 14px;
    line-height: 1.6;
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
# HEADER WITH LOGO
# ==================================================
if os.path.exists(LOGO_PATH):

    logo_column, title_column = st.columns(
        [0.15, 0.85],
        vertical_alignment="center"
    )

    with logo_column:
        st.image(
            LOGO_PATH,
            use_container_width=True
        )

    with title_column:
        st.markdown(
            """
<div class="header-container">
    <div class="header-text">
        <h1>StudentWell Analytics</h1>

        <p>
            AI-powered student-wellness screening for early insights,
            better support and stronger student outcomes.
        </p>
    </div>
</div>
""",
            unsafe_allow_html=True
        )

else:
    st.markdown(
        """
<div class="header-container">
    <div class="header-text">
        <h1>🧠 StudentWell Analytics</h1>

        <p>
            AI-powered student-wellness screening for early insights,
            better support and stronger student outcomes.
        </p>
    </div>
</div>
""",
        unsafe_allow_html=True
    )


# ==================================================
# BANNER IMAGE
# ==================================================
if os.path.exists(BANNER_PATH):
    st.image(
        BANNER_PATH,
        use_container_width=True
    )

else:
    st.warning(
        "Banner image missing. Place "
        "student_wellness_banner.png inside the images folder."
    )


# ==================================================
# DISCLAIMER
# ==================================================
st.markdown(
    """
<div class="information-card">
    <strong>Important notice:</strong>
    This application provides an estimated risk classification
    produced by a student machine learning project. It does not
    diagnose depression and must not replace assessment by a
    qualified mental-health professional. Results should only
    support further screening and student-wellness discussions.
</div>
""",
    unsafe_allow_html=True
)


# ==================================================
# MAIN COLUMNS
# ==================================================
input_column, result_column = st.columns(
    [1.15, 0.85],
    gap="large"
)


# ==================================================
# STUDENT INPUT FORM
# ==================================================
with input_column:

    st.markdown(
        """
<div class="section-heading">
    Student Information
</div>
""",
        unsafe_allow_html=True
    )

    st.write(
        "Enter the student's details below. "
        "All fields are required."
    )

    with st.form("student_wellness_form"):

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
                step=1
            )

        st.divider()

        st.subheader("📚 Academic Information")

        academic_left, academic_right = st.columns(2)

        with academic_left:
            academic_pressure = st.slider(
                "Academic Pressure",
                min_value=1,
                max_value=5,
                value=3,
                help=(
                    "1 means very low pressure and "
                    "5 means very high pressure."
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
                    "1 means very low satisfaction and "
                    "5 means very high satisfaction."
                )
            )

            financial_stress = st.slider(
                "Financial Stress",
                min_value=1,
                max_value=5,
                value=3,
                help=(
                    "1 means very low stress and "
                    "5 means very high stress."
                )
            )

        st.divider()

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

        st.subheader("💬 Personal and Family History")

        history_left, history_right = st.columns(2)

        with history_left:
            suicidal_thoughts = st.selectbox(
                "Have you ever had suicidal thoughts?",
                options=[
                    "No",
                    "Yes"
                ]
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
# RESULT PANEL BEFORE SUBMISSION
# ==================================================
with result_column:

    st.markdown(
        """
<div class="section-heading">
    Prediction Result
</div>
""",
        unsafe_allow_html=True
    )

    if not submitted:

        if os.path.exists(ILLUSTRATION_PATH):
            st.image(
                ILLUSTRATION_PATH,
                use_container_width=True
            )

        else:
            st.warning(
                "Illustration missing. Place "
                "student_support_illustration.png "
                "inside the images folder."
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

    <ol>
        <li>The engineered features are recreated.</li>
        <li>The saved preprocessing is applied.</li>
        <li>The trained Random Forest generates a prediction.</li>
        <li>The predicted class and probability are displayed.</li>
    </ol>
</div>
""",
            unsafe_allow_html=True
        )


# ==================================================
# GENERATE PREDICTION
# ==================================================
if submitted:

    try:
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
        Estimated probability of the model's at-risk class
    </p>
</div>
""",
                    unsafe_allow_html=True
                )

                st.warning(
                    "This result is not a diagnosis. "
                    "Further discussion or screening by an "
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
        Estimated probability of the model's at-risk class
    </p>
</div>
""",
                    unsafe_allow_html=True
                )

                st.info(
                    "A lower-risk prediction does not guarantee "
                    "that a student is not experiencing distress."
                )


            st.markdown("#### At-Risk Probability")

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


            st.markdown("#### Student Indicator Summary")

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


            st.markdown("#### Screening Notes")

            if suicidal_thoughts == "Yes":

                st.markdown(
                    """
<div class="urgent-card">
    <strong>Immediate human follow-up is important.</strong>
    A reported history of suicidal thoughts is a serious
    indicator regardless of the model prediction. The student
    should be supported through established professional and
    safeguarding procedures.
</div>
""",
                    unsafe_allow_html=True
                )

            if academic_pressure >= 4:

                st.markdown(
                    """
<div class="warning-card">
    <strong>High academic pressure:</strong>
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
    Financial-aid information, budgeting support or suitable
    student services may be relevant.
</div>
""",
                    unsafe_allow_html=True
                )

            if study_satisfaction <= 2:

                st.markdown(
                    """
<div class="warning-card">
    <strong>Low study satisfaction:</strong>
    Academic mentoring or a discussion about the student's
    learning experience may help identify concerns.
</div>
""",
                    unsafe_allow_html=True
                )

            if sleep_duration == "Less than 5 hours":

                st.markdown(
                    """
<div class="warning-card">
    <strong>Limited sleep:</strong>
    Sleep habits may be worth discussing as part of a broader
    student-wellness conversation.
</div>
""",
                    unsafe_allow_html=True
                )


        st.markdown("---")

        st.markdown(
            """
<div class="section-heading">
    Submitted Information
</div>
""",
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


        with st.expander(
            "How should this prediction be interpreted?"
        ):

            st.write(
                """
                The model estimates the probability of the class
                labelled as depression in the training dataset.
                A higher probability means that the submitted
                combination of indicators is more similar to
                records labelled Yes.
                """
            )

            st.write(
                """
                The result cannot confirm whether a student has
                depression. A qualified person should consider the
                output together with a confidential conversation
                and other relevant information.
                """
            )

            st.write(
                """
                The model was developed using 502 records.
                Its performance may change when used with students
                from other institutions or backgrounds.
                """
            )


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
# FOOTER
# ==================================================
st.markdown(
    """
<div class="footer">
    <strong>StudentWell Analytics</strong><br>
    Machine Learning Student-Wellness Screening Prototype<br><br>

    This application is intended for educational demonstration
    and early-screening support only. It does not provide medical
    advice or a diagnosis.
</div>
""",
    unsafe_allow_html=True
)