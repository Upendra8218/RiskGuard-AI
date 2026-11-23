import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RiskGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CREDIT_DATA_PATH = os.path.join(
    BASE_DIR, "data", "credit", "application_train.csv"
)

FRAUD_DATA_PATH = os.path.join(
    BASE_DIR, "data", "fraud", "creditcard.csv"
)

MODEL_FILES = {
    "credit_model": os.path.join(MODEL_DIR, "credit_logistic_model.pkl"),
    "credit_preprocessor": os.path.join(MODEL_DIR, "credit_preprocessor.pkl"),
    "fraud_model": os.path.join(MODEL_DIR, "fraud_logistic_model.pkl"),
    "fraud_scaler": os.path.join(MODEL_DIR, "fraud_scaler.pkl"),
    "fraud_threshold": os.path.join(MODEL_DIR, "fraud_threshold.pkl")
}

# ============================================================
# 3. PROFESSIONAL DASHBOARD DESIGN
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background: #151e35;
}

[data-testid="stSidebar"] * {
    color: #f8fafc;
}

h1, h2, h3 {
    color: #ffffff !important;
}

p, label {
    color: #cbd5e1;
}

.hero {
    background: linear-gradient(120deg, #172554, #1e40af, #075985);
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 38px;
}

.hero p {
    color: #dbeafe;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background: #192541;
    border: 1px solid #33456b;
    padding: 18px;
    border-radius: 13px;
}

div[data-testid="stMetric"] label {
    color: #a5b4fc;
}

div[data-testid="stMetricValue"] {
    color: white;
}

.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 9px;
    min-height: 45px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: #1d4ed8;
    color: white;
}

.card {
    background: #192541;
    border: 1px solid #33456b;
    padding: 22px;
    border-radius: 15px;
    min-height: 145px;
}

.card h3 {
    color: #93c5fd !important;
}

.card p {
    color: #cbd5e1;
}

.footer {
    text-align: center;
    color: #94a3b8;
    padding: 25px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

required_files = list(MODEL_FILES.values()) + [
    CREDIT_DATA_PATH,
    FRAUD_DATA_PATH
]

missing_files = [
    path for path in required_files
    if not os.path.exists(path)
]

if missing_files:
    st.error("The following required project files are missing:")
    for path in missing_files:
        st.write(path)
    st.stop()

# ============================================================
# 5. LOAD MODELS AND DATASETS
# ============================================================

@st.cache_resource
def load_models():

    credit_model = joblib.load(MODEL_FILES["credit_model"])
    credit_preprocessor = joblib.load(
        MODEL_FILES["credit_preprocessor"]
    )
    fraud_model = joblib.load(MODEL_FILES["fraud_model"])
    fraud_scaler = joblib.load(MODEL_FILES["fraud_scaler"])
    fraud_threshold = joblib.load(MODEL_FILES["fraud_threshold"])

    threshold = float(
        np.asarray(fraud_threshold).ravel()[0]
    )

    return (
        credit_model,
        credit_preprocessor,
        fraud_model,
        fraud_scaler,
        threshold
    )


@st.cache_data
def load_datasets():

    credit_df = pd.read_csv(
        CREDIT_DATA_PATH,
        low_memory=False
    )

    fraud_df = pd.read_csv(
        FRAUD_DATA_PATH
    )

    return credit_df, fraud_df


try:
    (
        credit_model,
        credit_preprocessor,
        fraud_model,
        fraud_scaler,
        fraud_threshold
    ) = load_models()

    credit_df, fraud_df = load_datasets()

except Exception as error:
    st.error("Unable to load models or datasets.")
    st.exception(error)
    st.stop()

# ============================================================
# 6. MODEL FEATURE COLUMNS
# ============================================================

if hasattr(credit_preprocessor, "feature_names_in_"):
    CREDIT_FEATURES = list(
        credit_preprocessor.feature_names_in_
    )
else:
    CREDIT_FEATURES = [
        c for c in credit_df.columns if c != "TARGET"
    ]

if hasattr(fraud_scaler, "feature_names_in_"):
    FRAUD_FEATURES = list(
        fraud_scaler.feature_names_in_
    )
else:
    FRAUD_FEATURES = [
        c for c in fraud_df.columns if c != "Class"
    ]

# ============================================================
# 7. INPUT PREPARATION
# ============================================================

def get_credit_baseline():

    baseline = credit_df.drop(
        columns=["TARGET"],
        errors="ignore"
    ).iloc[[0]].copy()

    baseline = baseline.reindex(
        columns=CREDIT_FEATURES
    )

    return baseline


def prepare_credit_input(input_data):

    if isinstance(input_data, pd.Series):
        input_data = input_data.to_frame().T

    prepared = get_credit_baseline()

    input_data = input_data.drop(
        columns=["TARGET"],
        errors="ignore"
    )

    for column in input_data.columns:

        if column in prepared.columns:

            prepared.loc[
                prepared.index[0], column
            ] = input_data.iloc[0][column]

    if "NUM_INSTALMENT" in prepared.columns:
        prepared["NUM_INSTALMENT"] = pd.to_numeric(
            prepared["NUM_INSTALMENT"],
            errors="coerce"
        )

    return prepared.reindex(
        columns=CREDIT_FEATURES
    )


def prepare_fraud_input(input_data):

    if isinstance(input_data, pd.Series):
        input_data = input_data.to_frame().T

    input_data = input_data.drop(
        columns=["Class"],
        errors="ignore"
    )

    return input_data.reindex(
        columns=FRAUD_FEATURES
    )


# ============================================================
# 8. PREDICTION FUNCTIONS
# ============================================================

def positive_probability(model, data):

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(data)
        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError("Positive class 1 is missing.")

        return float(
            probabilities[0][classes.index(1)]
        )

    if hasattr(model, "decision_function"):

        value = float(
            np.asarray(
                model.decision_function(data)
            ).ravel()[0]
        )

        value = np.clip(value, -500, 500)

        return float(
            1 / (1 + np.exp(-value))
        )

    raise ValueError("Model cannot calculate probabilities.")


def predict_credit(input_data):

    prepared = prepare_credit_input(input_data)

    transformed = credit_preprocessor.transform(
        prepared
    )

    return positive_probability(
        credit_model,
        transformed
    )


def predict_fraud(input_data):

    prepared = prepare_fraud_input(input_data)

    scaled = fraud_scaler.transform(
        prepared
    )

    return positive_probability(
        fraud_model,
        scaled
    )


# ============================================================
# 9. RISK CLASSIFICATION
# ============================================================

def credit_risk_band(probability):

    score = 100 * (1 - probability)

    if score >= 70:
        band = "LOW"
    elif score >= 40:
        band = "MEDIUM"
    else:
        band = "HIGH"

    return score, band


def fraud_risk_band(probability):

    if probability >= fraud_threshold:
        return "FRAUD REVIEW REQUIRED"
    elif probability >= 0.30:
        return "ELEVATED FRAUD RISK"
    else:
        return "LOW FRAUD SIGNAL"


def combined_decision(credit_probability, fraud_probability):

    score, credit_band = credit_risk_band(
        credit_probability
    )

    if fraud_probability >= fraud_threshold:

        return (
            "FRAUD REVIEW REQUIRED",
            "Fraud threshold reached. Manual investigation is recommended.",
            score,
            credit_band
        )

    if credit_band == "HIGH":

        return (
            "HIGH CREDIT RISK",
            "Higher estimated default risk. Additional assessment is recommended.",
            score,
            credit_band
        )

    if credit_band == "MEDIUM":

        return (
            "MANUAL REVIEW",
            "Additional application assessment is recommended.",
            score,
            credit_band
        )

    return (
        "LOW MODEL-ESTIMATED RISK",
        "Lower estimated default risk. This is not an automatic loan approval.",
        score,
        credit_band
    )


# ============================================================
# 10. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <h1 style="text-align:center;">🛡️ RiskGuard AI</h1>
    <p style="text-align:center;color:#93c5fd;">
    FINANCIAL RISK INTELLIGENCE
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Executive Dashboard",
        "Loan Application Form",
        "Credit Risk Analysis",
        "Fraud Detection",
        "Combined Risk Decision",
        "Project Information"
    ]
)

st.sidebar.divider()
st.sidebar.success("MODEL COMPONENTS LOADED")
st.sidebar.caption("Academic machine learning prototype")

# ============================================================
# 11. EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🛡️ RiskGuard AI</h1>
        <p>
        AI-Powered Loan Approval, Credit Risk & Fraud Detection System
        </p>
        <p>
        An integrated machine learning decision-support platform
        for credit default prediction and fraud detection.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.success("● SYSTEM ONLINE — ALL MODEL COMPONENTS LOADED")

    st.subheader("System Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Credit Applications",
        f"{len(credit_df):,}"
    )

    c2.metric(
        "Fraud Transactions",
        f"{len(fraud_df):,}"
    )

    c3.metric(
        "Credit Engine",
        "ACTIVE"
    )

    c4.metric(
        "Fraud Engine",
        "ACTIVE"
    )

    st.divider()

    st.subheader("Dataset Insights")

    left, right = st.columns(2)

    with left:

        st.markdown("### 🏦 Credit Dataset")

        credit_counts = credit_df["TARGET"].value_counts()
        credit_counts.index = credit_counts.index.map({
            0: "No Default",
            1: "Default"
        })

        st.bar_chart(credit_counts)

        st.metric(
            "Default Rate",
            f"{credit_df['TARGET'].mean() * 100:.2f}%"
        )

    with right:

        st.markdown("### 💳 Fraud Dataset")

        fraud_counts = fraud_df["Class"].value_counts()
        fraud_counts.index = fraud_counts.index.map({
            0: "Normal",
            1: "Fraud"
        })

        st.bar_chart(fraud_counts)

        st.metric(
            "Fraud Rate",
            f"{fraud_df['Class'].mean() * 100:.3f}%"
        )

    st.divider()

    st.subheader("Platform Capabilities")

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="card">
            <h3>🏦 Credit Risk Analysis</h3>
            <p>Estimates the probability of loan default and generates a risk score.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="card">
            <h3>💳 Fraud Detection</h3>
            <p>Evaluates anonymized transaction features to estimate fraud probability.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="card">
            <h3>🛡️ Combined Decision</h3>
            <p>Combines credit and fraud predictions into a review recommendation.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("System Architecture")

    st.info(
        "User Input → Data Preprocessing → Credit Model + Fraud Model "
        "→ Combined Risk Engine → Human Review"
    )

# ============================================================
# 12. LOAN APPLICATION FORM
# ============================================================

elif page == "Loan Application Form":

    st.title("🏦 Loan Application Form")

    st.write(
        "Enter applicant details to generate an educational credit-risk estimate."
    )

    st.info(
        "The trained Home Credit model uses many features. This form "
        "updates only supported fields; all other features retain "
        "reference-record values. This is not a complete bank assessment."
    )

    with st.form("loan_application_form"):

        # PERSONAL INFORMATION

        st.subheader("1. Personal Information")

        c1, c2 = st.columns(2)

        with c1:

            age = st.number_input(
                "Applicant Age",
                18, 100, 30
            )

            family_status = st.selectbox(
                "Family Status",
                [
                    "Single / not married",
                    "Married",
                    "Civil marriage",
                    "Widow",
                    "Separated",
                    "Unknown"
                ]
            )

            education = st.selectbox(
                "Education Level",
                [
                    "Secondary / secondary special",
                    "Higher education",
                    "Incomplete higher",
                    "Lower secondary",
                    "Academic degree"
                ]
            )

        with c2:

            housing_type = st.selectbox(
                "Housing Type",
                [
                    "House / apartment",
                    "Rented apartment",
                    "With parents",
                    "Municipal apartment",
                    "Office apartment",
                    "Co-op apartment"
                ]
            )

            children = st.number_input(
                "Number of Children",
                0, 20, 0
            )

            owns_car = st.selectbox(
                "Does applicant own a car?",
                ["No", "Yes"]
            )

            owns_property = st.selectbox(
                "Does applicant own property?",
                ["Yes", "No"]
            )

        st.divider()

        # EMPLOYMENT AND INCOME

        st.subheader("2. Employment and Income")

        c1, c2 = st.columns(2)

        with c1:

            employment = st.selectbox(
                "Employment Status",
                [
                    "Employed",
                    "Self-employed",
                    "Government Employee",
                    "Unemployed",
                    "Pensioner",
                    "Student"
                ]
            )

            years_employed = st.number_input(
                "Years of Employment",
                0, 50, 3
            )

        with c2:

            annual_income = st.number_input(
                "Annual Income (₹)",
                0, 100000000, 500000, step=10000
            )

            monthly_income = annual_income / 12

            st.metric(
                "Estimated Monthly Income",
                f"₹{monthly_income:,.2f}"
            )

        st.divider()

        # LOAN INFORMATION

        st.subheader("3. Loan Information")

        c1, c2 = st.columns(2)

        with c1:

            loan_amount = st.number_input(
                "Requested Loan Amount (₹)",
                10000, 100000000, 500000, step=10000
            )

            monthly_repayment = st.number_input(
                "Estimated Monthly Installment (₹)",
                0, 10000000, 15000, step=1000
            )

        with c2:

            goods_price = st.number_input(
                "Loan Purpose / Goods Value (₹)",
                0, 100000000, 500000, step=10000
            )

            contract_type = st.selectbox(
                "Loan Type",
                [
                    "Cash loans",
                    "Revolving loans"
                ]
            )

        st.divider()

        # CREDIT INFORMATION

        st.subheader("4. Credit Information")

        st.caption(
            "Enter credit-bureau enquiry counts if known."
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            bureau_hour = st.number_input(
                "Enquiries in last hour", 0, 100, 0
            )

            bureau_day = st.number_input(
                "Enquiries in last day", 0, 100, 0
            )

        with c2:

            bureau_week = st.number_input(
                "Enquiries in last week", 0, 100, 0
            )

            bureau_month = st.number_input(
                "Enquiries in last month", 0, 100, 0
            )

        with c3:

            bureau_quarter = st.number_input(
                "Enquiries in last 3 months", 0, 100, 0
            )

            bureau_year = st.number_input(
                "Enquiries in last year", 0, 100, 0
            )

        st.markdown("**External Credit-Risk Scores**")

        st.caption(
            "Optional Home Credit EXT_SOURCE scores. These are normalized "
            "values from 0 to 1, NOT CIBIL scores."
        )

        use_external = st.checkbox(
            "I have valid external credit-risk scores"
        )

        ext1 = ext2 = ext3 = None

        if use_external:

            e1, e2, e3 = st.columns(3)

            with e1:
                ext1 = st.number_input(
                    "External Score 1",
                    0.0, 1.0, 0.5, step=0.01
                )

            with e2:
                ext2 = st.number_input(
                    "External Score 2",
                    0.0, 1.0, 0.5, step=0.01
                )

            with e3:
                ext3 = st.number_input(
                    "External Score 3",
                    0.0, 1.0, 0.5, step=0.01
                )

        st.warning(
            "This dataset does not contain a conventional CIBIL score "
            "or complete existing-loan details. Do not enter invented values."
        )

        submitted = st.form_submit_button(
            "🔍 Analyze Loan Application"
        )

    if submitted:

        try:

            application = get_credit_baseline()

            income_mapping = {
                "Employed": "Working",
                "Self-employed": "Commercial associate",
                "Government Employee": "State servant",
                "Unemployed": "Unemployed",
                "Pensioner": "Pensioner",
                "Student": "Student"
            }

            inputs = {
                "DAYS_BIRTH": -int(age * 365),
                "CNT_CHILDREN": children,
                "NAME_FAMILY_STATUS": family_status,
                "NAME_EDUCATION_TYPE": education,
                "NAME_HOUSING_TYPE": housing_type,
                "FLAG_OWN_CAR": "Y" if owns_car == "Yes" else "N",
                "FLAG_OWN_REALTY": "Y" if owns_property == "Yes" else "N",
                "NAME_INCOME_TYPE": income_mapping[employment],
                "DAYS_EMPLOYED": (
                    365243 if employment == "Unemployed"
                    else -int(years_employed * 365)
                ),
                "AMT_INCOME_TOTAL": annual_income,
                "AMT_CREDIT": loan_amount,
                "AMT_ANNUITY": monthly_repayment,
                "AMT_GOODS_PRICE": goods_price,
                "NAME_CONTRACT_TYPE": contract_type,
                "AMT_REQ_CREDIT_BUREAU_HOUR": bureau_hour,
                "AMT_REQ_CREDIT_BUREAU_DAY": bureau_day,
                "AMT_REQ_CREDIT_BUREAU_WEEK": bureau_week,
                "AMT_REQ_CREDIT_BUREAU_MON": bureau_month,
                "AMT_REQ_CREDIT_BUREAU_QRT": bureau_quarter,
                "AMT_REQ_CREDIT_BUREAU_YEAR": bureau_year
            }

            if use_external:
                inputs.update({
                    "EXT_SOURCE_1": ext1,
                    "EXT_SOURCE_2": ext2,
                    "EXT_SOURCE_3": ext3
                })

            applied_fields = []

            for column, value in inputs.items():

                if column in application.columns:

                    application.loc[
                        application.index[0], column
                    ] = value

                    applied_fields.append(column)

            probability = predict_credit(application)

            score, band = credit_risk_band(probability)

            st.divider()
            st.subheader("📊 Loan Risk Assessment Result")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Default Probability",
                f"{probability * 100:.2f}%"
            )

            c2.metric(
                "Credit Risk Score",
                f"{score:.2f}/100"
            )

            c3.metric(
                "Risk Category",
                band
            )

            if band == "LOW":
                st.success("LOW MODEL-ESTIMATED RISK — Human review still required.")
            elif band == "MEDIUM":
                st.warning("MEDIUM MODEL-ESTIMATED RISK — Manual review recommended.")
            else:
                st.error("HIGH MODEL-ESTIMATED RISK — Additional assessment recommended.")

            st.subheader("Applicant Summary")

            summary = pd.DataFrame({
                "Information": [
                    "Age",
                    "Family Status",
                    "Education",
                    "Housing",
                    "Children",
                    "Employment",
                    "Years Employed",
                    "Annual Income",
                    "Requested Loan",
                    "Monthly Installment",
                    "Loan Type",
                    "Car Ownership",
                    "Property Ownership"
                ],
                "Value": [
                    age,
                    family_status,
                    education,
                    housing_type,
                    children,
                    employment,
                    years_employed,
                    f"₹{annual_income:,.0f}",
                    f"₹{loan_amount:,.0f}",
                    f"₹{monthly_repayment:,.0f}",
                    contract_type,
                    owns_car,
                    owns_property
                ]
            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Credit Information Summary")

            credit_summary = pd.DataFrame({
                "Credit Information": [
                    "Enquiries in last hour",
                    "Enquiries in last day",
                    "Enquiries in last week",
                    "Enquiries in last month",
                    "Enquiries in last 3 months",
                    "Enquiries in last year"
                ],
                "Value": [
                    bureau_hour,
                    bureau_day,
                    bureau_week,
                    bureau_month,
                    bureau_quarter,
                    bureau_year
                ]
            })

            st.dataframe(
                credit_summary,
                use_container_width=True,
                hide_index=True
            )

            if use_external:
                st.write(
                    f"External scores entered: {ext1:.2f}, "
                    f"{ext2:.2f}, {ext3:.2f}"
                )

            st.info(
                f"{len(applied_fields)} of {len(inputs)} submitted fields "
                "were found in the trained model's input columns. "
                "Other features use reference-record values."
            )

        except Exception as error:

            st.error("Loan prediction failed.")
            st.exception(error)

# ============================================================
# 13. CREDIT RISK ANALYSIS
# ============================================================

elif page == "Credit Risk Analysis":

    st.title("🏦 Credit Risk Analysis")

    st.write(
        "Analyze an application from the Home Credit dataset."
    )

    applicant_index = st.number_input(
        "Select Applicant Record",
        0,
        len(credit_df) - 1,
        min(10, len(credit_df) - 1)
    )

    applicant = credit_df.drop(
        columns=["TARGET"],
        errors="ignore"
    ).iloc[[int(applicant_index)]]

    with st.expander("View applicant data"):
        st.dataframe(applicant, use_container_width=True)

    if st.button("Analyze Credit Risk"):

        try:

            probability = predict_credit(applicant)
            score, band = credit_risk_band(probability)

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Default Probability",
                f"{probability * 100:.2f}%"
            )

            c2.metric(
                "Credit Score",
                f"{score:.2f}/100"
            )

            c3.metric(
                "Risk Category",
                band
            )

            st.progress(
                float(np.clip(probability, 0, 1)),
                text="Estimated default probability"
            )

        except Exception as error:
            st.error("Credit prediction failed.")
            st.exception(error)

# ============================================================
# 14. FRAUD DETECTION
# ============================================================

elif page == "Fraud Detection":

    st.title("💳 Credit Card Fraud Detection")

    st.info(
        "The dataset uses anonymized transaction features. "
        "It does not contain ordinary card numbers or merchant names."
    )

    transaction_index = st.number_input(
        "Select Transaction Record",
        0,
        len(fraud_df) - 1,
        0
    )

    transaction = fraud_df.drop(
        columns=["Class"],
        errors="ignore"
    ).iloc[[int(transaction_index)]].copy()

    if "Amount" in transaction.columns:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=float(transaction["Amount"].iloc[0])
        )

        transaction.loc[
            transaction.index[0], "Amount"
        ] = amount

    with st.expander("View transaction details"):
        st.dataframe(transaction, use_container_width=True)

    if st.button("Analyze Transaction"):

        try:

            probability = predict_fraud(transaction)
            category = fraud_risk_band(probability)

            c1, c2 = st.columns(2)

            c1.metric(
                "Fraud Probability",
                f"{probability * 100:.4f}%"
            )

            c2.metric(
                "Fraud Threshold",
                f"{fraud_threshold:.4f}"
            )

            if probability >= fraud_threshold:
                st.error("🚨 FRAUD REVIEW REQUIRED")
            elif probability >= 0.30:
                st.warning("⚠️ ELEVATED FRAUD RISK")
            else:
                st.success("✅ LOW FRAUD SIGNAL")

            st.write("Assessment:", category)

            st.progress(
                float(np.clip(probability, 0, 1)),
                text="Estimated fraud probability"
            )

        except Exception as error:
            st.error("Fraud prediction failed.")
            st.exception(error)

# ============================================================
# 15. COMBINED RISK DECISION
# ============================================================

elif page == "Combined Risk Decision":

    st.title("🛡️ Combined Risk Decision")

    st.write(
        "Combine credit-risk and fraud-detection predictions."
    )

    c1, c2 = st.columns(2)

    with c1:

        applicant_index = st.number_input(
            "Applicant Record",
            0,
            len(credit_df) - 1,
            min(10, len(credit_df) - 1)
        )

    with c2:

        transaction_index = st.number_input(
            "Transaction Record",
            0,
            len(fraud_df) - 1,
            min(10, len(fraud_df) - 1)
        )

    st.caption(
        "The selected applicant and transaction are independent demo records."
    )

    if st.button("Generate Combined Decision"):

        try:

            applicant = credit_df.drop(
                columns=["TARGET"],
                errors="ignore"
            ).iloc[[int(applicant_index)]]

            transaction = fraud_df.drop(
                columns=["Class"],
                errors="ignore"
            ).iloc[[int(transaction_index)]]

            credit_probability = predict_credit(applicant)
            fraud_probability = predict_fraud(transaction)

            (
                decision,
                explanation,
                score,
                credit_band
            ) = combined_decision(
                credit_probability,
                fraud_probability
            )

            fraud_band = fraud_risk_band(
                fraud_probability
            )

            st.divider()
            st.subheader("Risk Assessment Results")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Credit Risk Score",
                f"{score:.2f}/100"
            )

            c2.metric(
                "Default Probability",
                f"{credit_probability * 100:.2f}%"
            )

            c3.metric(
                "Fraud Probability",
                f"{fraud_probability * 100:.4f}%"
            )

            st.divider()
            st.subheader("Final Recommendation")

            if decision == "LOW MODEL-ESTIMATED RISK":
                st.success(decision)
            elif decision == "MANUAL REVIEW":
                st.warning(decision)
            else:
                st.error(decision)

            st.write(explanation)

            st.write("Credit Risk Category:", credit_band)
            st.write("Fraud Assessment:", fraud_band)

            st.info(
                "This is an academic decision-support result. "
                "It is not an automatic loan approval or rejection."
            )

        except Exception as error:
            st.error("Combined assessment failed.")
            st.exception(error)

# ============================================================
# 16. PROJECT INFORMATION
# ============================================================

elif page == "Project Information":

    st.title("📘 Project Information")

    st.markdown("""
    ### AI-Powered Loan Approval, Credit Risk & Fraud Detection System

    RiskGuard AI is an academic machine learning prototype that
    demonstrates credit default prediction, fraud detection and
    combined financial risk assessment.
    """)

    st.divider()

    st.subheader("Project Objectives")

    st.markdown("""
    - Predict the probability of credit default.
    - Generate an illustrative credit risk score.
    - Classify credit risk.
    - Detect potentially fraudulent transactions.
    - Handle imbalanced fraud data.
    - Combine credit and fraud model predictions.
    - Generate human-review recommendations.
    """)

    st.divider()

    st.subheader("Datasets Used")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        ### 🏦 Home Credit Default Risk

        - Credit default prediction
        - Target: `TARGET`
        - 0 = No default
        - 1 = Default
        - 307,511 original application records
        """)

    with c2:

        st.markdown("""
        ### 💳 Credit Card Fraud Detection

        - Transaction fraud detection
        - Target: `Class`
        - 0 = Normal
        - 1 = Fraud
        - 284,807 transaction records
        """)

    st.divider()

    st.subheader("Credit Model Performance")

    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Tuned Random Forest"
        ],
        "Accuracy": [
            0.6900, 0.6649, 0.6902
        ],
        "Precision": [
            0.1616, 0.1481, 0.1541
        ],
        "Recall": [
            0.6785, 0.6628, 0.6318
        ],
        "F1 Score": [
            0.2611, 0.2421, 0.2477
        ],
        "ROC-AUC": [
            0.7487, 0.7165, 0.7207
        ],
        "PR-AUC": [
            0.2276, 0.1921, 0.1931
        ]
    })

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "PR-AUC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "These are recorded credit-model evaluation results, "
        "not live predictions."
    )

    st.divider()

    st.subheader("Technology Stack")

    st.markdown("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - SMOTE / Imbalanced-learn during training
    - Joblib
    - Streamlit
    """)

    st.divider()

    st.subheader("Responsible AI Notice")

    st.warning(
        "This is an academic prototype. Predictions may be incorrect. "
        "Risk bands are illustrative and are not official banking standards. "
        "The application must not automatically approve or reject real loans. "
        "Human review is required."
    )

# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.markdown("""
<div class="footer">
    🛡️ <b>RiskGuard AI</b>
    <br>
    AI-Powered Loan Approval, Credit Risk & Fraud Detection System
    <br>
    Academic Machine Learning Project
</div>
""", unsafe_allow_html=True)
