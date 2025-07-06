import os
import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Google Drive file ID for churn_models.pkl
MODELS_FILE_ID = "1pDPW7TFXPZTYql02xslnIN-ja2GObd-X"  # Updated file ID from user link

# Download churn_models.pkl from Google Drive if not present
os.makedirs("models", exist_ok=True)
if not os.path.exists("models/churn_models.pkl"):
    url = f"https://drive.google.com/uc?id={MODELS_FILE_ID}"
    gdown.download(url, "models/churn_models.pkl", quiet=False)

# Load all models and preprocessors
artifacts = joblib.load("models/churn_models.pkl")

model_names = [
    'Logistic Regression',
    'Random Forest',
    'XGBoost',
    'SVM'
]

model_map = {
    'Logistic Regression': artifacts['best_logreg'],
    'Random Forest': artifacts['best_rf'],
    'XGBoost': artifacts['best_xgb'],
    'SVM': artifacts['best_svm']
}
scaler = artifacts['scaler']
num_imputer = artifacts['num_imputer']
cat_imputer = artifacts['cat_imputer']
categorical_cols = artifacts['categorical_cols']
numerical_cols = artifacts['numerical_cols']
X_columns = artifacts['X_columns']

def predict_telco_churn(selected_model, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
    input_dict = {
        'gender': gender,
        'SeniorCitizen': 1 if SeniorCitizen == 'Yes' else 0,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    input_df = pd.DataFrame([input_dict])
    for col in numerical_cols:
        if col not in input_df:
            input_df[col] = 0
    for col in categorical_cols:
        if col not in input_df:
            input_df[col] = 'No'
    input_df[numerical_cols] = num_imputer.transform(input_df[numerical_cols])
    input_df[categorical_cols] = cat_imputer.transform(input_df[categorical_cols])
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    for col in X_columns:
        if col not in input_encoded:
            input_encoded[col] = 0
    input_encoded = input_encoded[X_columns]
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])
    model = model_map[selected_model]
    try:
        proba = model.predict_proba(input_encoded)[0, 1] if hasattr(model, 'predict_proba') else model.decision_function(input_encoded)[0]
        pred = model.predict(input_encoded)[0]
    except Exception as e:
        return None, f"Error: {str(e)}"
    pred_text = 'Churn' if pred == 1 else 'No Churn'
    return proba, pred_text

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    layout="centered"
)

st.title("📊 Telco Customer Churn Prediction")

st.markdown("""
<p style="font-size:16px">
Built with <b>Scikit-learn</b>, <b>XGBoost</b> & <b>Streamlit</b> — by Kuldii Project
</p>

<p style="font-size:14px">
This app predicts the probability of customer churn for a Telco company based on:<br>
💁‍♂️ Demographic features<br>
📞 Service details<br>
💵 Billing information<br><br>
Select your desired model below and fill in customer details, then hit Predict!
</p>
""", unsafe_allow_html=True)

with st.form("churn_form"):
    st.subheader("Customer Information")

    model_choice = st.selectbox("✨ Select Model", model_names)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.selectbox('👤 Gender', ['Female', 'Male'])
    with col2:
        SeniorCitizen = st.selectbox('🎂 Senior Citizen', ['No', 'Yes'])
    with col3:
        Partner = st.selectbox('❤️ Has Partner', ['No', 'Yes'])
    with col4:
        Dependents = st.selectbox('👶 Has Dependents', ['No', 'Yes'])

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        tenure = st.number_input('⏳ Tenure (months)', min_value=0, max_value=80, value=12)
    with col6:
        PhoneService = st.selectbox('📞 Phone Service', ['No', 'Yes'])
    with col7:
        MultipleLines = st.selectbox('📱 Multiple Lines', ['No phone service', 'No', 'Yes'])
    with col8:
        InternetService = st.selectbox('🌐 Internet Service', ['DSL', 'Fiber optic', 'No'])

    col9, col10, col11, col12 = st.columns(4)
    with col9:
        Contract = st.selectbox('📝 Contract', ['Month-to-month', 'One year', 'Two year'])
    with col10:
        PaperlessBilling = st.selectbox('🧾 Paperless Billing', ['No', 'Yes'])
    with col11:
        PaymentMethod = st.selectbox('💳 Payment Method', [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
    with col12:
        MonthlyCharges = st.number_input('💲 Monthly Charges', min_value=0.0, value=70.0)

    TotalCharges = st.number_input('💰 Total Charges', min_value=0.0, value=1400.0)

    submitted = st.form_submit_button("🚀 Predict Churn")

if submitted:
    proba, pred_text = predict_telco_churn(
        model_choice, gender, SeniorCitizen, Partner, Dependents,
        tenure, PhoneService, MultipleLines, InternetService,
        Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges
    )
    if proba is not None:
        st.success(f"🔮 Churn Probability: {proba:.2%}\n\n✅ Prediction: {pred_text}")
    else:
        st.error(pred_text)
