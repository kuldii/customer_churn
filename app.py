import joblib
import numpy as np
import pandas as pd
import gradio as gr

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

def predict_telco_churn_blocks(selected_model, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
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
    # Ensure all expected columns are present
    for col in numerical_cols:
        if col not in input_df:
            input_df[col] = 0
    for col in categorical_cols:
        if col not in input_df:
            input_df[col] = 'No'
    # Impute missing values
    input_df[numerical_cols] = num_imputer.transform(input_df[numerical_cols])
    input_df[categorical_cols] = cat_imputer.transform(input_df[categorical_cols])
    # One-hot encode categoricals
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    # Add missing columns (from training)
    for col in X_columns:
        if col not in input_encoded:
            input_encoded[col] = 0
    input_encoded = input_encoded[X_columns]
    # Scale numerics
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])
    # Predict
    model = model_map[selected_model]
    try:
        proba = model.predict_proba(input_encoded)[0, 1] if hasattr(model, 'predict_proba') else model.decision_function(input_encoded)[0]
        pred = model.predict(input_encoded)[0]
    except Exception as e:
        return f"Error: {str(e)}"
    pred_text = 'Churn' if pred == 1 else 'No Churn'
    return f"🔮 Churn Probability: {proba:.2%}\n\n✅ Prediction: {pred_text}"

with gr.Blocks() as demo:
    gr.Markdown("""
    <h2>📊 Telco Customer Churn Prediction</h2>
    <p style="font-size: 16px;">
    Built with Scikit-learn, XGBoost & Gradio — by Kuldii Project
    </p>
    <p style="font-size: 14px;">
    This app predicts the probability of customer churn for a Telco company based on:<br>
    💁‍♂️ Demographic features<br>
    📞 Service details<br>
    💵 Billing information<br>
    <br>
    Select your desired model below and fill in customer details, then hit Predict!
    </p>
    """)
    
    model_choice = gr.Dropdown(
        choices=model_names,
        label="✨ Select Model",
        value=model_names[0]
    )

    with gr.Row():
        gender = gr.Dropdown(['Female', 'Male'], label='👤 Gender', value='Female')
        SeniorCitizen = gr.Dropdown(['No', 'Yes'], label='🎂 Senior Citizen', value='No')
        Partner = gr.Dropdown(['No', 'Yes'], label='❤️ Has Partner', value='No')
        Dependents = gr.Dropdown(['No', 'Yes'], label='👶 Has Dependents', value='No')

    with gr.Row():
        tenure = gr.Number(label='⏳ Tenure (months)', value=12, minimum=0, maximum=80)
        PhoneService = gr.Dropdown(['No', 'Yes'], label='📞 Phone Service', value='Yes')
        MultipleLines = gr.Dropdown(['No phone service', 'No', 'Yes'], label='📱 Multiple Lines', value='No')
        InternetService = gr.Dropdown(['DSL', 'Fiber optic', 'No'], label='🌐 Internet Service', value='DSL')

    with gr.Row():
        Contract = gr.Dropdown(['Month-to-month', 'One year', 'Two year'], label='📝 Contract', value='Month-to-month')
        PaperlessBilling = gr.Dropdown(['No', 'Yes'], label='🧾 Paperless Billing', value='Yes')
        PaymentMethod = gr.Dropdown([
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], label='💳 Payment Method', value='Electronic check')
        MonthlyCharges = gr.Number(label='💲 Monthly Charges', value=70, minimum=0)
        TotalCharges = gr.Number(label='💰 Total Charges', value=1400, minimum=0)

    predict_btn = gr.Button("🚀 Predict Churn")
    output = gr.Textbox(label="🔎 Prediction Result")

    predict_btn.click(
        fn=predict_telco_churn_blocks,
        inputs=[
            model_choice, gender, SeniorCitizen, Partner, Dependents,
            tenure, PhoneService, MultipleLines, InternetService,
            Contract, PaperlessBilling, PaymentMethod,
            MonthlyCharges, TotalCharges
        ],
        outputs=output
    )

demo.launch(server_name="0.0.0.0", root_path="/customer_churn", server_port=9002)