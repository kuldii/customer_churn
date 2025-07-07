# Telco Customer Churn Classification App

A professional, production-ready machine learning app for predicting customer churn using the real Telco Customer Churn dataset from Kaggle. Built with robust preprocessing, rich EDA, multiple classification models, and a modern Gradio UI. Fully containerized for easy deployment.

---

## 🚀 Features

- **Robust Preprocessing**: Missing value handling, categorical encoding, and feature scaling
- **Rich EDA**: Churn distribution, demographic analysis, contract/tenure/charges visualizations, correlation heatmap
- **Multiple Classification Models**: Logistic Regression, Random Forest, XGBoost, SVM (with hyperparameter tuning)
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, ROC curves
- **Interactive Gradio UI**: User-friendly inputs, model selection, and instant prediction
- **Production-Ready**: Dockerized, reproducible environment, and easy deployment

---

## 🏗️ Project Structure

```
├── app.py                        # Gradio app for prediction (production-ready)
├── customer_churn_classification.ipynb  # Full EDA, modeling, and training notebook
├── models/
│   └── churn_models.pkl          # Trained classification models and preprocessors (joblib)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Containerization for deployment
└── README.md                     # Project documentation
```

---

## 📊 Data & Preprocessing

- **Dataset**: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Preprocessing**:
  - Drop `customerID`, convert `TotalCharges` to numeric
  - Impute missing values (median for numerics, mode for categoricals)
  - One-hot encoding for categorical features
  - Standardization of numeric features

---

## 🧠 Models

- **Logistic Regression** (with GridSearchCV)
- **Random Forest Classifier** (with RandomizedSearchCV)
- **XGBoost Classifier** (with RandomizedSearchCV)
- **Support Vector Machine (SVM)** (with GridSearchCV)

All models are trained, tuned, and saved for instant prediction in the app.

---

## 🖥️ Gradio App

- **Dropdowns** and **sliders** for all features
- **Model selection** dropdown
- **Prediction output**: Churn probability and class
- **Production config**: Ready for local or Docker deployment

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/kuldii/customer_churn.git
cd customer_churn
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Train Models
- All models and preprocessors are pre-trained and saved in `models/`.
- To retrain, use the notebook `customer_churn_classification.ipynb` and re-export the models.

### 4. Run the App
```bash
python app.py
```
- The app will be available at `http://localhost:9002` by default.

---

## 🐳 Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t customer-churn .
```

### 2. Run the Container
```bash
docker run -p 9002:9002 customer-churn
```
- Access the app at `http://localhost:9002`

---

## 🖥️ Usage

1. Open the app in your browser.
2. Input customer features (demographics, services, billing, etc).
3. Select a classification model.
4. Click **Predict Churn** to get the probability and prediction.

---

## 📊 Visualizations & EDA
- See `customer_churn_classification.ipynb` for:
  - Churn distribution
  - Demographic and contract analysis
  - Tenure and charges visualizations
  - Correlation heatmap
  - Model comparison and evaluation

---

## 📝 Model Details
- **Preprocessing**: StandardScaler, SimpleImputer, OneHotEncoder
- **Models**: LogisticRegression, RandomForestClassifier, XGBClassifier, SVC (with hyperparameter tuning)
- **Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, ROC curves

---

## 📁 File Descriptions
- `app.py`: Gradio app, loads models, handles prediction and UI.
- `models/churn_models.pkl`: Dictionary of trained classification models and preprocessors.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Containerization instructions.
- `customer_churn_classification.ipynb`: Full EDA, preprocessing, model training, and export.

---

## 🌐 Demo & Credits
- **Author**: Sandikha Rahardi (Kuldii Project)
- **Website**: https://kuldiiproject.com
- **Dataset**: [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **UI**: [Gradio](https://gradio.app/)
- **ML**: [Scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)

---

For questions or contributions, please open an issue or pull request.
