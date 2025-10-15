⚾ MLB Player Stats ML Project

End-to-end machine learning system predicting MLB player performance metrics using scikit-learn, SHAP, and Flask. Includes full EDA, preprocessing, model training, explainability, and serving.

🧩 Project Overview

This repository contains a complete Applied ML Engineering workflow, from data ingestion and preprocessing to model serving.
It uses MLB player statistics to train a regression model that predicts On-Base Percentage (OBP) based on offensive performance metrics.

This project was developed as part of a 30-day MLOps upskilling sprint, emphasizing practical experience with reproducible and deployable ML systems.

🧱 Architecture
Data (.csv) → EDA & Cleaning → Feature Engineering →
Pipeline (Scaler + OneHot + Model) → Evaluation → Explainability (SHAP) → API (Flask)

🧠 Model Objective

Goal: Predict player OBP (on-base percentage) from season performance stats

Model Type: Linear Regression via scikit-learn Pipeline

Inputs: games, at-bats, hits, doubles, home runs, walks, strikeouts, etc.

Outputs: Continuous prediction (OBP)

🧰 Environment Setup
1️⃣ Clone and Create Virtual Environment
git clone git@github.com:woodskevinj/mlb-stats-ml.git
cd mlb-stats-ml
python3 -m venv venv
source venv/bin/activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Verify Installation
python --version
pip list

📁 Directory Layout
mlb-stats-ml/
├── app/
│ └── mlb_predict.py # Flask inference API
│
├── data/
│ ├── mlb_players_18_clean.csv # Cleaned dataset (local)
│ └── (raw CSV ignored via .gitignore)
│
├── models/
│ └── mlb_salary_model.joblib # Serialized trained model
│
├── notebooks/
│ └── mlb_eda.ipynb # EDA, preprocessing, model dev, SHAP
│
├── requirements.txt # Reproducible dependencies
├── .gitignore # Ignore data, venv, cache files
└── README.md # Documentation

🧮 Reproducible Training Workflow

EDA + Cleaning (mlb_eda.ipynb)

Validate schema, inspect for missing values, normalize numeric features.

Export clean CSV → data/mlb_players_18_clean.csv.

Feature Engineering

Apply StandardScaler to numeric columns.

One-Hot Encode position categorical column (handle_unknown='ignore').

Model Training

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from joblib import dump

pipeline = Pipeline([
('preprocessor', ColumnTransformer([
('num', StandardScaler(), numeric_cols),
('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])),
('regressor', LinearRegression())
])
pipeline.fit(X_train, y_train)
dump(pipeline, "models/mlb_salary_model.joblib")

Evaluation Metrics

MAE, MSE, RMSE, R²

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

📊 Model Explainability (SHAP)

Use SHAP to compute global and per-sample feature influence.

import shap

explainer = shap.Explainer(model.named_steps["regressor"], X_train)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")

Waterfall visualization for one player:

shap.waterfall_plot(shap_values[0])

🌐 Flask API Serving

Run locally:

cd app
python3 mlb_predict.py

Test prediction:

curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"games":120,"AB":410,"R":68,"H":125,"doubles":20,"triples":2,"HR":14,"RBI":55,"walks":40,"strike_outs":90,"stolen_bases":8,"caught_stealing_base":3,"AVG":0.305,"SLG":0.420,"OPS":0.725,"position":"OF"}'

Response:

{"predicted_OBP": 0.305}

🔍 Diagnostic Metrics Table
Metric Formula Description
MAE mean( yₙ - ŷₙ
RMSE sqrt(mean((yₙ - ŷₙ)²)) Penalizes large errors
R² 1 - (SS_res / SS_tot) Model variance explanation
🧠 Feature Importance via Coefficients
import pandas as pd

coeffs = model.named*steps["regressor"].coef*
importance = pd.DataFrame({
"Feature": X.columns,
"Weight": coeffs
}).sort_values("Weight", ascending=False)
print(importance.head(10))

🐳 Next Phase: Containerization

Planned Docker structure:

Dockerfile
│
├── FROM python:3.10-slim
├── COPY . /app
├── WORKDIR /app
├── RUN pip install -r requirements.txt
├── EXPOSE 5000
└── CMD ["python3", "app/mlb_predict.py"]

🧾 Reference & Credits
@article{madewithml,
author = {Goku Mohandas},
title = {Setup - Made With ML},
howpublished = {\url{https://madewithml.com/}},
year = {2023}
}

✅ Current Progress

EDA + Cleaning

Feature Engineering

Model Training + Evaluation

Explainability (SHAP)

Flask Deployment

Dockerization

AWS Deployment (EC2 / Lambda)
