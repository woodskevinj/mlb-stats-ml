⚾ MLB Player Stats ML Project

⚾ End-to-end machine learning pipeline predicting MLB player performance — featuring data preprocessing, model training, SHAP explainability, and Flask API deployment.

💼 Portfolio Summary

This project demonstrates my ability to design, build, and deploy end-to-end machine learning solutions using modern MLOps principles. It showcases a practical understanding of the full lifecycle — from data preprocessing and model development to explainability and API deployment — aligning directly with the work of an Applied ML Engineer.

🧠 Overview

This project predicts Major League Baseball player performance metrics (such as OBP — On-Base Percentage) using a machine learning model built from real-world player statistics.

It demonstrates the full Applied Machine Learning Engineering workflow — from exploratory data analysis (EDA) through model explainability (SHAP) and API deployment with Flask.

This project is part of a 30-day sprint to become interview-ready for an Applied ML Engineer role.

🚀 Project Goals

Build an end-to-end ML pipeline using modern Python tools.

Learn and demonstrate:

Data preprocessing & feature engineering

Model training, evaluation, and interpretability

Model serving through an API (Flask)

Reproducible workflows & environment management

🧩 Tech Stack
Category Tools
Language Python 3.10
Data & ML pandas, numpy, scikit-learn
Visualization matplotlib, seaborn
Explainability SHAP
API / Serving Flask
Environment virtualenv
Version Control Git + GitHub
Future Deployment Docker + AWS (EC2 / Lambda)
📂 Project Structure
mlb-stats-ml/
│
├── app/
│ └── mlb_predict.py # Flask API for model inference
│
├── data/
│ ├── mlb_players_18_clean.csv # Cleaned dataset
│ └── (raw CSV ignored via .gitignore)
│
├── models/
│ └── mlb_salary_model.joblib # Trained model artifact
│
├── notebooks/
│ └── mlb_eda.ipynb # Exploratory & modeling notebook
│
├── requirements.txt # Dependencies
├── .gitignore # Ignore venvs, data, etc.
└── README.md # Project documentation

⚙️ Setup & Installation
1️⃣ Clone the repository
git clone git@github.com:woodskevinj/mlb-stats-ml.git
cd mlb-stats-ml

2️⃣ Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Launch Jupyter Notebook
jupyter notebook

5️⃣ Run the Flask API locally
cd app
python3 mlb_predict.py

🧮 Model Overview

Target Variable: On-Base Percentage (OBP)

Features Used: Player stats including games, at-bats, home runs, walks, strikeouts, stolen bases, and more.

Model Type: Linear Regression inside a scikit-learn Pipeline

Preprocessing: Scaling (StandardScaler) + One-Hot Encoding (for categorical variables)

📊 Model Interpretation

Feature Importance: Extracted via model coefficients and SHAP summary plots.

Explainability Tools: SHAP values reveal which player stats most influence predictions.

Per-Player Analysis: Waterfall plots show how each feature impacts an individual player’s predicted OBP.

Example:

shap.summary_plot(shap_values, X_preprocessed, plot_type="bar")
shap.waterfall_plot(shap_values[0])

🌐 API Usage Example

Run locally:

curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"games": 120, "AB": 410, "R": 68, "H": 125, "doubles": 20, "triples": 2, "HR": 14, "RBI": 55, "walks": 40, "strike_outs": 90, "stolen_bases": 8, "caught_stealing_base": 3, "AVG": 0.305, "SLG": 0.420, "OPS": 0.725, "position": "OF"}'

Sample Output:

{
"predicted_OBP": 0.305
}

🧱 Current Progress

✅ Data Cleaning & EDA

✅ Feature Engineering & Model Training

✅ Evaluation & Explainability (SHAP)

✅ Flask Model API

🔜 Containerization (Docker)

🔜 Cloud Deployment (AWS Lambda / EC2)

💡 Next Steps

Add Dockerfile for containerized serving

Deploy API to AWS (Lambda or EC2)

Set up CI/CD with GitHub Actions

Build Streamlit dashboard for interactive demos

✍️ Citation

If referencing materials from Made With ML:

@article{madewithml,
author = {Goku Mohandas},
title = {Setup - Made With ML},
howpublished = {\url{https://madewithml.com/}},
year = {2023}
}
