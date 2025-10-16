---
# ⚾ MLB Player Stats ML Project

End-to-end machine learning system predicting MLB player performance metrics using scikit-learn, SHAP, and Flask. Includes full EDA, preprocessing, model training, explainability, containerization, and serving.
---

## 🧩 Project Overview

This repository contains a complete **Applied ML Engineering workflow**, from **data ingestion and preprocessing** to **model serving and container deployment**.
It uses MLB player statistics to train a regression model that predicts **On-Base Percentage (OBP)** from key offensive metrics.

Developed as part of a **30-day MLOps upskilling sprint**, emphasizing reproducibility, modular pipeline design, and real-world deployment readiness.

---

## 🧱 Architecture

```
Data (.csv) → EDA & Cleaning → Feature Engineering →
Pipeline (Scaler + OneHot + Model) → Evaluation → Explainability (SHAP) → API (Flask) → Docker
```

---

## 🧠 Model Objective

- **Goal:** Predict player OBP (on-base percentage)
- **Model Type:** Linear Regression (`scikit-learn` Pipeline)
- **Inputs:** games, at-bats, hits, doubles, home runs, walks, strikeouts, etc.
- **Outputs:** Continuous regression prediction (OBP)

---

## 🧰 Environment Setup

<details>
<summary><b>Show setup commands</b></summary>

```bash
# 1️⃣ Clone repository
git clone git@github.com:woodskevinj/mlb-stats-ml.git
cd mlb-stats-ml

# 2️⃣ Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt
```

</details>

---

## 📁 Directory Layout

```
mlb-stats-ml/
├── app/
│   └── mlb_predict.py            # Flask inference API
│
├── data/
│   ├── mlb_players_18_clean.csv  # Cleaned dataset
│
├── models/
│   └── mlb_salary_model.joblib   # Trained model artifact
│
├── notebooks/
│   └── mlb_eda.ipynb             # EDA, preprocessing, SHAP analysis
│
├── Dockerfile                    # Container definition
├── requirements.txt              # Dependencies
└── README.md
```

---

## 🧮 Reproducible Workflow

<details>
<summary><b>Data Cleaning, Modeling, and Evaluation</b></summary>

```python
# Clean missing values
df = df.dropna()
df.to_csv("data/mlb_players_18_clean.csv", index=False)

# Pipeline and training
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])),
    ('regressor', LinearRegression())
])
pipeline.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

</details>

---

## 🌐 Flask Model API

<details>
<summary><b>API Example</b></summary>

```python
from flask import Flask, request, jsonify
import joblib, pandas as pd

app = Flask(__name__)
model = joblib.load("../models/mlb_salary_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({"predicted_OBP": round(float(prediction), 3)})

if __name__ == "__main__":
    app.run(debug=True)
```

</details>

**Local test:**

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"games":120,"AB":410,"R":68,"H":125,"doubles":20,"HR":14,"walks":40,"OPS":0.725,"position":"OF"}'
```

---

## 🐳 Docker Deployment

### 🧩 Build the Image

```bash
docker build -t mlb-ml-app .
```

### ▶️ Run the Container

```bash
docker run -p 5050:5000 mlb-ml-app
```

Flask will start inside Docker:

```
* Running on http://0.0.0.0:5000 (Press CTRL+C to quit)
```

### 🧪 Test the Running Container

```bash
curl -X POST http://127.0.0.1:5050/predict \
-H "Content-Type: application/json" \
-d '{"games":120,"AB":410,"R":68,"H":125,"doubles":20,"HR":14,"RBI":55,"walks":40,"OPS":0.725,"position":"OF"}'
```

✅ Example output:

```json
{ "predicted_OBP": 0.305 }
```

---

## ☁️ Next Step (Coming Soon)

- Push container image to **AWS ECR**
- Deploy to **AWS ECS (Fargate)** as a production-ready ML endpoint
- Add CI/CD workflow (GitHub Actions)

---

## ✅ Current Progress

- [x] Data Cleaning + EDA
- [x] Model Training + Evaluation
- [x] SHAP Explainability
- [x] Flask API Deployment
- [x] Docker Containerization
- [ ] AWS Deployment

---

## 🧾 Reference

```
@article{madewithml,
  author       = {Goku Mohandas},
  title        = {Setup - Made With ML},
  howpublished = {\url{https://madewithml.com/}},
  year         = {2023}
}
```

---
