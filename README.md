---

# ⚾ MLB Player Stats ML Project

> End-to-end machine learning system predicting MLB player performance metrics using scikit-learn, SHAP, and Flask. Includes full EDA, preprocessing, model training, explainability, and serving.

---

## 🧩 Project Overview

This repository contains a complete **Applied ML Engineering workflow**, from **data ingestion and preprocessing** to **model serving**.
It uses MLB player statistics to train a regression model that predicts **On-Base Percentage (OBP)** based on offensive performance metrics.

Developed as part of a **30-day MLOps upskilling sprint**, emphasizing reproducibility, pipeline design, and deployable ML systems.

---

## 🧱 Architecture

```
Data (.csv) → EDA & Cleaning → Feature Engineering →
Pipeline (Scaler + OneHot + Model) → Evaluation → Explainability (SHAP) → API (Flask)
```

---

## 🧠 Model Objective

- **Goal:** Predict player OBP (on-base percentage) from season stats
- **Model Type:** Linear Regression (wrapped in scikit-learn `Pipeline`)
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

# 4️⃣ Verify setup
python --version
pip list
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
│   ├── mlb_players_18_clean.csv  # Cleaned dataset (local)
│   └── (raw CSV ignored via .gitignore)
│
├── models/
│   └── mlb_salary_model.joblib   # Serialized trained model
│
├── notebooks/
│   └── mlb_eda.ipynb             # EDA, preprocessing, model dev, SHAP
│
├── requirements.txt              # Reproducible dependencies
├── .gitignore                    # Ignore data, venv, cache files
└── README.md                     # Documentation
```

---

## 🧮 Reproducible Training Workflow

<details>
<summary><b>Step 1: Data Cleaning & EDA</b></summary>

```python
import pandas as pd

df = pd.read_csv("data/mlb_players_18.csv")
df = df.dropna()
df.to_csv("data/mlb_players_18_clean.csv", index=False)
```

</details>

<details>
<summary><b>Step 2: Feature Engineering & Model Pipeline</b></summary>

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from joblib import dump

numeric_cols = ["games", "AB", "R", "H", "doubles", "HR", "walks", "OPS"]
categorical_cols = ["position"]

pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)
dump(pipeline, "models/mlb_salary_model.joblib")
```

</details>

<details>
<summary><b>Step 3: Evaluation Metrics</b></summary>

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
```

</details>

---

## 📊 Explainability with SHAP

<details>
<summary><b>Step 4: Global and Local Feature Importance</b></summary>

```python
import shap

explainer = shap.Explainer(model.named_steps["regressor"], X_train)
shap_values = explainer(X_train)

# Global feature importance
shap.summary_plot(shap_values, X_train, plot_type="bar")

# Local explanation for a single player
shap.waterfall_plot(shap_values[0])
```

</details>

---

## 🌐 Flask Model API

<details>
<summary><b>Flask Endpoint Example</b></summary>

```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

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

**Run locally:**

```bash
cd app
python3 mlb_predict.py
```

**Test via `curl`:**

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"games":120,"AB":410,"R":68,"H":125,"doubles":20,"HR":14,"walks":40,"OPS":0.725,"position":"OF"}'
```

---

## 🔍 Diagnostic Metrics Table

| Metric | Formula                | Description                |     |                                   |
| :----- | :--------------------- | :------------------------- | --- | --------------------------------- |
| MAE    | mean(                  | yₙ - ŷₙ                    | )   | Average absolute prediction error |
| RMSE   | sqrt(mean((yₙ - ŷₙ)²)) | Penalizes large errors     |     |                                   |
| R²     | 1 - (SS_res / SS_tot)  | Model variance explanation |     |                                   |

---

## 🧠 Feature Importance via Coefficients

<details>
<summary><b>Show example</b></summary>

```python
import pandas as pd

coeffs = model.named_steps["regressor"].coef_
importance = pd.DataFrame({
    "Feature": X.columns,
    "Weight": coeffs
}).sort_values("Weight", ascending=False)
importance.head(10)
```

</details>

---

## 🐳 Containerization (Planned)

<details>
<summary><b>Dockerfile (Preview)</b></summary>

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python3", "app/mlb_predict.py"]
```

</details>

---

## ✅ Current Progress

- [x] Data Cleaning + EDA
- [x] Model Training + Evaluation
- [x] SHAP Explainability
- [x] Flask API Deployment
- [ ] Dockerization
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
