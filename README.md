---
# ⚾ MLB Player Stats ML Project

End-to-end machine learning system predicting MLB player performance metrics using **scikit-learn**, **SHAP**, and **Flask**.
Includes full EDA, preprocessing, model training, explainability, containerization, and cloud deployment readiness.
---

## 🧩 Project Overview

This repository contains a complete **Applied ML Engineering workflow**, from **data ingestion and preprocessing** to **model serving and container deployment**.
It uses MLB player statistics to train a regression model that predicts **On-Base Percentage (OBP)** from key offensive metrics.

Developed as part of a **30-day MLOps upskilling sprint**, emphasizing reproducibility, modular pipeline design, explainability, and real-world deployment readiness.

---

## 🧱 Architecture

```
Data (.csv) → EDA & Cleaning → Feature Engineering →
Pipeline (Scaler + OneHot + Model) → Evaluation → Explainability (SHAP) → API (Flask) → Docker → AWS ECR
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

## 🧩 Model Explainability (Enhanced SHAP Insights)

This project integrates **SHAP (SHapley Additive exPlanations)** to provide both global and local interpretability for the regression model.
The explainability layer now includes **fully labeled features** derived from the pipeline’s transformed feature space, ensuring each numerical and categorical contribution is clearly visible.

<details>
<summary><b>Global and Local Interpretability</b></summary>

```python
import shap

# Initialize SHAP explainer
explainer = shap.Explainer(model.named_steps["regressor"], X_transformed)
shap_values = explainer(X_transformed)

# Global feature importance
shap.summary_plot(shap_values, X_transformed, plot_type="bar")

# --- Local Explainability (Labeled Features) ---

encoded_feature_names = preprocessor.get_feature_names_out().tolist()
sample_raw = X.iloc[[0]]
sample_transformed = preprocessor.transform(sample_raw)

shap_values_single = explainer(sample_transformed)

# Reconstruct explanation with proper feature names
shap_values_single = shap.Explanation(
    values=shap_values_single.values,
    base_values=shap_values_single.base_values,
    data=shap_values_single.data,
    feature_names=encoded_feature_names
)

# Visualize how each feature contributed to this player's OBP prediction
shap.waterfall_plot(shap_values_single[0], max_display=10)
```

✅ **Results:**

- Each feature (e.g., `HR`, `OPS`, `walks`, `position_OF`) now appears clearly in SHAP plots.
- **Global summary plots** show which features most strongly influence OBP predictions across all players.
- **Local waterfall plots** reveal how each individual player’s stats push their prediction up or down.

</details>

---

## 🧠 Model Explainability Preview

Here’s a visual summary of SHAP results from this project:

| Global Feature Importance                     | Local Player Breakdown                                   |
| :-------------------------------------------- | :------------------------------------------------------- |
| ![SHAP Bar](docs/images/shap_summary_bar.png) | ![SHAP Waterfall](docs/images/shap_waterfall_sample.png) |

> 📊 _Figure: SHAP summary and waterfall plots showing how hitting metrics like HR and OPS influence predicted OBP._

To recreate these visuals:

```bash
mkdir -p docs/images
# Then save figures using matplotlib or shap.plots API
```

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

## ☁️ AWS ECR Deployment & Cleanup

<details>
<summary><b>Push Container to AWS ECR</b></summary>

```bash
# 1️⃣ Create repository (only once)
aws ecr create-repository --repository-name mlb-ml-app --region us-east-1

# 2️⃣ Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com

# 3️⃣ Tag image
docker tag mlb-ml-app:latest <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/mlb-ml-app:latest

# 4️⃣ Push to ECR
docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/mlb-ml-app:latest
```

</details>

---

<details>
<summary><b>Clean Up Resources to Avoid Charges</b></summary>

```bash
# 🧹 Delete image from ECR
aws ecr batch-delete-image \
  --repository-name mlb-ml-app \
  --image-ids imageTag=latest \
  --region us-east-1

# 🧼 Remove local image
docker rmi mlb-ml-app

# 🧾 Optional: Delete repository (only if no longer needed)
aws ecr delete-repository \
  --repository-name mlb-ml-app \
  --region us-east-1 \
  --force
```

✅ Note: Keeping an empty repository incurs **no cost**.

</details>

---

## ✅ Current Progress

- [x] Data Cleaning + EDA
- [x] Model Training + Evaluation
- [x] SHAP Explainability
- [x] **Enhanced SHAP labeling for interpretability ✅**
- [x] Flask API Deployment
- [x] Docker Containerization
- [x] AWS ECR Upload + Cleanup
- [ ] AWS ECS (Deployment)

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
