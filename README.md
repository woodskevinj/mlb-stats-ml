# ⚾ MLB Stats ML Project

This project explores baseball player statistics and builds machine learning models to predict player performance metrics using real MLB data.  
The goal is to demonstrate a complete applied ML workflow — from data exploration to deployment — using clean, reproducible engineering practices.

---

## 📊 Dataset

**Source:** [MLB Players Dataset (2018)](https://www.kaggle.com/datasets)  
File: `data/mlb_players_18.csv`

Each row represents a Major League Baseball player with various statistical and demographic features (e.g., batting average, home runs, strikeouts, salary, position, etc.).

---

## 🧭 Project Structure

mlb-stats-ml/
│
├── data/ # Raw and processed data
│ └── mlb_players_18.csv
│
├── notebooks/ # Jupyter notebooks for exploration
│ └── mlb_eda.ipynb
│
├── scripts/ # Training and inference scripts
│ └── train_model.py
│
├── models/ # Saved trained models
│
├── requirements.txt # Environment dependencies
│
└── README.md # Project overview

---

## ⚙️ Environment Setup

```bash
# Clone repository
git clone git@github.com:woodskevinj/mlb-stats-ml.git
cd mlb-stats-ml

# Create virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

🚀 How to Run

1️⃣ Start Jupyter Notebook for EDA

jupyter notebook notebooks/mlb_eda.ipynb


2️⃣ Train Model

python scripts/train_model.py


3️⃣ Predict New Player Stats
Later, a Flask API will be added for real-time predictions.

🧠 Tech Stack

Python 3.10+

pandas, NumPy, matplotlib, seaborn

scikit-learn

Flask (for serving the model)

Jupyter Notebook

🧩 Key Concepts Practiced

Data cleaning and preprocessing

Feature engineering

Model training and evaluation

Model serialization (joblib)

Serving predictions via API

Version control and documentation

🧑‍💻 Author

Kevin Woods
Software Engineer | Applied ML Engineer in Progress
GitHub

🧾 Acknowledgments

Base project inspired by Made With ML by Goku Mohandas
.
Dataset sourced from public baseball statistics repositories on Kaggle.
```
