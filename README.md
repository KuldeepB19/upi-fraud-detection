# 🛡️ UPI Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> **A full-stack Machine Learning web app that detects fraudulent UPI transactions in real time — no database, no complex setup, just 3 commands.**

---

## 📌 Table of Contents
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#️-project-structure)
- [How It Works](#-how-it-works)
- [Models](#-models)
- [Risk Score Logic](#-risk-score-logic)
- [Tech Stack](#️-tech-stack)
- [Author](#️-author)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏠 Dashboard | Live charts — fraud rate, amounts, locations, hour of day |
| 🔮 Predictor | Enter any transaction → get fraud risk score 0–100% |
| 📊 Data Explorer | Filter and browse transactions interactively |
| 📁 CSV Upload | Upload your own dataset for bulk fraud detection |
| 🤖 Dual Models | Random Forest + XGBoost compared side by side |
| 🎲 Data Generator | Generate synthetic UPI data — no CSV needed |
| 🌑 Dark Mode | Fully styled dark theme UI |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/KuldeepB19/upi-fraud-detection.git
cd upi-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> The app **auto-generates synthetic data** and **trains both models** on the very first run. No setup needed.

---

## 🗂️ Project Structure

```
upi-fraud-detection/
├── app.py                        ← Dashboard (entry point)
├── pages/
│   ├── 1_🔮_Predict.py           ← Single transaction predictor
│   ├── 2_📊_Data_Explorer.py     ← Filter & browse data
│   └── 3_📁_Upload_CSV.py        ← Bulk CSV analysis
├── src/
│   ├── data_generator.py         ← Synthetic UPI data generator
│   ├── train_model.py            ← RF + XGBoost training pipeline
│   └── utils.py                  ← Shared helpers & preprocessing
├── models/                       ← Saved .pkl model files
├── .streamlit/config.toml        ← Dark theme config
└── requirements.txt
```

---

## 🧠 How It Works

```
User Input / CSV
      ↓
Feature Engineering
(is_night, is_high_amount, is_unknown_location, ...)
      ↓
Random Forest  ──┐
                 ├──→ Ensemble Risk Score (0–100%)
XGBoost       ──┘
      ↓
🟢 Legitimate / 🟡 Review / 🔴 Fraud
```

1. **Data Generation** — Synthetic UPI transactions are created with realistic fraud patterns (night transfers, high amounts, unknown locations)
2. **Feature Engineering** — Raw fields are transformed into ML-ready features
3. **Dual Model Prediction** — Both RF and XGBoost give independent scores
4. **Risk Scoring** — Scores are averaged and mapped to a 0–100% fraud probability

---

## 🤖 Models

| Model | Algorithm | Handling Class Imbalance |
|-------|-----------|--------------------------|
| Random Forest | Ensemble (100 trees) | `class_weight='balanced'` |
| XGBoost | Gradient Boosting | `scale_pos_weight` |

**Input Features:**

| Feature | Type |
|---------|------|
| Transaction Amount | Numerical |
| Hour of Day | Numerical |
| Location | Categorical |
| Transaction Type | Categorical |
| Sender / Receiver Bank | Categorical |
| New Device Flag | Binary |
| Failed PIN Attempts | Numerical |
| `is_night` | Engineered |
| `is_high_amount` | Engineered |
| `is_unknown_location` | Engineered |

---

## 📊 Risk Score Logic

| Score | Level | Action |
|-------|-------|--------|
| 0–30% | 🟢 LOW | Transaction is likely legitimate |
| 30–60% | 🟡 MEDIUM | Needs manual review |
| 60–100% | 🔴 HIGH | Likely fraudulent — block/alert |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| Streamlit | Web UI & dashboard |
| Scikit-learn | Random Forest model |
| XGBoost | Gradient boosting model |
| Plotly | Interactive charts |
| Pandas / NumPy | Data processing |
| Joblib | Model serialisation |

---

## 👨‍💻 Author

**Kuldeep** — Big Data Capstone Project (Sem 3–4)

[![GitHub](https://img.shields.io/badge/GitHub-KuldeepB19-black?logo=github)](https://github.com/KuldeepB19)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).