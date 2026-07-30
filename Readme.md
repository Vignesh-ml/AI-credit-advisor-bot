# 💳 AI Credit Advisor Bot

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange)
![AI](https://img.shields.io/badge/AI-Llama%203.3-purple)
![Deploy](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-green)

> An end-to-end AI product that predicts credit risk for thin-file users using Machine Learning and provides personalized financial advice via an AI chatbot.

## 🔗 Live Demo
**[Try it here → https://rfbjjhrhzvekgarajqvsxw.streamlit.app/](https://rfbjjhrhzvekgarajqvsxw.streamlit.app/)**

---

## 🎯 Problem Statement

Millions of people in India — students, freshers, rural populations — get rejected for loans not because they're irresponsible, but because they have **no credit history**. Banks call them **thin-file users**.

Traditional credit scoring systems can't evaluate them. Our bot solves this.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 ML Prediction | Logistic Regression model trained on 1000 credit records |
| ⚠️ Risk Explanation | Identifies specific risk factors from user profile |
| 💡 Improvement Tips | Personalized steps to improve credit score |
| 💬 AI Chatbot | Llama 3.3 powered chatbot for follow-up questions |
| 📊 Credit Score | Estimates score on 300-900 scale |
| 🔐 Optional Login | Save prediction history across sessions, or skip and use as guest |
| 📈 Score History | Visual trend chart of past credit assessments |
| 📄 PDF Report | Downloadable credit assessment report |
| 🌐 Live Deployed | Accessible from any device |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML Model:** Scikit-learn (Logistic Regression)
- **AI Chatbot:** Groq API + Llama 3.3
- **PDF Generation:** ReportLab
- **Data:** German Credit Dataset (1000 records, 20 features)
- **Deployment:** Streamlit Cloud

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 74% |
| Recall (Bad Credit) | 76% |
| AUC Score | 0.81 |
| Precision (Bad Credit) | 54% |

> Logistic Regression was chosen over Random Forest specifically for its higher recall on bad credit detection (0.76 vs 0.42) — critical for credit risk applications.

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Vignesh-ml/AI-credit-advisor-bot.git

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key in .env file
GROQ_API_KEY=your-key-here

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure
   credit_bot/
├── app.py # Main application
├── credit_model.pkl # Trained ML model
├── scaler.pkl # Feature scaler
├── columns.pkl # Feature columns
├── requirements.txt # Dependencies
└── README.md # Documentation

---

## 👥 Built By

| Name | Contribution |
|---|---|
| Vignesh | Data prep, ML model, model evaluation, risk logic |
| Dev | Streamlit UI, AI chatbot, PDF report, deployment |

---

## 📚 What We Learned

- End-to-end ML pipeline from raw data to deployed product
- Why recall matters more than accuracy in credit scoring
- Integrating LLMs into real applications via API
- Streamlit for rapid AI app development
- Deploying ML apps to production

---

*Built as part of AIML coursework — BMS College of Engineering*