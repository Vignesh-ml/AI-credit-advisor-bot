# AI Credit Advisor Bot

An AI-powered credit risk assessment bot built with Python, Scikit-learn, Streamlit, and Groq LLM.

## What it does
- Predicts credit risk (Good/Bad) using Machine Learning
- Explains risk factors in simple language
- Gives personalized improvement suggestions
- AI chatbot for follow-up questions
- Generates downloadable PDF report

## Tech Stack
- Python
- Scikit-learn (Logistic Regression)
- Streamlit
- Groq API (Llama 3.3)
- ReportLab (PDF generation)
- Pandas, NumPy

## Dataset
German Credit Dataset (1000 records, 20 features)

## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Add your Groq API key in app.py
3. Run: streamlit run app.py

## Project Structure
- app.py — Main application
- credit_model.pkl — Trained ML model
- scaler.pkl — Feature scaler
- columns.pkl — Feature columns

## Built By
AIML Student — 2nd Semester