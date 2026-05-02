import streamlit as st
import pandas as pd
import pickle
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=groq_api_key)


# Get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model, scaler, columns
with open(os.path.join(BASE_DIR, 'credit_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, 'columns.pkl'), 'rb') as f:
    feature_columns = pickle.load(f)


def predict_credit(input_data):
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    return prediction, probability


def generate_explanation(input_data, prediction):
    reasons = []
    suggestions = []

    if input_data['checking_status'] == 'no checking':
        reasons.append("No checking account detected")
        suggestions.append("Open and maintain an active checking account")
    elif input_data['checking_status'] == 'little':
        reasons.append("Very low checking account balance")
        suggestions.append("Maintain a healthy checking account balance")

    if input_data['duration'] > 36:
        reasons.append(f"Very long loan duration ({input_data['duration']} months)")
        suggestions.append("Try to reduce loan duration below 24 months")
    elif input_data['duration'] > 24:
        reasons.append(f"Long loan duration ({input_data['duration']} months)")
        suggestions.append("Consider reducing loan duration if possible")

    if input_data['credit_amount'] > 10000:
        reasons.append(f"Very high loan amount requested (${input_data['credit_amount']})")
        suggestions.append("Consider requesting a smaller loan amount")
    elif input_data['credit_amount'] > 5000:
        reasons.append(f"High loan amount requested (${input_data['credit_amount']})")
        suggestions.append("A smaller loan amount would reduce your risk")

    if input_data['credit_history'] == 'delayed previously':
        reasons.append("Previous payment delays on record")
        suggestions.append("Clear all delayed payments and maintain clean history")
    elif input_data['credit_history'] == 'critical/other existing credit':
        reasons.append("Critical credit history detected")
        suggestions.append("Resolve all existing critical credit issues first")

    if input_data['savings_status'] == 'no known savings':
        reasons.append("No savings account detected")
        suggestions.append("Build a savings account with at least 3 months of expenses")
    elif input_data['savings_status'] == 'little':
        reasons.append("Very low savings balance")
        suggestions.append("Increase your savings to strengthen your application")

    if input_data['employment'] == 'unemployed':
        reasons.append("Currently unemployed")
        suggestions.append("Secure stable employment before applying for credit")
    elif input_data['employment'] == 'less than 1 year':
        reasons.append("Less than 1 year at current job")
        suggestions.append("Maintaining stable employment for longer improves your score")

    if input_data['housing'] == 'rent':
        reasons.append("Currently renting — no property ownership")
        suggestions.append("Property ownership significantly improves credit profile")

    if prediction == 1:
        if input_data['checking_status'] in ['moderate', 'rich']:
            reasons.append("Healthy checking account balance")
        if input_data['savings_status'] in ['quite rich', 'rich']:
            reasons.append("Strong savings account")
        if input_data['employment'] in ['4 to 7 years', 'more than 7 years']:
            reasons.append("Stable long term employment")
        if input_data['credit_history'] == 'existing paid':
            reasons.append("Good existing credit repayment history")

    if not reasons:
        if prediction == 1:
            reasons.append("Overall financial profile looks stable")
        else:
            reasons.append("Multiple risk factors detected in your profile")

    return reasons, suggestions


def chat_with_advisor(user_question, credit_profile, prediction, score):
    if prediction == 1:
        risk_level = "GOOD CREDIT RISK"
    else:
        risk_level = "HIGH CREDIT RISK"

    context = f"""
    You are an AI Credit Advisor bot. A user has just received their credit assessment.
    
    Their Credit Profile:
    - Age: {credit_profile['age']}
    - Loan Duration: {credit_profile['duration']} months
    - Credit Amount: ${credit_profile['credit_amount']}
    - Checking Status: {credit_profile['checking_status']}
    - Savings Status: {credit_profile['savings_status']}
    - Employment: {credit_profile['employment']}
    - Credit History: {credit_profile['credit_history']}
    - Housing: {credit_profile['housing']}
    - Purpose: {credit_profile['purpose']}
    
    Their Assessment Result:
    - Risk Level: {risk_level}
    - Estimated Credit Score: {score}/900
    
    Answer the user's question in a helpful, friendly, and simple way.
    Keep answers concise — maximum 4-5 sentences.
    Focus only on credit and financial advice.
    Don't make up specific numbers unless they're from the profile above.
    """

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": user_question}
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.units import inch
import io

def generate_pdf_report(profile, prediction, probability, score, reasons, suggestions):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                           rightMargin=inch, leftMargin=inch,
                           topMargin=inch, bottomMargin=inch)

    styles = getSampleStyleSheet()
    story = []

    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#00C851'),
        spaceAfter=10
    )

    # Header style
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=6
    )

    # Normal style
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=4
    )

    # Title
    story.append(Paragraph("AI Credit Advisor Report", title_style))
    story.append(Paragraph("Automated Credit Risk Assessment", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.green))
    story.append(Spacer(1, 0.2*inch))

    # Assessment Result
    story.append(Paragraph("Credit Assessment Result", header_style))
    if prediction == 1:
        result_text = f"<font color='green'><b>✓ GOOD CREDIT RISK</b></font>"
        prob_text = f"Approval Probability: {probability[1]*100:.1f}%"
    else:
        result_text = f"<font color='red'><b>✗ HIGH CREDIT RISK</b></font>"
        prob_text = f"Risk Probability: {probability[0]*100:.1f}%"

    story.append(Paragraph(result_text, normal_style))
    story.append(Paragraph(prob_text, normal_style))
    story.append(Paragraph(f"Estimated Credit Score: {score} / 900", normal_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 0.2*inch))

    # Profile Summary
    story.append(Paragraph("Your Financial Profile", header_style))
    profile_items = [
        f"Age: {profile['age']}",
        f"Loan Duration: {profile['duration']} months",
        f"Credit Amount: ${profile['credit_amount']}",
        f"Checking Status: {profile['checking_status']}",
        f"Savings Status: {profile['savings_status']}",
        f"Employment: {profile['employment']}",
        f"Credit History: {profile['credit_history']}",
        f"Housing: {profile['housing']}",
        f"Loan Purpose: {profile['purpose']}"
    ]
    for item in profile_items:
        story.append(Paragraph(f"• {item}", normal_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 0.2*inch))

    # Risk Factors or Positive Factors
    if prediction == 1:
        story.append(Paragraph("Positive Factors", header_style))
        positive = [r for r in reasons if r in [
            "Healthy checking account balance",
            "Strong savings account",
            "Stable long term employment",
            "Good existing credit repayment history"
        ]]
        if positive:
            for r in positive:
                story.append(Paragraph(f"✓ {r}", normal_style))
        else:
            story.append(Paragraph("• Overall financial profile appears stable", normal_style))
    else:
        story.append(Paragraph("Risk Factors Detected", header_style))
        for r in reasons:
            story.append(Paragraph(f"• {r}", normal_style))

    story.append(Spacer(1, 0.2*inch))

    # Improvement Suggestions
    if prediction == 0 and suggestions:
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("How to Improve Your Credit", header_style))
        for i, s in enumerate(suggestions, 1):
            story.append(Paragraph(f"{i}. {s}", normal_style))
        story.append(Spacer(1, 0.2*inch))

    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.green))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "⚠ This is an AI prototype for educational purposes only. Not financial advice.",
        styles['Italic']
    ))
    story.append(Paragraph(
        "Built with Python, Scikit-learn, Streamlit and ReportLab.",
        styles['Italic']
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Page config
st.set_page_config(
    page_title="AI Credit Advisor",
    page_icon="💳",
    layout="centered"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif !important;
    }
    .main {
        max-width: 800px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #00C851;
        color: white;
        font-size: 18px;
        padding: 12px;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #007E33;
    }
    .stMetric {
        background-color: #1E1E2E;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #00C851;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_profile' not in st.session_state:
    st.session_state.current_profile = None
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_score' not in st.session_state:
    st.session_state.current_score = None

# Title
st.title("💳 AI Credit Advisor Bot")
st.subheader("Find out your credit risk instantly")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📘 Credit Score Guide")
    st.markdown("---")
    st.markdown("**Score Ranges:**")
    st.markdown("🟢 750-900 — Excellent")
    st.markdown("🟡 650-749 — Good")
    st.markdown("🟠 550-649 — Fair")
    st.markdown("🔴 300-549 — Poor")
    st.markdown("---")
    st.markdown("**Key Tips:**")
    st.markdown("✅ Pay bills on time")
    st.markdown("✅ Keep loan amounts low")
    st.markdown("✅ Maintain savings")
    st.markdown("✅ Stable employment helps")
    st.markdown("✅ Shorter loan duration = better")
    st.markdown("---")
    st.caption("This is an AI prototype for educational purposes.")

# Input form
st.header("📋 Enter Your Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
with col2:
    duration = st.number_input("Loan Duration (months)", min_value=1, max_value=72, value=12)

col3, col4 = st.columns(2)
with col3:
    credit_amount = st.number_input("Credit Amount", min_value=100, max_value=20000, value=1000)
with col4:
    installment_commitment = st.number_input("Installment Commitment (1-4)", min_value=1, max_value=4, value=2)

st.markdown("---")
st.header("🏦 Financial Information")

col5, col6 = st.columns(2)
with col5:
    checking_status = st.selectbox(
        "Checking Account Status",
        ["no checking", "little", "moderate", "rich"]
    )
with col6:
    savings_status = st.selectbox(
        "Savings Account Status",
        ["no known savings", "little", "moderate", "quite rich", "rich"]
    )

col7, col8 = st.columns(2)
with col7:
    credit_history = st.selectbox(
        "Credit History",
        ["no credits/all paid", "all paid", "existing paid",
         "delayed previously", "critical/other existing credit"]
    )
with col8:
    employment = st.selectbox(
        "Employment Duration",
        ["unemployed", "less than 1 year", "1 to 4 years",
         "4 to 7 years", "more than 7 years"]
    )

st.markdown("---")
st.header("👤 Personal Information")

col9, col10 = st.columns(2)
with col9:
    housing = st.selectbox("Housing", ["own", "free", "rent"])
with col10:
    purpose = st.selectbox(
        "Loan Purpose",
        ["car", "furniture/equipment", "radio/tv",
         "domestic appliance", "repairs", "education",
         "business", "other"]
    )

col11, col12 = st.columns(2)
with col11:
    job = st.selectbox(
        "Job Type",
        ["unskilled resident", "unskilled non-resident",
         "skilled", "highly skilled"]
    )
with col12:
    personal_status = st.selectbox(
        "Personal Status",
        ["male single", "female div/dep/mar",
         "male div/sep", "male mar/wid"]
    )

col13, col14 = st.columns(2)
with col13:
    property_magnitude = st.selectbox(
        "Property",
        ["real estate", "life insurance", "car", "no known property"]
    )
with col14:
    other_payment_plans = st.selectbox(
        "Other Payment Plans",
        ["bank", "stores", "none"]
    )

col15, col16 = st.columns(2)
with col15:
    other_parties = st.selectbox(
        "Other Parties",
        ["none", "co applicant", "guarantor"]
    )
with col16:
    own_telephone = st.selectbox("Own Telephone", ["yes", "none"])

col17, col18 = st.columns(2)
with col17:
    foreign_worker = st.selectbox("Foreign Worker", ["yes", "no"])
with col18:
    existing_credits = st.number_input(
        "Existing Credits", min_value=1, max_value=4, value=1
    )

col19, col20 = st.columns(2)
with col19:
    residence_since = st.number_input(
        "Residence Since (years)", min_value=1, max_value=4, value=2
    )
with col20:
    num_dependents = st.number_input(
        "Number of Dependents", min_value=1, max_value=2, value=1
    )

st.markdown("---")

# Predict button
predict_btn = st.button("🔍 Check My Credit Risk", use_container_width=True)

if predict_btn:
    if credit_amount < 100:
        st.warning("⚠️ Credit amount seems too low. Please check your input.")
        st.stop()
    if age < 18:
        st.warning("⚠️ Applicant must be at least 18 years old.")
        st.stop()

    input_data = {
        'checking_status': checking_status,
        'duration': duration,
        'credit_history': credit_history,
        'purpose': purpose,
        'credit_amount': credit_amount,
        'savings_status': savings_status,
        'employment': employment,
        'installment_commitment': installment_commitment,
        'personal_status': personal_status,
        'other_parties': other_parties,
        'residence_since': residence_since,
        'property_magnitude': property_magnitude,
        'age': age,
        'other_payment_plans': other_payment_plans,
        'housing': housing,
        'existing_credits': existing_credits,
        'job': job,
        'num_dependents': num_dependents,
        'own_telephone': own_telephone,
        'foreign_worker': foreign_worker
    }

    with st.spinner("Analyzing your credit profile..."):
        prediction, probability = predict_credit(input_data)

    reasons, suggestions = generate_explanation(input_data, prediction)

    st.markdown("---")
    st.header("📊 Your Credit Assessment")

    if prediction == 1:
        st.success("✅ GOOD CREDIT RISK")
        st.metric("Approval Probability", f"{probability[1]*100:.1f}%")
    else:
        st.error("❌ HIGH CREDIT RISK")
        st.metric("Risk Probability", f"{probability[0]*100:.1f}%")

    st.markdown("---")
    if prediction == 1:
        st.subheader("✅ Positive Factors")
        positive_reasons = [r for r in reasons if r in [
            "Healthy checking account balance",
            "Strong savings account",
            "Stable long term employment",
            "Good existing credit repayment history"
        ]]
        if positive_reasons:
            for reason in positive_reasons:
                st.markdown(f"• {reason}")
        else:
            st.markdown("• Overall financial profile appears stable")
    else:
        st.subheader("⚠️ Risk Factors Detected")
        for reason in reasons:
            st.markdown(f"• {reason}")

    if prediction == 0:
        st.markdown("---")
        st.subheader("💡 How to Improve Your Credit")
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f"**{i}.** {suggestion}")
        else:
            st.markdown("Focus on maintaining stable income and payment history")

    st.markdown("---")
    st.subheader("📊 Estimated Credit Score Range")
    if prediction == 1:
        score = int(600 + (probability[1] * 300))
    else:
        score = int(300 + (probability[0] * 200))

    st.session_state.current_profile = input_data
    st.session_state.current_prediction = prediction
    st.session_state.current_score = score
    st.session_state.chat_history = []

    st.metric("Estimated Score", f"{score} / 900")
    st.progress(min(max(score / 900, 0.0), 1.0))

    # PDF Download
    st.markdown("---")
    st.subheader("📄 Download Your Report")
    pdf_buffer = generate_pdf_report(
        input_data, prediction, probability,
        score, reasons, suggestions
    )
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_buffer,
        file_name="credit_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    st.markdown("---")
    st.caption("⚠️ This is an AI prototype for educational purposes only. Not financial advice.")
    st.caption("Built with Python, Scikit-learn, and Streamlit.")

# Chat section — only shows after prediction
if st.session_state.current_profile is not None:
    st.markdown("---")
    st.subheader("💬 Ask Your Credit Advisor")
    st.caption("Ask anything about your credit result")

    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(f"**You:** {chat['message']}")
        else:
            st.markdown(f"**Advisor:** {chat['message']}")

    with st.form(key='chat_form', clear_on_submit=True):
        user_question = st.text_area(
            "Type your question here...",
            placeholder="e.g. Why is my score low? How can I improve?",
            height=100
        )
        ask_btn = st.form_submit_button("Ask Advisor 💬")

    if ask_btn and user_question:
        with st.spinner("Advisor is thinking..."):
            response = chat_with_advisor(
                user_question,
                st.session_state.current_profile,
                st.session_state.current_prediction,
                st.session_state.current_score
            )
        st.session_state.chat_history.append({
            'role': 'user',
            'message': user_question
        })
        st.session_state.chat_history.append({
            'role': 'advisor',
            'message': response
        })
        st.rerun()