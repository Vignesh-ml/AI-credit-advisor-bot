import streamlit as st
import pandas as pd
import pickle
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from database import init_db, create_user, verify_user, save_prediction, get_user_history

init_db()

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

def compare_scenarios(original_data, modified_changes):
    modified_data = original_data.copy()
    modified_data.update(modified_changes)
    
    original_pred, original_prob = predict_credit(original_data)
    modified_pred, modified_prob = predict_credit(modified_data)
    
    if original_pred == 1:
        original_score = int(600 + (original_prob[1] * 300))
    else:
        original_score = int(300 + (original_prob[0] * 200))
    
    if modified_pred == 1:
        modified_score = int(600 + (modified_prob[1] * 300))
    else:
        modified_score = int(300 + (modified_prob[0] * 200))
    
    return {
        'original_pred': original_pred,
        'original_score': original_score,
        'modified_pred': modified_pred,
        'modified_score': modified_score,
        'score_change': modified_score - original_score
    }

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
    # 1. Show on Streamlit Web UI
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a2e1a, #0a1f0a); 
                        border: 2px solid #00C851; border-radius: 15px; 
                        padding: 1.5rem; text-align: center; margin: 1rem 0;">
                <h2 style="color: #00C851; margin: 0;">✅ GOOD CREDIT RISK</h2>
                <p style="color: #888; margin: 0.5rem 0 0 0;">
                    Approval Probability: <strong style="color: #00C851;">{probability[1]*100:.1f}%</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
    
        # 2. Define variables for the PDF Report
        result_text = f"<font color='green'><b>✓ GOOD CREDIT RISK</b></font>"
        prob_text = f"Approval Probability: {probability[1]*100:.1f}%"
    else:
        # 1. Show on Streamlit Web UI
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2e1a1a, #1f0a0a); 
                        border: 2px solid #ff4444; border-radius: 15px; 
                        padding: 1.5rem; text-align: center; margin: 1rem 0;">
                <h2 style="color: #ff4444; margin: 0;">❌ HIGH CREDIT RISK</h2>
                <p style="color: #888; margin: 0.5rem 0 0 0;">
                    Risk Probability: <strong style="color: #ff4444;">{probability[0]*100:.1f}%</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # 2. Define variables for the PDF Report
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
    [data-testid="stSidebarHeader"] {
    display: none;
    }
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    body, p, div, span, h1, h2, h3, h4, h5, h6, button, input, textarea, label {
        font-family: 'Poppins', sans-serif !important;
    }
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Outlined' !important;
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
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None


# Title
st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1 style="color: #00C851; font-size: 2.5rem; margin-bottom: 0;">💳 AI Credit Advisor Bot</h1>
        <p style="color: #888; font-size: 1.1rem; margin-top: 0.5rem;">
            Powered by Machine Learning + Llama 3.3 AI
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
            <span style="background: #1a1a2e; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; color: #00C851;">
                ✅ ML Powered
            </span>
            <span style="background: #1a1a2e; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; color: #00C851;">
                🤖 AI Chatbot
            </span>
            <span style="background: #1a1a2e; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; color: #00C851;">
                📄 PDF Report
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    # Login Section
    if not st.session_state.logged_in:
        st.markdown("### 👤 Account")
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                if st.button("Login", use_container_width=True):
                    if verify_user(login_username, login_password):
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            with col_l2:
                if st.button("Skip →", use_container_width=True):
                    st.session_state.logged_in = False
                    st.session_state.username = "guest"
        
        with tab2:
            signup_username = st.text_input("Choose Username", key="signup_user")
            signup_password = st.text_input("Choose Password", type="password", key="signup_pass")
            
            if st.button("Create Account", use_container_width=True):
                if len(signup_username) < 3:
                    st.error("Username must be at least 3 characters")
                elif len(signup_password) < 4:
                    st.error("Password must be at least 4 characters")
                else:
                    success, message = create_user(signup_username, signup_password)
                    if success:
                        st.success(message)
                        st.session_state.logged_in = True
                        st.session_state.username = signup_username
                        st.rerun()
                    else:
                        st.error(message)
    else:
        st.markdown(f"""
            <div style="background: #12121e; border-radius: 10px; padding: 1rem; 
                        margin-bottom: 1rem; text-align: center;">
                <p style="color: #00C851; margin: 0; font-weight: 600;">
                    👋 Welcome, {st.session_state.username}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
    
    st.markdown("---")

    # Logo/Header with gradient
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0 1rem 0; 
                    background: linear-gradient(135deg, #0a1f0a, #1a1a2e);
                    border-radius: 12px; margin-bottom: 1rem;">
            <div style="font-size: 2.5rem;">💳</div>
            <h2 style="color: #00C851; margin: 0.3rem 0 0 0; font-size: 1.3rem;">
                Credit Guide
            </h2>
            <p style="color: #666; font-size: 0.75rem; margin: 0.2rem 0 0 0;">
                Know your score. Improve it.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Visual Score Meter
    st.markdown("""
        <div style="background: #12121e; border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
            <p style="color: #888; font-size: 0.8rem; margin: 0 0 0.5rem 0; font-weight: 600;">
                SCORE SPECTRUM
            </p>
            <div style="height: 8px; border-radius: 10px; margin-bottom: 0.5rem;
                        background: linear-gradient(90deg, #FF4444 0%, #FF8C00 33%, #FFD700 66%, #00C851 100%);">
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #666;">
                <span>300</span>
                <span>900</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Score Ranges — refined cards
    st.markdown("""<p style="color: #888; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem;">SCORE RANGES</p>""", unsafe_allow_html=True)
    
    score_ranges = [
        ("750-900", "Excellent", "#00C851", "🟢"),
        ("650-749", "Good", "#FFD700", "🟡"),
        ("550-649", "Fair", "#FF8C00", "🟠"),
        ("300-549", "Poor", "#FF4444", "🔴")
    ]
    
    for range_val, label, color, emoji in score_ranges:
        st.markdown(f"""
            <div style="background: #12121e; border: 1px solid {color}33;
                        border-left: 4px solid {color}; 
                        border-radius: 8px; padding: 0.6rem 0.9rem; 
                        margin-bottom: 0.5rem; display: flex; 
                        justify-content: space-between; align-items: center;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
                <span style="color: {color}; font-weight: 600; font-size: 0.9rem;">{emoji} {label}</span>
                <span style="color: #666; font-size: 0.8rem; font-family: monospace;">{range_val}</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin: 1.2rem 0;'></div>", unsafe_allow_html=True)

    # Key Tips — icon cards
    st.markdown("""<p style="color: #888; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem;">HOW TO IMPROVE</p>""", unsafe_allow_html=True)
    
    tips = [
        ("💰", "Pay bills on time"),
        ("📉", "Keep loan amounts low"),
        ("🏦", "Maintain savings"),
        ("💼", "Stable employment helps"),
        ("⏱️", "Shorter loan duration")
    ]
    
    for icon, tip in tips:
        st.markdown(f"""
            <div style="background: #12121e; border-radius: 8px; 
                        padding: 0.65rem 0.9rem; margin-bottom: 0.4rem;
                        display: flex; align-items: center; gap: 0.7rem;
                        border: 1px solid #ffffff0d;">
                <span style="font-size: 1.1rem;">{icon}</span>
                <span style="color: #ccc; font-size: 0.85rem;">{tip}</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin: 1.2rem 0;'></div>", unsafe_allow_html=True)

    # Tech badge
    st.markdown("""
        <div style="background: #12121e; border-radius: 10px; padding: 0.8rem; 
                    text-align: center; margin-bottom: 1rem; border: 1px solid #ffffff0d;">
            <p style="color: #666; font-size: 0.7rem; margin: 0 0 0.4rem 0;">POWERED BY</p>
            <p style="color: #00C851; font-size: 0.8rem; margin: 0; font-weight: 600;">
                Scikit-learn · Llama 3.3
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0;">
            <p style="color: #444; font-size: 0.7rem; margin: 0;">
                ⚠️ Educational prototype only
            </p>
        </div>
    """, unsafe_allow_html=True)

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

st.markdown("---")
compare_mode = st.checkbox("🔄 Compare with a modified scenario")

if compare_mode:
    st.info("💡 Adjust the values below to see how they affect your credit score")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        new_duration = st.number_input("New Loan Duration (months)", min_value=1, max_value=72, value=duration)
    with col_c2:
        new_credit_amount = st.number_input("New Credit Amount", min_value=100, max_value=20000, value=credit_amount)
    
    col_c3, col_c4 = st.columns(2)
    with col_c3:
        new_checking_status = st.selectbox(
            "New Checking Status",
            ["no checking", "little", "moderate", "rich"],
            index=["no checking", "little", "moderate", "rich"].index(checking_status)
        )
    with col_c4:
        new_savings_status = st.selectbox(
            "New Savings Status",
            ["no known savings", "little", "moderate", "quite rich", "rich"],
            index=["no known savings", "little", "moderate", "quite rich", "rich"].index(savings_status)
        )

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

    # Comparison mode
    if compare_mode:
        modified_changes = {
            'duration': new_duration,
            'credit_amount': new_credit_amount,
            'checking_status': new_checking_status,
            'savings_status': new_savings_status
        }
        
        comparison = compare_scenarios(input_data, modified_changes)
        
        st.markdown("---")
        st.subheader("📊 Scenario Comparison")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.markdown("**Current Profile**")
            st.metric("Score", f"{comparison['original_score']}/900")
            if comparison['original_pred'] == 1:
                st.success("GOOD")
            else:
                st.error("BAD")
        
        with col_r2:
            st.markdown("**Modified Profile**")
            st.metric("Score", f"{comparison['modified_score']}/900")
            if comparison['modified_pred'] == 1:
                st.success("GOOD")
            else:
                st.error("BAD")
        
        with col_r3:
            st.markdown("**Impact**")
            change = comparison['score_change']
            if change > 0:
                st.metric("Change", f"+{change} points", delta=f"{change}")
            elif change < 0:
                st.metric("Change", f"{change} points", delta=f"{change}")
            else:
                st.metric("Change", "No change")
        
        if comparison['score_change'] > 0:
            st.success(f"✅ These changes would improve your score by {comparison['score_change']} points!")
        elif comparison['score_change'] < 0:
            st.warning(f"⚠️ These changes would decrease your score by {abs(comparison['score_change'])} points")
        else:
            st.info("These changes don't significantly affect your score")

    st.session_state.current_profile = input_data
    st.session_state.current_prediction = prediction
    st.session_state.current_score = score
    st.session_state.chat_history = []

    # Save to database if logged in
    if st.session_state.logged_in and st.session_state.username != "guest":
        result_text = "Good" if prediction == 1 else "Bad"
        save_prediction(
            st.session_state.username, 
            score, 
            result_text, 
            credit_amount, 
            duration
        )

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

    
# Show history for logged in users
    if st.session_state.logged_in and st.session_state.username != "guest":
        history = get_user_history(st.session_state.username)
        
        if len(history) > 1:
            st.markdown("---")
            st.subheader("📈 Your Credit Score History")
            
            scores = [h[0] for h in history]
            attempts = list(range(1, len(scores) + 1))
            
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            
            ax.plot(attempts, scores, marker='o', color='#00C851', linewidth=2, markersize=8)
            ax.fill_between(attempts, scores, alpha=0.1, color='#00C851')
            
            ax.set_xlabel('Attempt', color='#888')
            ax.set_ylabel('Credit Score', color='#888')
            ax.tick_params(colors='#888')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#444')
            ax.spines['bottom'].set_color('#444')
            ax.grid(True, alpha=0.1)
            
            st.pyplot(fig)
            
            col_h1, col_h2, col_h3 = st.columns(3)
            with col_h1:
                st.metric("First Score", f"{scores[0]}/900")
            with col_h2:
                st.metric("Latest Score", f"{scores[-1]}/900")
            with col_h3:
                change = scores[-1] - scores[0]
                st.metric("Total Change", f"{change:+d} pts")

            show_history = st.checkbox("Show All Past Predictions")
            if show_history:
                for i, h in enumerate(history, 1):
                    st.markdown(f"**Attempt {i}** — Score: {h[0]}/900 | {h[1]} | {h[4]}")

# Chat section — only shows after prediction
if st.session_state.current_profile is not None:
    st.markdown("---")
    st.markdown("""
        <div style="background: #1a1a2e; border-radius: 10px; 
                    padding: 1rem; margin-bottom: 1rem;">
            <h3 style="color: #00C851; margin: 0;">💬 Ask Your Credit Advisor</h3>
            <p style="color: #888; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                Powered by Llama 3.3 — Ask anything about your credit result
            </p>
        </div>
    """, unsafe_allow_html=True)

    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(f"""
                <div style="background: #1a1a2e; border-radius: 10px; 
                            padding: 0.8rem; margin: 0.5rem 0; 
                            border-left: 3px solid #888;">
                    <strong style="color: #888;">You:</strong>
                    <p style="margin: 0.3rem 0 0 0;">{chat['message']}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background: #0a1f0a; border-radius: 10px; 
                            padding: 0.8rem; margin: 0.5rem 0;
                            border-left: 3px solid #00C851;">
                    <strong style="color: #00C851;">AI Advisor:</strong>
                    <p style="margin: 0.3rem 0 0 0;">{chat['message']}</p>
                </div>
            """, unsafe_allow_html=True)

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