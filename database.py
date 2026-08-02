import bcrypt
from datetime import datetime
from supabase import create_client
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Get credentials - works both locally and on Streamlit Cloud
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def init_db():
    # Tables already created via SQL Editor - nothing needed here
    pass


def create_user(username, password):
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    try:
        supabase.table('users').insert({
            'username': username,
            'password_hash': password_hash
        }).execute()
        return True, "Account created successfully"
    except Exception as e:
        if 'duplicate' in str(e).lower():
            return False, "Username already exists"
        return False, "Error creating account"


def verify_user(username, password):
    result = supabase.table('users').select('password_hash').eq('username', username).execute()
    
    if not result.data:
        return False
    
    stored_hash = result.data[0]['password_hash']
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))


def save_prediction(username, score, result, credit_amount, duration):
    supabase.table('predictions').insert({
        'username': username,
        'score': score,
        'result': result,
        'credit_amount': credit_amount,
        'duration': duration
    }).execute()


def get_user_history(username):
    result = supabase.table('predictions').select(
        'score, result, credit_amount, duration, created_at'
    ).eq('username', username).order('id', desc=False).execute()
    
    history = []
    for row in result.data:
        history.append((
            row['score'],
            row['result'],
            row['credit_amount'],
            row['duration'],
            row['created_at']
        ))
    return history