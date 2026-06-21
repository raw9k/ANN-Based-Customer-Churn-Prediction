import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pandas as pd
import pickle
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

# --- 2. CACHE MODEL & SCALER ---
# This prevents TF from reloading on every single button click or slider move
@st.cache_resource
def load_assets():
    model = load_model("model.h5")
    with open("scaler.pkl", 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

model, scaler = load_assets()

# --- 3. HEADER & STYLING ---
st.markdown("<h1 style='text-align: center; color: #1E88E5;'>🏦 Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: gray;'>Enter customer demographics and account details below to predict their likelihood of leaving.</p>", unsafe_allow_html=True)
st.divider()

# --- 4. COLUMN LAYOUT FOR INPUTS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Demographics")
    Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    Gender = st.selectbox("Gender", ["Female", "Male"])
    Age = st.slider("Age", 18, 90, 30)

with col2:
    st.markdown("### 💰 Financials")
    EstimatedSalary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0)
    Balance = st.number_input("Account Balance ($)", min_value=0.0, value=0.0, step=1000.0)
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

with col3:
    st.markdown("### 📈 Account Details")
    Tenure = st.slider("Tenure (Years)", 0, 10, 5)
    NumOfProducts = st.slider("Number of Products", 1, 4, 1)
    
    # Using Yes/No for better UX, then mapping to 1/0 later
    HasCrCard = st.selectbox('Has Credit Card', ["Yes", "No"])
    IsActiveMember = st.selectbox("Is Active Member", ["Yes", "No"])

st.divider()

# --- 5. PREDICTION BUTTON & OUTPUT ---
# We put the button and the result in columns to center the visual output
btn_col, chart_col = st.columns([1, 2])

with btn_col:
    st.markdown("<br><br>", unsafe_allow_html=True) # Spacer
    predict_button = st.button("Predict Churn Probability", type="primary", use_container_width=True)

if predict_button:
    # Map UX strings back to model integers
    card_val = 1 if HasCrCard == "Yes" else 0
    active_val = 1 if IsActiveMember == "Yes" else 0

    input_data = {
        'CreditScore': CreditScore,
        'Geography': Geography, 
        'Gender': Gender,   
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': card_val,
        'IsActiveMember': active_val,    
        'EstimatedSalary': EstimatedSalary
    }

    # Your exact preprocessing logic
    input_df = pd.DataFrame([input_data])
    input_df["Geography"] = pd.Categorical(
        input_df["Geography"], 
        categories=["France", "Germany", "Spain"]
    )
    input_df = pd.get_dummies(input_df, columns=["Geography"], dtype=int)
    
    le = LabelEncoder()
    le.fit(["Female", "Male"])
    input_df["Gender"] = le.transform(input_df["Gender"])
    
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)
    predict_prob = prediction[0][0]
    
    # --- 6. GAUGE CHART VISUALIZATION ---
    with chart_col:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predict_prob * 100,
            title = {'text': "Churn Probability (%)", 'font': {'size': 24}},
            number = {'suffix': "%", 'font': {'size': 40, 'color': "white" if predict_prob > 0.5 else "black"}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"}, # Hide standard bar
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': "#2ecc71"},   # Green - Safe
                    {'range': [30, 70], 'color': "#f1c40f"},  # Yellow - Warning
                    {'range': [70, 100], 'color': "#e74c3c"}  # Red - Danger
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': predict_prob * 100
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Final Text Verdict
        if predict_prob > 0.5:
            st.error("🚨 **High Risk:** The model indicates this customer is likely to churn.")
        else:
            st.success("✅ **Low Risk:** The model indicates this customer is likely to stay.")