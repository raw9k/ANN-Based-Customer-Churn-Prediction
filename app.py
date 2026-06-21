import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pandas as pd
import pickle


model = load_model("model.h5")

with open("scaler.pkl",'rb') as file:
    scaler = pickle.load(file)
    
st.title("Customer churn Prediction")

Geography= st.selectbox("Geography",["France", "Germany", "Spain"])
Gender = st.selectbox("Gender",["Female","Male"])
CreditScore = st.number_input("Credit Score")
Age = st.slider("Age",18,90)
Tenure=st.slider("Tenure",0,10)
Balance=st.number_input("Balance")
NumOfProducts=st.slider("Number of Products",1,4)
HasCrCard= st.selectbox('Has Credit Card', [0,1])
IsActiveMember=	st.selectbox("Is Active Member",[0,1])
EstimatedSalary= st.number_input("Estimated Salary")


input_data = {
    'CreditScore' :CreditScore,
    'Geography' :Geography,	
    'Gender' :Gender,	
    'Age':Age,
    'Tenure':Tenure,
    'Balance':Balance,
    'NumOfProducts':NumOfProducts,
    'HasCrCard':HasCrCard,
    'IsActiveMember':IsActiveMember,	
    'EstimatedSalary': EstimatedSalary
}

input_df = pd.DataFrame([input_data])
input_df["Geography"] = pd.Categorical(
    input_df["Geography"], 
    categories=["France", "Germany", "Spain"]
)
input_df
le = LabelEncoder()

le.fit(["Female", "Male"])

input_df["Gender"] = le.transform(input_df["Gender"])
scaled_input = scaler.transform(input_df)

prediction = model.predict(scaled_input)
predict_prob = prediction[0][0]
if predict_prob>0.5:
    print('The customer is likely to churn')
else:
    print('The customer is not likely to churn') 