import requests
import streamlit as st
from pathlib import Path


st.title("Gendered Noun Predictions Across Languages")

text = st.text_area("Enter a noun:", "Write your text here...")

# Button to trigger the prediction
if st.button("Predict"):
    # API request
    response = requests.post("http://localhost:8000/predict", json={"text": text})

    # Display the result
    if response.status_code == 200:
        result = response.json()["result"]
        st.success(f"The predicted gender is: {result}")
    else:
        st.error("Failed to get prediction. Please try again.")