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


# import requests
# import streamlit as st
# from pathlib import Path

# # Streamlit UI
# st.title("Gender Prediction App")

# # Language choices
# language = st.selectbox("Select a language:", ["English", "French", "Spanish", "German", "Polish"])

# # Mapping language to text file
# language_to_file = {
#     "English": "english_text.txt",
#     "French": "french_text.txt",
#     "Spanish": "spanish_text.txt",
#     "German": "german_text.txt",
#     "Polish": "polish_text.txt",
# }

# # Load the selected text file
# text_file_path = Path("text_files") / language_to_file[language]

# # User input
# text = st.text_area("Enter a noun:", "Write your text here...")

# # Button to trigger the prediction
# if st.button("Predict"):
#     # API request
#     response = requests.post("http://localhost:8000/predict", json={"text": text})

#     # Display the result
#     if response.status_code == 200:
#         result = response.json()["result"]
#         st.success(f"The predicted gender is: {result}")
#     else:
#         st.error("Failed to get prediction. Please try again.")
