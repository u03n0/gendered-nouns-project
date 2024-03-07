import requests
import streamlit as st
from pathlib import Path


st.title("Gendered Noun Predictions Across Languages")

text = st.text_area("Enter a noun:", "Write your text here...")
