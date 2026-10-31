import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads the .env file (local only)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"])
question = st.text_input("Ask a question")
response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=question
)
st.write(response.text)




