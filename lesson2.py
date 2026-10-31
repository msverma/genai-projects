import pandas as pd
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads the .env file
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

file_path = 'Sample.xlsx'  # Specify the path to your CSV file
df = pd.read_excel(file_path)
summary= df.describe().to_string()
col= df.columns.tolist()

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=f"You are a data analyst. Here are the columns: {col} and summary statistics: {summary}. Give a short description of this dataset and 3 insights."
)
print(response.text)



#print("Summary of the DataFrame:", summary)
#print("Columns in the DataFrame:", col)

#print(df.head())
