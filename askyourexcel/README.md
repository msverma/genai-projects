# Chat with your Excel

Ask questions in plain English and get correct answers from your Excel data.
The LLM (Gemini) only translates your question into a JSON instruction;
pandas does all the calculation, so the numbers are always exact.

## Setup
1. Put these files in one folder and open a terminal there
2. `python -m venv venv` then activate it (Windows: `venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. Create a `.env` file containing: `GEMINI_API_KEY=your_key_here`

## Run
- Terminal version: `python cli.py sample_sales.xlsx`
- Web version: `streamlit run app.py` (opens in your browser)

## Try these questions
- region wise sales
- category wise sales
- month wise sales for 2024
- year wise sales
- create pivot of region by year
- top 3 products in North
- number of orders by category and quarter
- average sales by region

## Files
- engine.py: the brain (LLM translator + pandas calculator)
- cli.py: command-line chat
- app.py: Streamlit web app with chat box, charts, and Excel download
- sample_sales.xlsx: 1,200 practice orders (2023 to 2024)

Note: your Excel needs a date column for month/year questions.
Year, Quarter and Month columns are created automatically from it.
