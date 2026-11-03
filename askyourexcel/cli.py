"""
cli.py - Chat with your Excel file from the terminal.
Run:  python cli.py sample_sales.xlsx
"""
import sys
from engine import load_data, answer

path = sys.argv[1] if len(sys.argv) > 1 else "Sample.xlsx"
df = load_data(path)
print(f"Loaded {path}: {len(df)} rows | columns: {', '.join(df.columns)}")
print("Ask things like: region wise sales | month wise sales for 2024 | "
      "create pivot of category by year | top 5 products")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("exit", "quit", "q"):
        break
    if not question:
        continue
    try:
        spec, result = answer(question, df)
        print(f"\n[Understood] {spec.explanation}\n")
        print(result.to_string(), "\n")
        if input("Save this result to Excel? (y/n): ").strip().lower() == "y":
            name = input("File name (e.g. result.xlsx): ").strip() or "result.xlsx"
            result.to_excel(name)
            print(f"Saved to {name}\n")
    except Exception as e:
        print(f"Sorry: {e}\n")
