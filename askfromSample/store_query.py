"""
store_query.py - Ask questions about your store data, get answers as tables.

How it works:
  1. Pandas builds many summary tables in advance (region wise, segment wise, pivots...)
  2. Gemini reads your question and picks the right table (or a custom grouping)
  3. Pandas returns the real table, so the numbers are always exact

Run:  python store_query.py
"""
import os
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ===========================================================================
# SETTINGS: change these to match your file and column names
# ===========================================================================
FILE_PATH = "Sample.xlsx"
SHEET_NAME = 0                      # 0 = first sheet, or write the sheet name
MODEL = "gemini-3.5-flash"

ORDER_ID = "orderid"
ORDER_DATE = "orderdate"
SHIP_DATE = "shipdate"
SHIP_MODE = "shipmode"
CUSTOMER = "customername"
SEGMENT = "segment"
CITY = "city"
STATE = "state"
REGION = "region"
CATEGORY = "category"
SUB_CATEGORY = "sub_category"
PRODUCT = "productname"
SALES = "sales"
QUANTITY = "quantity"
DISCOUNT = "discount"
PROFIT = "profit"

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))   # created once


# ===========================================================================
# 1. Load data and add helper columns
# ===========================================================================
def load_data():
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]
    df[ORDER_DATE] = pd.to_datetime(df[ORDER_DATE], errors="coerce")
    df["Year"] = df[ORDER_DATE].dt.year
    df["Quarter"] = df[ORDER_DATE].dt.year.astype(str) + "-Q" + df[ORDER_DATE].dt.quarter.astype(str)
    df["Month"] = df[ORDER_DATE].dt.to_period("M").astype(str)          # 2024-03
    df["Month Name"] = df[ORDER_DATE].dt.strftime("%b")                 # Mar
    if SHIP_DATE in df.columns:
        df[SHIP_DATE] = pd.to_datetime(df[SHIP_DATE], errors="coerce")
        df["Ship Days"] = (df[SHIP_DATE] - df[ORDER_DATE]).dt.days
    if DISCOUNT in df.columns:
        df["Discount Band"] = pd.cut(df[DISCOUNT], bins=[-0.01, 0, 0.1, 0.2, 0.3, 1],
                                     labels=["No discount", "Up to 10%", "11-20%", "21-30%", "Above 30%"]).astype(str)
    return df


# ===========================================================================
# 2. One standard summary used by every table
# ===========================================================================
def summarize(data, by):
    """Sales, Profit, Quantity, Orders, Margin % and Share % for any grouping."""
    t = data.groupby(by, observed=True).agg(
        Sales=(SALES, "sum"),
        Profit=(PROFIT, "sum"),
        Quantity=(QUANTITY, "sum"),
        Orders=(ORDER_ID, "nunique"),
    ).reset_index()
    t["Profit Margin %"] = (t["Profit"] / t["Sales"] * 100).round(1)
    t["Sales Share %"] = (t["Sales"] / t["Sales"].sum() * 100).round(1)
    return t.round(2)


def pivot(data, rows, cols, value=SALES):
    p = pd.pivot_table(data, index=rows, columns=cols, values=value, aggfunc="sum",
                       fill_value=0, margins=True, margins_name="Total")
    p.columns = [str(c) for c in p.columns]
    return p.round(2).reset_index()


# ===========================================================================
# 3. Build all summary tables (name -> description + DataFrame)
# ===========================================================================
def build_tables(df):
    T = {}

    def add(name, desc, table):
        T[name] = {"description": desc, "data": table}

    # --- single dimension ---
    for name, col in [("region", REGION), ("state", STATE), ("city", CITY), ("segment", SEGMENT),
                      ("category", CATEGORY), ("sub_category", SUB_CATEGORY), ("product", PRODUCT),
                      ("customer", CUSTOMER), ("ship_mode", SHIP_MODE)]:
        if col in df.columns:
            add(name, f"{col} wise sales, profit, quantity, orders, margin %, share %",
                summarize(df, col).sort_values("Sales", ascending=False))

    # --- time ---
    year = summarize(df, "Year").sort_values("Year")
    year["Sales Growth %"] = (year["Sales"].pct_change() * 100).round(1)
    add("year", "Year wise sales, profit, margin and year-on-year (YoY) growth %", year)
    add("quarter", "Quarter wise sales and profit", summarize(df, "Quarter").sort_values("Quarter"))
    month = summarize(df, "Month").sort_values("Month")
    month["Sales Growth %"] = (month["Sales"].pct_change() * 100).round(1)
    add("month", "Month wise sales, profit and month-on-month (MoM) growth %", month)

    # --- two dimensions (long format, easy to filter) ---
    add("region_category", "Region and category wise sales and profit", summarize(df, [REGION, CATEGORY]))
    add("region_year", "Region and year wise sales and profit", summarize(df, [REGION, "Year"]))
    add("segment_year", "Segment and year wise sales and profit", summarize(df, [SEGMENT, "Year"]))
    add("category_year", "Category and year wise sales and profit", summarize(df, [CATEGORY, "Year"]))
    add("sub_category_region", "Sub-category and region wise sales and profit",
        summarize(df, [SUB_CATEGORY, REGION]))

    # --- pivots (wide format) ---
    add("pivot_region_category_sales", "PIVOT: regions as rows, categories as columns, values = sales",
        pivot(df, REGION, CATEGORY))
    add("pivot_region_category_profit", "PIVOT: regions as rows, categories as columns, values = profit",
        pivot(df, REGION, CATEGORY, PROFIT))
    add("pivot_category_year_sales", "PIVOT: categories as rows, years as columns, values = sales",
        pivot(df, CATEGORY, "Year"))
    add("pivot_segment_year_sales", "PIVOT: segments as rows, years as columns, values = sales",
        pivot(df, SEGMENT, "Year"))
    add("pivot_region_year_sales", "PIVOT: regions as rows, years as columns, values = sales",
        pivot(df, REGION, "Year"))
    add("pivot_sub_category_region_profit", "PIVOT: sub-categories as rows, regions as columns, values = profit",
        pivot(df, SUB_CATEGORY, REGION, PROFIT))

    # --- rankings ---
    rp = summarize(df, [REGION, PRODUCT]).sort_values([REGION, "Sales"], ascending=[True, False])
    add("top3_products_each_region", "Top 3 products by sales in each region",
        rp.groupby(REGION).head(3).reset_index(drop=True))

    # --- problem finders ---
    prod = summarize(df, PRODUCT)
    add("loss_making_products", "Products with negative total profit (losses)",
        prod[prod["Profit"] < 0].sort_values("Profit"))
    sub = summarize(df, SUB_CATEGORY)
    add("loss_making_sub_categories", "Sub-categories with negative total profit",
        sub[sub["Profit"] < 0].sort_values("Profit"))
    if "Discount Band" in df.columns:
        add("discount_band", "How discount levels affect sales, profit and margin",
            summarize(df, "Discount Band"))
    if "Ship Days" in df.columns:
        sd = df.groupby(SHIP_MODE)["Ship Days"].agg(["mean", "min", "max"]).round(1).reset_index()
        sd.columns = [SHIP_MODE, "Avg Ship Days", "Min Ship Days", "Max Ship Days"]
        add("shipping_days", "Average, min and max shipping days for each ship mode", sd)
    return T


# ===========================================================================
# 4. Data profile: what Gemini needs to know (never the full data)
# ===========================================================================
def data_profile(df):
    lines = [f"Rows: {len(df)}",
             f"Date range: {df[ORDER_DATE].min().date()} to {df[ORDER_DATE].max().date()}",
             "Columns:"]
    for col in df.columns:
        if df[col].dtype.kind == "M":
            lines.append(f"- {col} (date)")
        elif df[col].dtype.kind in "if" and col != "Year":
            lines.append(f"- {col} (number)")
        elif df[col].nunique() <= 30:
            lines.append(f"- {col}: {', '.join(map(str, sorted(df[col].dropna().unique())))}")
        else:
            lines.append(f"- {col} (text, {df[col].nunique()} unique values)")
    return "\n".join(lines)


def table_catalog(tables):
    return "\n".join(f"- {name}: {t['description']} | columns: {list(t['data'].columns)}"
                     for name, t in tables.items())


# ===========================================================================
# 5. Ask Gemini to choose a table (or a custom grouping)
# ===========================================================================
SYSTEM_PROMPT = """You help users query a store sales dataset. You never calculate numbers.
You only decide WHICH table answers the question and how to filter/sort it.

Return ONLY a JSON object:
{
  "route": "table" or "custom" or "unknown",
  "table": table name from the catalog (for route "table"),
  "group_by": list of columns (for route "custom"),
  "pivot_columns": column to spread across the top, or null (for a custom pivot),
  "pivot_value": "Sales" or "Profit" or "Quantity" (for a custom pivot),
  "filters": [{"column": "...", "op": "==|!=|>|<|>=|<=|in", "value": ...}],
  "sort_by": column name or null,
  "ascending": true or false,
  "top_n": number or null,
  "explanation": "one short sentence of what you understood"
}

Rules:
- Prefer route "table" when a ready table fully answers the question (filters/sort/top_n allowed on it).
- Use route "custom" when the question needs a combination no table has, e.g.
  "region wise sales of Furniture in 2024" -> group_by ["Region"], filters Category == Furniture and Year == 2024.
- "X wise" means group by X. "pivot"/"X vs Y"/"X by Y" means a pivot.
- Use exact column and table names and exact values from the data profile.
- "top 5" -> sort descending + top_n 5. "lowest/bottom/least" -> ascending true.
- If there is a previous query and the new question is a follow-up ("now only West",
  "sort by profit", "make it a pivot"), modify the previous query instead of starting over.
- If the data cannot answer it, route "unknown" and explain why in "explanation".
"""


def ask_llm(question, tables, profile, last_spec=None, error=None):
    prompt = f"DATA PROFILE:\n{profile}\n\nTABLE CATALOG:\n{table_catalog(tables)}\n\n"
    if last_spec:
        prompt += f"PREVIOUS QUERY (for follow-ups):\n{json.dumps(last_spec)}\n\n"
    if error:
        prompt += f"YOUR LAST ANSWER FAILED WITH THIS ERROR, FIX IT:\n{error}\n\n"
    prompt += f"QUESTION: {question}"

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
                                           response_mime_type="application/json",
                                           temperature=0),
    )
    return json.loads(response.text)


# ===========================================================================
# 6. Run Gemini's choice with pandas
# ===========================================================================
def find_column(data, name):
    """Case-insensitive column match: 'region' finds 'Region'."""
    for c in data.columns:
        if c.lower() == str(name).lower():
            return c
    raise ValueError(f"Column '{name}' not found. Available: {list(data.columns)}")


def apply_filters(data, filters):
    ops = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}
    for f in filters or []:
        col = find_column(data, f["column"])
        val = f["value"]
        if data[col].dtype.kind in "if":                       # numeric column
            vals = val if isinstance(val, list) else [val]
            vals = [float(v) for v in vals]
            data = data[data[col].isin(vals)] if f["op"] == "in" else data[getattr(data[col], ops[f["op"]])(vals[0])]
        else:                                                   # text column, ignore case
            text = data[col].astype(str).str.lower()
            vals = [str(v).lower() for v in (val if isinstance(val, list) else [val])]
            if f["op"] in ("==", "in"):
                data = data[text.isin(vals)]
            elif f["op"] == "!=":
                data = data[~text.isin(vals)]
            else:
                raise ValueError(f"Operator {f['op']} doesn't work on text column {col}")
    return data


def run_spec(spec, df, tables):
    route = spec.get("route")
    if route == "unknown":
        raise ValueError(spec.get("explanation") or "This question can't be answered from the data.")

    if route == "table":
        if spec.get("table") not in tables:
            raise ValueError(f"Table '{spec.get('table')}' doesn't exist.")
        result = apply_filters(tables[spec["table"]]["data"], spec.get("filters"))

    elif route == "custom":
        data = apply_filters(df, spec.get("filters"))
        if data.empty:
            raise ValueError("No rows match these filters.")
        rows = [find_column(df, c) for c in spec.get("group_by") or []]
        if spec.get("pivot_columns"):
            value = find_column(df, spec.get("pivot_value") or SALES)
            result = pivot(data, rows, find_column(df, spec["pivot_columns"]), value)
        elif rows:
            result = summarize(data, rows)
        else:   # a single total, e.g. "total sales in 2024"
            result = summarize(data.assign(All="Total"), "All").drop(columns="Sales Share %")
    else:
        raise ValueError(f"Unknown route '{route}'.")

    if result.empty:
        raise ValueError("No rows match these filters." if spec.get("filters")
                         else "The result is empty (nothing in the data meets this condition).")
    if spec.get("sort_by"):
        result = result.sort_values(find_column(result, spec["sort_by"]),
                                    ascending=bool(spec.get("ascending")))
    if spec.get("top_n"):
        result = result.head(int(spec["top_n"]))
    return result.reset_index(drop=True)


# ===========================================================================
# 7. Display helpers (Indian number format: 12,45,000.50)
# ===========================================================================
def inr(x):
    if pd.isna(x):
        return ""
    sign = "-" if x < 0 else ""
    whole, dec = f"{abs(x):.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        whole = ",".join([head] + groups + [tail]) if head else ",".join(groups + [tail])
    return f"{sign}{whole}.{dec}"


def pretty(table):
    out = table.copy()
    for col in out.columns:
        if out[col].dtype.kind not in "if" or col in ("Year",):
            continue
        if "%" in col:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.1f}%")
        elif col in ("Quantity", "Orders") or "Days" in col:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}" if float(v).is_integer() else f"{v:.1f}")
        else:
            out[col] = out[col].map(inr)
    return out.to_string(index=False)


# ===========================================================================
# 8. Main loop
# ===========================================================================
def answer(question, df, tables, profile, last_spec):
    """Ask Gemini, run with pandas. If it fails, send the error back once to self-correct."""
    spec = ask_llm(question, tables, profile, last_spec)
    try:
        return spec, run_spec(spec, df, tables)
    except Exception as first_error:
        if spec.get("route") == "unknown":
            raise
        spec = ask_llm(question, tables, profile, last_spec, error=str(first_error))
        return spec, run_spec(spec, df, tables)


def main():
    df = load_data()
    tables = build_tables(df)
    profile = data_profile(df)
    print(f"Loaded {len(df):,} rows from {FILE_PATH} | {len(tables)} summary tables ready")
    print("Try: region wise sales | top 5 products by profit | pivot of category vs year |")
    print("     segment wise sales of Furniture in 2024 | loss making products | year wise growth")
    print("Commands: 'tables' = list tables, 'save' = save last result, 'exit' = quit\n")

    last_spec, last_result = None, None
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "tables":
            print(table_catalog(tables), "\n")
            continue
        if q.lower() == "save":
            if last_result is None:
                print("Nothing to save yet.\n")
            else:
                last_result.to_excel("query_result.xlsx", index=False)
                print("Saved to query_result.xlsx\n")
            continue
        try:
            spec, result = answer(q, df, tables, profile, last_spec)
            source = spec["table"] if spec["route"] == "table" else "custom calculation"
            print(f"\n[{spec.get('explanation', '')}]  (source: {source})\n")
            print(pretty(result), "\n")
            last_spec, last_result = spec, result
        except Exception as e:
            print(f"Sorry: {e}\n")


if __name__ == "__main__":
    main()
