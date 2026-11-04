"""
it simply reads an Excel file, builds summary tables, and lets you ask questions in natural language.


"""
import os
import json
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ===========================================================================
# configureation:   change column names to match your file and column names
# ===========================================================================
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sample.xlsx")  # CHANGE 2
SHEET_NAME = 0                      # 0 = first sheet, or write the sheet name
MODEL = "gemini-3.5-flash"

ORDER_ID = "Order ID"
ORDER_DATE = "Order Date"
SHIP_DATE = "Ship Date"
SHIP_MODE = "Ship Mode"
CUSTOMER = "Customer Name"
SEGMENT = "Segment"
CITY = "City"
STATE = "State"
REGION = "Region"
CATEGORY = "Category"
SUB_CATEGORY = "Sub-Category"
PRODUCT = "Product Name"
SALES = "Sales"
QUANTITY = "Quantity"
DISCOUNT = "Discount"
PROFIT = "Profit"

load_dotenv()


# CHANGE 1: read the key from .env (local) or Streamlit Secrets (cloud)
@st.cache_resource
def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
    if not key:
        st.error("GEMINI_API_KEY not found. Add it to .env (local) or Streamlit Cloud -> Settings -> Secrets.")
        st.stop()
    return genai.Client(api_key=key)


# ===========================================================================
# 1. Load data and add helper columns
# ===========================================================================
def load_data(source=FILE_PATH):
    df = pd.read_excel(source, sheet_name=SHEET_NAME)
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
# 3. Building all summary tables (name -> description + DataFrame)
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
# 4. Data profile: what Gemini LLM needs to know
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

    response = get_client().models.generate_content(
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


# CHANGE 4: cache data + tables so they are built once, not on every click
@st.cache_data(show_spinner="Building summary tables...")
def prepare(file_bytes=None):
    df = load_data(io.BytesIO(file_bytes) if file_bytes else FILE_PATH)
    return df, build_tables(df), data_profile(df)


def money_format(table):
    """Indian number format for display, keeping the real numbers underneath."""
    fmt = {}
    for col in table.columns:
        if table[col].dtype.kind not in "if" or col == "Year":
            continue
        if "%" in col:
            fmt[col] = lambda v: "" if pd.isna(v) else f"{v:.1f}%"
        elif col in ("Quantity", "Orders") or "Days" in col:
            fmt[col] = lambda v: "" if pd.isna(v) else f"{v:,.1f}".replace(".0", "")
        else:
            fmt[col] = inr
    return table.style.format(fmt)


def show_result(item):
    st.caption(f"{item['explanation']}  ·  source: {item['source']}")
    st.dataframe(money_format(item["result"]), use_container_width=True, hide_index=True)


# CHANGE 3: Streamlit chat interface instead of input()/print()
def main():
    st.set_page_config(page_title="Store Insights", page_icon="🛒", layout="wide")
    st.title("🛒 Store Insights")
    st.caption("Ask anything about the Sample store data. pandas calculates the numbers and Gemini provides the insights. only 10 questions per 7hour are allowed, so please be patient if you hit the limit.")

    with st.sidebar:
        uploaded = st.file_uploader("Use your own Excel (optional)", type=["xlsx", "xls"])
        st.caption("Columns must match the names set at the top of the code.")
        if st.button("🗑️ Clear chat"):
            st.session_state.history, st.session_state.last_spec = [], None
            st.rerun()

    try:
        df, tables, profile = prepare(uploaded.getvalue() if uploaded else None)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    with st.sidebar:
        st.metric("Rows", f"{len(df):,}")
        st.metric("Ready tables", len(tables))
        with st.expander("See all tables"):
            for name, t in tables.items():
                st.write(f"**{name}**: {t['description']}")

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_spec", None)

    # Suggested questions for new users
    question = None
    if not st.session_state.history:
        st.write("**Try one of these:**")
        examples = ["region wise sales", "top 5 products by profit", "pivot of category vs year",
                    "segment wise sales of Furniture in 2024", "year wise growth", "which discount level makes losses"]
        cols = st.columns(3)
        for i, ex in enumerate(examples):
            if cols[i % 3].button(ex, use_container_width=True):
                question = ex

    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            if "error" in item:
                st.warning(item["error"])
            else:
                show_result(item)

    typed = st.chat_input("e.g. region wise sales, top 5 products in South, pivot of segment vs year")
    question = typed or question
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    spec, result = answer(question, df, tables, profile, st.session_state.last_spec)
                item = {"question": question, "result": result,
                        "explanation": spec.get("explanation", ""),
                        "source": spec["table"] if spec["route"] == "table" else "custom calculation"}
                show_result(item)
                buf = io.BytesIO()
                result.to_excel(buf, index=False)
                st.download_button("⬇️ Download Excel", buf.getvalue(), file_name="query_result.xlsx",
                                   key=f"dl_{len(st.session_state.history)}")
                st.session_state.last_spec = spec
            except Exception as e:
                item = {"question": question, "error": f"Sorry: {e}"}
                st.warning(item["error"])
        st.session_state.history.append(item)


main()
