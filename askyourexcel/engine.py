"""
engine.py - The brain of the "Chat with your Excel" app.

Design: the LLM is only a TRANSLATOR (question -> JSON instruction).
        pandas is the CALCULATOR (JSON instruction -> correct numbers).
"""
import os
import json
from typing import Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()
MODEL = "gemini-3.5-flash"
def get_api_key():
    """Read the key from .env (local) or Streamlit secrets (cloud)."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
    if not key:
        raise ValueError("GEMINI_API_KEY not found. Add it to .env (local) "
                         "or to Streamlit Cloud → Settings → Secrets.")
    return key

# ---------------------------------------------------------------------------
# 1. The "instruction format" the LLM must return (validated with pydantic)
# ---------------------------------------------------------------------------
class Filter(BaseModel):
    column: str
    op: Literal["==", "!=", ">", "<", ">=", "<=", "in"]
    value: Union[str, int, float, list]


class QuerySpec(BaseModel):
    action: Literal["aggregate", "pivot", "top_n", "describe", "unknown"]
    group_by: list[str] = []           # rows of the result, e.g. ["Region"]
    pivot_columns: list[str] = []      # only for pivot, e.g. ["Year"]
    value: Optional[str] = None        # numeric column, e.g. "Sales"
    agg: Literal["sum", "mean", "count", "max", "min"] = "sum"
    time_grain: Literal["none", "year", "quarter", "month"] = "none"
    filters: list[Filter] = []
    n: Optional[int] = None            # for top_n
    sort: Literal["desc", "asc", "natural"] = "desc"
    explanation: str = ""              # one line: what the LLM understood


# ---------------------------------------------------------------------------
# 2. Load data and add helper time columns (Year, Quarter, Month)
# ---------------------------------------------------------------------------
def load_data(path_or_file) -> pd.DataFrame:
    df = pd.read_excel(path_or_file)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = find_date_column(df)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["Year"] = df[date_col].dt.year
        df["Quarter"] = df[date_col].dt.to_period("Q").astype(str)
        df["Month"] = df[date_col].dt.to_period("M").astype(str)   # e.g. 2024-03
    return df


def find_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:  # fallback: a column whose name contains "date"
        if "date" in col.lower():
            return col
    return None


# ---------------------------------------------------------------------------
# 3. Describe the data to the LLM (column names + sample values, NOT all rows)
# ---------------------------------------------------------------------------
def _is_text(series: pd.Series) -> bool:
    """True for text columns (works on old 'object' and new 'str' pandas dtypes)."""
    return not (pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_datetime64_any_dtype(series))


def schema_context(df: pd.DataFrame) -> str:
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if _is_text(df[col]) and df[col].nunique() <= 25:
            samples = ", ".join(map(str, df[col].dropna().unique()[:25]))
            lines.append(f"- {col} ({dtype}); values: {samples}")
        else:
            lines.append(f"- {col} ({dtype}); example: {df[col].dropna().iloc[0]}")
    return "\n".join(lines)


SYSTEM_PROMPT = """You convert a user's question about a table into a JSON instruction.
Return ONLY a JSON object with these keys:
  action: "aggregate" | "pivot" | "top_n" | "describe" | "unknown"
  group_by: list of column names for result rows
  pivot_columns: list of column names spread across the top (pivot only)
  value: the numeric column to calculate (or null for count)
  agg: "sum" | "mean" | "count" | "max" | "min"
  time_grain: "none" | "year" | "quarter" | "month"
  filters: list of {"column", "op", "value"}; op is one of == != > < >= <= in
  n: number of rows for top_n, else null
  sort: "desc" | "asc" | "natural"
  explanation: one short sentence describing what you understood

Rules:
- Use ONLY column names from the table description, spelled exactly.
- "X wise" means group_by X. "month wise" -> time_grain "month", "year wise" -> "year".
- When time_grain is used, sort "natural" (chronological) unless the user asks for highest/lowest.
- "pivot" or "X by Y" with two dimensions -> action "pivot" (group_by = rows, pivot_columns = columns).
- "top 5 products" -> action "top_n", n = 5.
- If the question cannot be answered from these columns, action "unknown" and explain why.
"""


def ask_llm_for_spec(question: str, df: pd.DataFrame) -> QuerySpec:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = f"Table columns:\n{schema_context(df)}\n\nUser question: {question}"
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",   # forces pure JSON output
            temperature=0,                           # same question -> same answer
        ),
    )
    try:
        return QuerySpec(**json.loads(response.text))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"The AI returned an invalid instruction: {e}")


# ---------------------------------------------------------------------------
# 4. Execute the instruction with pandas (safe: no exec(), only known actions)
# ---------------------------------------------------------------------------
def _match_col(df: pd.DataFrame, name: Optional[str]) -> Optional[str]:
    """Case-insensitive column match, so 'region' finds 'Region'."""
    if name is None:
        return None
    lookup = {c.lower(): c for c in df.columns}
    if name.lower() not in lookup:
        raise ValueError(f"Column '{name}' not found. Available: {list(df.columns)}")
    return lookup[name.lower()]


def _apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    for f in filters:
        col = _match_col(df, f.column)
        val = f.value
        if f.op == "in":
            val = val if isinstance(val, list) else [val]
            df = df[df[col].astype(str).str.lower().isin([str(v).lower() for v in val])]
        elif f.op in ("==", "!=") and _is_text(df[col]):
            mask = df[col].astype(str).str.lower() == str(val).lower()
            df = df[mask] if f.op == "==" else df[~mask]
        else:
            ops = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}
            df = df[getattr(df[col], ops[f.op])(val)]
    return df


def execute_spec(spec: QuerySpec, df: pd.DataFrame) -> pd.DataFrame:
    if spec.action == "unknown":
        raise ValueError(spec.explanation or "I couldn't understand that question.")

    data = _apply_filters(df, spec.filters)
    if data.empty:
        raise ValueError("No rows match your filters.")

    rows = [_match_col(df, c) for c in spec.group_by]
    time_col = {"year": "Year", "quarter": "Quarter", "month": "Month"}.get(spec.time_grain)
    if time_col and time_col not in rows and time_col not in spec.pivot_columns:
        rows.append(time_col)
    value = _match_col(df, spec.value)

    if spec.action == "describe":
        return data.describe(include="all").round(2)

    if spec.action == "pivot":
        cols = [_match_col(df, c) for c in spec.pivot_columns]
        if value is None:  # count pivot, e.g. "number of orders by region and year"
            return pd.crosstab([data[r] for r in rows], [data[c] for c in cols],
                               margins=True, margins_name="Total")
        return pd.pivot_table(
            data, index=rows or None, columns=cols or None, values=value,
            aggfunc="count" if spec.agg == "count" else spec.agg,
            fill_value=0, margins=True, margins_name="Total",
        ).round(2)

    # aggregate / top_n
    if not rows:  # a single total, e.g. "total sales"
        result = data[value].agg(spec.agg) if value else len(data)
        return pd.DataFrame({f"{spec.agg} of {value or 'rows'}": [round(result, 2)]})

    if spec.agg == "count" or value is None:
        result = data.groupby(rows).size().reset_index(name="Count")
        measure = "Count"
    else:
        result = data.groupby(rows)[value].agg(spec.agg).reset_index()
        measure = f"{spec.agg.capitalize()} of {value}" if spec.agg != "sum" else value
        result = result.rename(columns={value: measure})

    if spec.sort == "natural":
        result = result.sort_values(rows)
    else:
        result = result.sort_values(measure, ascending=(spec.sort == "asc"))
    if spec.action == "top_n" or spec.n:
        result = result.head(spec.n or 5)
    return result.reset_index(drop=True).round(2)


# ---------------------------------------------------------------------------
# 5. One function the CLI and the web app both call
# ---------------------------------------------------------------------------
def answer(question: str, df: pd.DataFrame):
    spec = ask_llm_for_spec(question, df)
    result = execute_spec(spec, df)
    return spec, result
