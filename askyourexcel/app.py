"""
app.py - Web version with a text box.
Run:  streamlit run app.py
"""
import io
import streamlit as st
from engine import load_data, answer

st.set_page_config(page_title="Chat with your Excel", page_icon="📊", layout="wide")
st.title("📊 Chat with your Excel")

uploaded = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
if not uploaded:
    st.info("Upload a file to start. Try sample_store.xlsx.")
    st.stop()

# Load once per file (st.cache_data avoids re-reading on every click)
@st.cache_data
def get_data(file_bytes):
    return load_data(io.BytesIO(file_bytes))

df = get_data(uploaded.getvalue())
with st.expander(f"Preview data ({len(df)} rows)"):
    st.dataframe(df.head(20), use_container_width=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Show earlier questions and answers
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["q"])
    with st.chat_message("assistant"):
        if "error" in item:
            st.error(item["error"])
        else:
            st.caption(item["explanation"])
            st.dataframe(item["result"], use_container_width=True)

question = st.chat_input("e.g. region wise sales, month wise sales for 2024, create pivot of category by year")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                spec, result = answer(question, df)
            st.caption(f"Understood: {spec.explanation}")
            st.dataframe(result, use_container_width=True)

            # Simple chart for 1-dimension results (e.g. region wise sales)
            if spec.action in ("aggregate", "top_n") and result.shape[1] == 2:
                st.bar_chart(result.set_index(result.columns[0]))

            buf = io.BytesIO()
            result.to_excel(buf)
            st.download_button("⬇️ Download as Excel", buf.getvalue(),
                               file_name="result.xlsx", key=f"dl{len(st.session_state.history)}")
            st.session_state.history.append(
                {"q": question, "explanation": spec.explanation, "result": result})
        except Exception as e:
            st.error(f"Sorry: {e}")
            st.session_state.history.append({"q": question, "error": str(e)})
