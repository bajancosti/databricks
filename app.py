import os
import pandas as pd
import streamlit as st
from databricks import sql

st.set_page_config(page_title="PowerApps Parity Demo (Day 2)", layout="wide")

# ---- read env vars ----
SERVER_HOSTNAME = os.getenv("DATABRICKS_SERVER_HOSTNAME")
HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
TOKEN = os.getenv("DATABRICKS_TOKEN")

MAIN_TABLE = os.getenv("MAIN_TABLE", "workspace.app_demo.main_table")

if not SERVER_HOSTNAME or not HTTP_PATH or not TOKEN:
    st.error("Missing env vars. Need DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN.")
    st.stop()

@st.cache_data(ttl=30)
def query_df(query: str, params=None) -> pd.DataFrame:
    params = params or []
    with sql.connect(
        server_hostname=SERVER_HOSTNAME,
        http_path=HTTP_PATH,
        access_token=TOKEN,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=cols)

st.title("Databricks App - Test")

# Load lookup values
ref_category = query_df("SELECT category FROM workspace.app_demo.ref_category ORDER BY category")
ref_status = query_df("SELECT status FROM workspace.app_demo.ref_status ORDER BY status")

categories = ["(All)"] + (ref_category["category"].tolist() if not ref_category.empty else [])
statuses = ["(All)"] + (ref_status["status"].tolist() if not ref_status.empty else [])

# Filters
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    f_category = st.selectbox("Category", categories)
with c2:
    f_status = st.selectbox("Status", statuses)
with c3:
    f_search = st.text_input("Search by name (contains)", value="")

where = ["is_deleted = false"]
params = []

if f_category != "(All)":
    where.append("category = ?")
    params.append(f_category)

if f_status != "(All)":
    where.append("status = ?")
    params.append(f_status)

if f_search.strip():
    where.append("LOWER(name) LIKE ?")
    params.append(f"%{f_search.strip().lower()}%")

where_sql = " AND ".join(where)

q = f"""
SELECT id, name, category, amount, status, valid_id, updated_at, updated_by
FROM {MAIN_TABLE}
WHERE {where_sql}
ORDER BY updated_at DESC
"""

df = query_df(q, params)

left, right = st.columns([2, 1])

with left:
    st.subheader("Records")
    st.dataframe(df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Row details")
    if df.empty:
        st.info("No rows match the filters.")
    else:
        selected_id = st.selectbox("Select id", df["id"].tolist())
        row = df[df["id"] == selected_id].iloc[0].to_dict()
        st.json(row)

st.caption("Day 3: Add/Edit/Delete + validations + audit + Power Automate trigger.")
