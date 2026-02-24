import json
import os
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from databricks import sql

# ---------------------------
# APP SETTINGS
# ---------------------------
st.set_page_config(page_title="Databricks App — CRUD + Validation + Audit", layout="wide")

# Optional CSS fallback to force white background even if Streamlit theme is dark.
# (Recommended: use .streamlit/config.toml, see note at bottom.)
st.markdown(
    """
    <style>
      .stApp { background-color: white; color: black; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Databricks App — Day 5: UX polish + Field Validation + Concurrency + Roles")

# ---------------------------
# CONNECTION SETTINGS
# ---------------------------
SERVER_HOSTNAME = os.getenv("DATABRICKS_SERVER_HOSTNAME", "dbc-9bac1c79-b73f.cloud.databricks.com")
HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/a62d5f5dc3cff4c0")
TOKEN = os.getenv("DATABRICKS_TOKEN", "dapifd0b178a46048f2f36213023f1822b18")

MAIN_TABLE = os.getenv("MAIN_TABLE", "workspace.app_demo.main_table")
AUDIT_TABLE = "workspace.app_demo.audit_table"

REF_CATEGORY = "workspace.app_demo.ref_category"
REF_STATUS = "workspace.app_demo.ref_status"
REF_VALID_IDS = "workspace.app_demo.ref_valid_ids"

if "PASTE_" in SERVER_HOSTNAME or "PASTE_" in HTTP_PATH or "PASTE_" in TOKEN:
    st.error(
        "Connection settings are not configured. "
        "Paste your SERVER_HOSTNAME / HTTP_PATH / TOKEN in app.py (demo), "
        "or set them as environment variables later."
    )
    st.stop()

# ---------------------------
# SQL HELPERS
# ---------------------------
@st.cache_data(ttl=15)
def query_df(query: str, params=None) -> pd.DataFrame:
    """
    Runs a SELECT query on the Databricks SQL Warehouse and returns a pandas DataFrame.
    Uses parameter placeholders (?) to avoid SQL injection and quoting errors.
    """
    params = params or []
    with sql.connect(server_hostname=SERVER_HOSTNAME, http_path=HTTP_PATH, access_token=TOKEN) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=cols)


def exec_sql(query: str, params=None) -> None:
    """
    Runs a modifying query (INSERT/UPDATE/DELETE). No return.
    """
    params = params or []
    with sql.connect(server_hostname=SERVER_HOSTNAME, http_path=HTTP_PATH, access_token=TOKEN) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)


@st.cache_data(ttl=60)
def get_lookups():
    """
    Loads reference tables used for dropdowns & validation rules.
    """
    cat_df = query_df(f"SELECT category FROM {REF_CATEGORY} ORDER BY category")
    status_df = query_df(f"SELECT status FROM {REF_STATUS} ORDER BY status")
    valid_ids_df = query_df(f"SELECT valid_id FROM {REF_VALID_IDS} ORDER BY valid_id")

    categories = cat_df["category"].tolist() if not cat_df.empty else []
    statuses = status_df["status"].tolist() if not status_df.empty else []
    valid_ids = valid_ids_df["valid_id"].tolist() if not valid_ids_df.empty else []

    return categories, statuses, valid_ids


CATEGORIES, STATUSES, VALID_IDS = get_lookups()

# ---------------------------
# DAY 5 VALIDATION: field-level errors
# ---------------------------
def validate_row_fields(obj: dict) -> dict:
    """
    PowerApps-style validation: returns {field: error_message}. Empty dict = OK.
    """
    errors = {}

    # Required fields
    for k in ["id", "name", "category", "status", "valid_id"]:
        if not str(obj.get(k, "")).strip():
            errors[k] = "Required"

    # Amount numeric + range
    try:
        amt = float(obj.get("amount"))
        if amt < 0 or amt > 1_000_000:
            errors["amount"] = "Must be between 0 and 1,000,000"
    except Exception:
        errors["amount"] = "Must be a number"

    # Allowed list rules
    if CATEGORIES and obj.get("category") not in CATEGORIES:
        errors["category"] = "Invalid value (not in ref_category)"
    if STATUSES and obj.get("status") not in STATUSES:
        errors["status"] = "Invalid value (not in ref_status)"

    # Referential rule
    if VALID_IDS and obj.get("valid_id") not in VALID_IDS:
        errors["valid_id"] = "Invalid value (not in ref_valid_ids)"

    # Cross-field rule example: if BLOCKED then amount must be 0
    if obj.get("status") == "BLOCKED":
        try:
            if float(obj.get("amount")) != 0:
                errors["amount"] = "Must be 0 when status=BLOCKED"
        except Exception:
            errors["amount"] = "Must be 0 when status=BLOCKED"

    return errors


# ---------------------------
# AUDIT LOGGING
# ---------------------------
def write_audit(event_type: str, row_id: str, actor: str, old_obj, new_obj, ok: bool, msg: str):
    """
    Writes audit record for both successful and blocked attempts.
    """
    event_id = str(uuid.uuid4())
    event_ts = datetime.now(timezone.utc).isoformat()

    old_json = json.dumps(old_obj, default=str) if old_obj is not None else None
    new_json = json.dumps(new_obj, default=str) if new_obj is not None else None

    exec_sql(
        f"""
        INSERT INTO {AUDIT_TABLE}
        (event_id, event_ts, event_type, table_name, row_id, actor, old_json, new_json, validation_ok, validation_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [event_id, event_ts, event_type, MAIN_TABLE, row_id, actor, old_json, new_json, ok, msg],
    )


# ---------------------------
# HELPERS
# ---------------------------
def get_actor() -> str:
    # Demo actor (replace with real identity later)
    return "app_user_demo"


def clear_caches():
    st.cache_data.clear()


def show_field_error(errors: dict, field: str):
    """
    Small UI helper to show a red message under a field if it has an error.
    """
    if field in errors:
        st.caption(f"❌ {errors[field]}")


# ---------------------------
# DAY 5 UI: Sidebar navigation + Role toggle
# ---------------------------
st.sidebar.header("App Controls")
role = st.sidebar.selectbox("Role", ["Viewer", "Editor"], index=1)
is_editor = role == "Editor"

page = st.sidebar.radio("Navigation", ["Browse", "Add", "Edit", "Delete", "Audit"], index=0)

st.sidebar.caption(
    "Tip: Viewer disables write actions.\n"
    "This simulates business permissions in a prototype."
)

# ===========================
# PAGE: BROWSE
# ===========================
if page == "Browse":
    st.subheader("Browse records")

    # Day 5: Browse improvements
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    with c1:
        f_category = st.selectbox("Category", ["(All)"] + CATEGORIES, key="browse_category")
    with c2:
        f_status = st.selectbox("Status", ["(All)"] + STATUSES, key="browse_status")
    with c3:
        f_search = st.text_input("Search by name (contains)", value="", key="browse_search")
    with c4:
        show_deleted = st.checkbox("Show deleted", value=False, key="browse_deleted")

    sort_by = st.selectbox(
        "Sort",
        ["updated_at DESC", "id ASC", "name ASC", "amount DESC"],
        index=0,
        key="browse_sort",
    )

    where = ["1=1"]
    params = []

    if not show_deleted:
        where.append("is_deleted = false")

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
    SELECT id, name, category, amount, status, valid_id, is_deleted, updated_at, updated_by
    FROM {MAIN_TABLE}
    WHERE {where_sql}
    ORDER BY {sort_by}
    """

    df = query_df(q, params)

    st.caption(f"Records shown: {len(df)}")

    left, right = st.columns([2, 1])
    with left:
        st.markdown("### Records")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="main_table_export.csv",
            mime="text/csv",
            disabled=df.empty,
        )

    with right:
        st.markdown("### Row details")
        if df.empty:
            st.info("No rows match the filters.")
        else:
            selected_id = st.selectbox("Select id", df["id"].tolist(), key="browse_id")
            row = df[df["id"] == selected_id].iloc[0].to_dict()
            st.json(row)

# ===========================
# PAGE: ADD
# ===========================
elif page == "Add":
    st.subheader("Add record (Create)")

    if not is_editor:
        st.warning("Viewer role: Add is disabled.")
        st.stop()

    # Persist field errors in session_state so they show after submit
    if "add_errors" not in st.session_state:
        st.session_state["add_errors"] = {}

    errors = st.session_state["add_errors"]

    with st.form("add_form"):
        new_id = st.text_input("id (primary key)", value="")
        show_field_error(errors, "id")

        new_name = st.text_input("name", value="")
        show_field_error(errors, "name")

        new_category = st.selectbox("category", CATEGORIES, index=0 if CATEGORIES else None)
        show_field_error(errors, "category")

        new_amount = st.number_input("amount", min_value=0.0, max_value=1_000_000.0, value=0.0, step=1.0)
        show_field_error(errors, "amount")

        new_status = st.selectbox("status", STATUSES, index=0 if STATUSES else None)
        show_field_error(errors, "status")

        new_valid_id = st.selectbox("valid_id", VALID_IDS, index=0 if VALID_IDS else None)
        show_field_error(errors, "valid_id")

        submitted = st.form_submit_button("Add")

    if submitted:
        actor = get_actor()
        new_obj = {
            "id": new_id.strip(),
            "name": new_name.strip(),
            "category": new_category,
            "amount": float(new_amount),
            "status": new_status,
            "valid_id": new_valid_id,
        }

        # Duplicate check
        exists = query_df(
            f"SELECT 1 FROM {MAIN_TABLE} WHERE id = ? AND is_deleted = false LIMIT 1",
            [new_obj["id"]],
        )
        if not exists.empty:
            msg = "ID already exists. Use Edit instead."
            st.session_state["add_errors"] = {"id": "Already exists"}
            st.error(msg)
            write_audit("INSERT", new_obj["id"], actor, None, new_obj, False, msg)
        else:
            field_errors = validate_row_fields(new_obj)
            st.session_state["add_errors"] = field_errors

            if field_errors:
                msg = "Validation failed: " + "; ".join([f"{k}={v}" for k, v in field_errors.items()])
                st.error("Please fix the highlighted fields.")
                write_audit("INSERT", new_obj["id"], actor, None, new_obj, False, msg)
            else:
                exec_sql(
                    f"""
                    INSERT INTO {MAIN_TABLE}
                    (id, name, category, amount, status, valid_id, is_deleted, updated_at, updated_by)
                    VALUES (?, ?, ?, ?, ?, ?, false, current_timestamp(), ?)
                    """,
                    [
                        new_obj["id"],
                        new_obj["name"],
                        new_obj["category"],
                        new_obj["amount"],
                        new_obj["status"],
                        new_obj["valid_id"],
                        actor,
                    ],
                )
                write_audit("INSERT", new_obj["id"], actor, None, new_obj, True, "OK")
                st.success("Record added.")
                st.session_state["add_errors"] = {}
                clear_caches()

# ===========================
# PAGE: EDIT
# ===========================
elif page == "Edit":
    st.subheader("Edit record (Update)")

    if not is_editor:
        st.warning("Viewer role: Edit is disabled.")
        st.stop()

    if "edit_errors" not in st.session_state:
        st.session_state["edit_errors"] = {}

    errors = st.session_state["edit_errors"]

    ids_df = query_df(f"SELECT id FROM {MAIN_TABLE} WHERE is_deleted = false ORDER BY id")
    if ids_df.empty:
        st.info("No active records to edit.")
        st.stop()

    edit_id = st.selectbox("Select id", ids_df["id"].tolist(), key="edit_id")

    # Day 5: Load updated_at for optimistic concurrency
    current = query_df(
        f"""
        SELECT id, name, category, amount, status, valid_id, updated_at
        FROM {MAIN_TABLE}
        WHERE id = ? AND is_deleted = false
        """,
        [edit_id],
    )

    if current.empty:
        st.error("Record not found.")
        st.stop()

    cur = current.iloc[0].to_dict()
    cur_updated_at = cur["updated_at"]

    with st.form("edit_form"):
        e_name = st.text_input("name", value=cur["name"])
        show_field_error(errors, "name")

        e_category = st.selectbox(
            "category",
            CATEGORIES,
            index=CATEGORIES.index(cur["category"]) if cur["category"] in CATEGORIES else 0,
        )
        show_field_error(errors, "category")

        e_amount = st.number_input(
            "amount", min_value=0.0, max_value=1_000_000.0, value=float(cur["amount"]), step=1.0
        )
        show_field_error(errors, "amount")

        e_status = st.selectbox(
            "status",
            STATUSES,
            index=STATUSES.index(cur["status"]) if cur["status"] in STATUSES else 0,
        )
        show_field_error(errors, "status")

        e_valid_id = st.selectbox(
            "valid_id",
            VALID_IDS,
            index=VALID_IDS.index(cur["valid_id"]) if cur["valid_id"] in VALID_IDS else 0,
        )
        show_field_error(errors, "valid_id")

        submitted_edit = st.form_submit_button("Save changes")

    if submitted_edit:
        actor = get_actor()
        new_obj = {
            "id": edit_id,
            "name": e_name.strip(),
            "category": e_category,
            "amount": float(e_amount),
            "status": e_status,
            "valid_id": e_valid_id,
        }

        field_errors = validate_row_fields(new_obj)
        st.session_state["edit_errors"] = field_errors

        if field_errors:
            msg = "Validation failed: " + "; ".join([f"{k}={v}" for k, v in field_errors.items()])
            st.error("Please fix the highlighted fields.")
            write_audit("UPDATE", edit_id, actor, cur, new_obj, False, msg)
        else:
            # Day 5: Optimistic concurrency check:
            # Update only if updated_at hasn't changed since we loaded the record.
            exec_sql(
                f"""
                UPDATE {MAIN_TABLE}
                SET name = ?, category = ?, amount = ?, status = ?, valid_id = ?,
                    updated_at = current_timestamp(), updated_by = ?
                WHERE id = ? AND is_deleted = false AND updated_at = ?
                """,
                [
                    new_obj["name"],
                    new_obj["category"],
                    new_obj["amount"],
                    new_obj["status"],
                    new_obj["valid_id"],
                    actor,
                    edit_id,
                    cur_updated_at,
                ],
            )

            # Verify whether update actually happened by re-reading updated_at
            after = query_df(
                f"SELECT updated_at, updated_by FROM {MAIN_TABLE} WHERE id = ? AND is_deleted = false",
                [edit_id],
            )

            if after.empty:
                msg = "Record disappeared or was deleted."
                st.error(msg)
                write_audit("UPDATE", edit_id, actor, cur, new_obj, False, msg)
            else:
                new_updated_at = after.iloc[0]["updated_at"]
                new_updated_by = after.iloc[0]["updated_by"]

                if new_updated_at == cur_updated_at:
                    # No change → someone else modified it first or concurrency condition failed.
                    msg = "Concurrency conflict: record was modified by another user. Reload and try again."
                    st.error(msg)
                    write_audit("UPDATE", edit_id, actor, cur, new_obj, False, msg)
                else:
                    write_audit("UPDATE", edit_id, actor, cur, new_obj, True, f"OK (updated_by={new_updated_by})")
                    st.success("Record updated.")
                    st.session_state["edit_errors"] = {}
                    clear_caches()

# ===========================
# PAGE: DELETE
# ===========================
elif page == "Delete":
    st.subheader("Delete record (Soft delete)")

    if not is_editor:
        st.warning("Viewer role: Delete is disabled.")
        st.stop()

    ids_df = query_df(f"SELECT id FROM {MAIN_TABLE} WHERE is_deleted = false ORDER BY id")
    if ids_df.empty:
        st.info("No active records to delete.")
        st.stop()

    del_id = st.selectbox("Select id to delete", ids_df["id"].tolist(), key="del_id")
    confirm = st.checkbox("I understand this will mark the record as deleted.", key="del_confirm")

    if st.button("Delete", disabled=not confirm):
        actor = get_actor()
        old = query_df(
            f"""
            SELECT id, name, category, amount, status, valid_id
            FROM {MAIN_TABLE}
            WHERE id = ? AND is_deleted = false
            """,
            [del_id],
        )
        old_obj = old.iloc[0].to_dict() if not old.empty else None

        exec_sql(
            f"""
            UPDATE {MAIN_TABLE}
            SET is_deleted = true, updated_at = current_timestamp(), updated_by = ?
            WHERE id = ? AND is_deleted = false
            """,
            [actor, del_id],
        )

        write_audit("DELETE", del_id, actor, old_obj, None, True, "OK")
        st.success("Record deleted (soft).")
        clear_caches()

# ===========================
# PAGE: AUDIT
# ===========================
elif page == "Audit":
    st.subheader("Audit log (last 100 events)")

    audit_df = query_df(
        f"""
        SELECT event_ts, event_type, row_id, actor, validation_ok, validation_message
        FROM {AUDIT_TABLE}
        ORDER BY event_ts DESC
        LIMIT 100
        """
    )

    st.dataframe(audit_df, use_container_width=True, hide_index=True)
    st.caption("Audit shows both successful and blocked actions, plus concurrency conflicts.")
