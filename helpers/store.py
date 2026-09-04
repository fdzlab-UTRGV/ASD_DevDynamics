"""
helpers/store.py
─────────────────────────────────────────────────────────────────────────────
Per-source data store helpers + on-demand merge.

Architecture:
- Each instrument has its own store (dcdq-store, rbs-store, etc.)
- Each store holds parquet-encoded base64 of that instrument's per-patient scores
- get_merged_data(stores) computes merged DataFrame on demand from source stores
- No "merged" store exists - merge is recomputed when analyses need it

This is the foundation that prevents the v105 problem of multiple writers
clobbering each other's state.
"""

import base64
import io
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Source data store IDs - one per instrument
SOURCE_STORE_IDS = [
    "dcdq-store",
    "rbs-store",
    "scq-store",
    "ados-store",
    "cbcl-store",
    "cov-store",
    "sensory-store",
    "css-store",
]


def df_to_store(df: pd.DataFrame) -> str:
    """Serialize DataFrame to base64-encoded parquet for storage in dcc.Store."""
    if df is None or df.empty:
        return None
    buf = io.BytesIO()
    df.to_parquet(buf, index=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def df_from_store(store_value) -> pd.DataFrame | None:
    """Deserialize base64-encoded parquet from a store value back to a DataFrame.

    Crucial: convert Arrow dtypes to numpy dtypes for downstream numpy/scipy/MOFA.
    """
    if not store_value:
        return None
    try:
        raw = base64.b64decode(store_value)
        df = pd.read_parquet(io.BytesIO(raw))
        # Force conversion of Arrow dtypes - prevents downstream errors
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        return df
    except Exception as e:
        print(f"[df_from_store] decode error: {e}")
        return None


def get_merged_data(
    dcdq=None, rbs=None, scq=None, ados=None, cbcl=None,
    cov=None, sensory=None, css=None,
) -> pd.DataFrame | None:
    """
    Compute the merged DataFrame on demand from source stores.

    Each argument is either None or a store_value (base64 parquet string).
    Returns a single DataFrame indexed by person_id with columns from all
    available instruments, or None if no data is loaded.
    """
    sources = {
        "dcdq":    df_from_store(dcdq),
        "rbs":     df_from_store(rbs),
        "scq":     df_from_store(scq),
        "ados":    df_from_store(ados),
        "cbcl":    df_from_store(cbcl),
        "cov":     df_from_store(cov),
        "sensory": df_from_store(sensory),
        "css":     df_from_store(css),
    }

    # Filter out empty sources
    sources = {k: v for k, v in sources.items() if v is not None and not v.empty}
    if not sources:
        return None

    # Outer-join all sources on the index (person_id)
    merged = None
    for name, df in sources.items():
        if merged is None:
            merged = df.copy()
        else:
            # Avoid duplicate columns by using join (which respects index)
            new_cols = [c for c in df.columns if c not in merged.columns]
            if new_cols:
                merged = merged.join(df[new_cols], how="outer")

    # ── Auto-compute ADOS Calibrated Severity Scores ─────────────────────────
    # The css_total / css_sa / css_rrb columns are paper outcomes (Figs 1-3).
    # When raw ADOS columns are present and CSS has not already been merged in,
    # compute CSS on the fly so analyses see the columns without a manual step.
    if merged is not None and "css_total" not in merged.columns:
        raw_ados = {"_ados_raw_sa", "_ados_raw_rrb", "_ados_module"}
        if raw_ados.issubset(merged.columns):
            try:
                from modules.stats import compute_css_for_merged
                css_df = compute_css_for_merged(merged)
                for col in ("css_sa", "css_rrb", "css_total"):
                    if col in css_df.columns:
                        merged[col] = css_df[col]
            except Exception as e:
                print(f"[get_merged_data] CSS computation skipped: {e}")

    return merged


def has_data(store_value) -> bool:
    """Quick check whether a store has data without full deserialization."""
    return store_value is not None and len(store_value) > 0


def patient_count(store_value) -> int:
    """Number of patients (rows) in a store. 0 if empty."""
    df = df_from_store(store_value)
    return 0 if df is None else len(df)
