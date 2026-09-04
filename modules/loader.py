"""
loader.py — Streaming CSV loader and data processing for SPARK dataset.
All functions return DataFrames or dicts keyed by person_id.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from modules.schema import (
    DCDQ, RBS, SCQ, ADOS, CBCL, COVARIATE_FIELDS, NUMERIC_COVARIATES,
    ADOS_MODULES, CBCL_MAP, SCQ_REVERSED
)


# ── Utilities ──────────────────────────────────────────────────────────────────

def is_proband(pid) -> bool:
    """Exclude sibling IDs (contain '.s'). Robust to non-string dtypes:
    person_id may be read as int by pandas when all IDs are numeric, which
    previously caused the entire file to be filtered out."""
    return ".s" not in str(pid)


def safe_float(val) -> float | None:
    """Parse a value to float, returning None for missing/invalid."""
    if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None


def recode_ados(val) -> float | None:
    """ADOS algorithm recoding: 3→2, 7/8/9→0, others pass through."""
    v = safe_float(val)
    if v is None:
        return None
    if v in (7, 8, 9):
        return 0.0
    if v == 3:
        return 2.0
    return max(0.0, min(2.0, v))


def domain_avg(row: pd.Series, prefix: str, items: list,
               transform=None) -> float | None:
    """Compute mean of items in a row, applying optional transform."""
    vals = []
    for item in items:
        col = prefix + item
        if col not in row.index:
            continue
        v = safe_float(row[col])
        if v is None:
            continue
        if transform:
            v = transform(v)
            if v is None:
                continue
        vals.append(v)
    return float(np.mean(vals)) if vals else None


def read_csv_chunked(path: str | Path, chunksize: int = 50_000) -> pd.DataFrame:
    """Read a potentially large CSV/TSV in chunks, filtering probands."""
    path = Path(path)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline()
    if first.count("\t") > first.count(","):
        sep = "\t"

    chunks = []
    for chunk in pd.read_csv(path, sep=sep, chunksize=chunksize,
                              low_memory=False, encoding="utf-8"):
        if "person_id" not in chunk.columns:
            raise ValueError(f"No 'person_id' column found in {path.name}")
        chunk = chunk[chunk["person_id"].apply(is_proband)]
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ── Scale loaders ──────────────────────────────────────────────────────────────

def load_dcdq(path: str | Path) -> pd.DataFrame:
    """
    Load DCDQ CSV. Returns DataFrame with person_id + domain columns.
    DCDQ scores are inverted (higher raw = better → invert before averaging).
    """
    df = read_csv_chunked(path)
    prefix = DCDQ["prefix"]
    lo, hi = DCDQ["score_range"]
    records = []
    for _, row in df.iterrows():
        pid = str(row["person_id"]).strip()
        rec = {"person_id": pid}
        for domain, items in DCDQ["domains"].items():
            vals = []
            for item in items:
                col = prefix + item
                v = safe_float(row.get(col))
                if v is not None:
                    vals.append(hi - v + lo)
            rec[f"dcdq_{domain}"] = float(np.mean(vals)) if vals else None
        # Preserve eval age for forward-gap analysis
        age_m = safe_float(row.get(prefix + "age_at_eval_months"))
        if age_m is None:
            age_y = safe_float(row.get(prefix + "age_at_eval_years"))
            age_m = age_y * 12.0 if age_y is not None else None
        rec["dcdq_age_months"] = age_m
        records.append(rec)
    return pd.DataFrame(records).set_index("person_id")


def load_rbs(path: str | Path) -> pd.DataFrame:
    """Load RBS-R CSV. Returns DataFrame with person_id + domain columns."""
    df = read_csv_chunked(path)
    prefix = RBS["prefix"]
    records = []
    for _, row in df.iterrows():
        pid = str(row["person_id"]).strip()
        rec = {"person_id": pid}
        for domain, items in RBS["domains"].items():
            avg = domain_avg(row, prefix, items)
            rec[f"rbs_{domain}"] = avg
        # Preserve eval age for forward-gap analysis
        age_m = safe_float(row.get(prefix + "age_at_eval_months"))
        if age_m is None:
            age_y = safe_float(row.get(prefix + "age_at_eval_years"))
            age_m = age_y * 12.0 if age_y is not None else None
        rec["rbs_age_months"] = age_m
        records.append(rec)
    return pd.DataFrame(records).set_index("person_id")


def load_scq(path: str | Path) -> pd.DataFrame:
    """
    Load SCQ CSV. Applies reversal to items where NO=1.
    Returns DataFrame with person_id + domain columns.
    """
    df = read_csv_chunked(path)
    prefix = SCQ["prefix"]
    records = []
    for _, row in df.iterrows():
        pid = str(row["person_id"]).strip()
        rec = {"person_id": pid}
        for domain, items in SCQ["domains"].items():
            vals = []
            for item in items:
                col = prefix + item
                v = safe_float(row.get(col))
                if v is None:
                    continue
                if item in SCQ_REVERSED:
                    v = 1.0 - v
                vals.append(v)
            rec[f"scq_{domain}"] = float(np.mean(vals)) if vals else None
        # Preserve eval age for forward-gap analysis
        age_m = safe_float(row.get(prefix + "age_at_eval_months"))
        if age_m is None:
            age_y = safe_float(row.get(prefix + "age_at_eval_years"))
            age_m = age_y * 12.0 if age_y is not None else None
        rec["scq_age_months"] = age_m
        records.append(rec)
    return pd.DataFrame(records).set_index("person_id")


def detect_ados_module(df: pd.DataFrame) -> str | None:
    """Detect which ADOS module a DataFrame contains by column prefix."""
    cols = set(df.columns)
    for mod_key, mod in ADOS_MODULES.items():
        prefix = mod["prefix"]
        if any(c.startswith(prefix) for c in cols):
            return mod_key
    return None


def load_ados(paths: list[str | Path]) -> pd.DataFrame:
    """
    Load one or more ADOS module files. Auto-detects module from columns.
    SA/RRB derived from raw items using revised algorithm (Gotham 2007/2008).

    Stores both:
      - Domain averages: ados_Social Affect, ados_RRB, ados_Play/Imag., ados_Other Behav.
      - Raw totals for CSS: _ados_raw_sa, _ados_raw_rrb (sum of recoded items)
      - Module key: _ados_module
      - Age at assessment (if available): ados_age_months

    CSS computation requires raw totals, not averages. Raw totals are stored
    with underscore prefix to distinguish them from domain averages.
    """
    all_records = {}

    for path in paths:
        df = read_csv_chunked(path)
        mod_key = detect_ados_module(df)
        if mod_key is None:
            print(f"Warning: could not detect ADOS module for {Path(path).name}")
            continue

        mod = ADOS_MODULES[mod_key]
        prefix = mod["prefix"]

        # Detect age column — SPARK uses various column names across modules
        # The actual column in SPARK ADOS files is prefix + "age_at_eval_months"
        age_cols = [
            prefix + "age_at_eval_months",   # actual SPARK ADOS-2 column name
            prefix + "age_at_eval_years",     # fallback (×12 below)
            prefix + "child_age",
            prefix + "age",
            "age_at_eval_months",
            "core_descriptive_variables.age_at_registration_months",
        ]

        for _, row in df.iterrows():
            pid = str(row["person_id"]).strip()

            # Domain averages (for fingerprint grid / mediation)
            sa    = domain_avg(row, prefix, mod["sa"],    recode_ados)
            rrb   = domain_avg(row, prefix, mod["rrb"],   recode_ados)
            play  = domain_avg(row, prefix, mod["play"],  recode_ados) if mod["play"] else None
            other = domain_avg(row, prefix, mod["other"], recode_ados) if mod["other"] else None

            if sa is None and rrb is None:
                continue

            # Raw totals for CSS lookup (sum, not average)
            raw_sa  = _domain_sum(row, prefix, mod["sa"],  recode_ados)
            raw_rrb = _domain_sum(row, prefix, mod["rrb"], recode_ados)

            # Age at assessment — try months first, then years×12
            age_m = None
            for ac in age_cols:
                v = safe_float(row.get(ac))
                if v is not None:
                    # If the column is years-based, convert to months
                    if "years" in ac:
                        v = v * 12
                    age_m = v
                    break

            all_records[pid] = {
                "ados_Social Affect": sa,
                "ados_RRB":           rrb,
                "ados_Play/Imag.":    play,
                "ados_Other Behav.":  other,
                "_ados_module":       mod_key,
                "_ados_raw_sa":       raw_sa,
                "_ados_raw_rrb":      raw_rrb,
                "ados_age_months":    age_m,
            }

    if not all_records:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(all_records, orient="index").rename_axis("person_id")


def _domain_sum(row: pd.Series, prefix: str, items: list,
                transform=None) -> float | None:
    """Compute SUM (not mean) of recoded items — needed for CSS lookup tables."""
    vals = []
    for item in items:
        col = prefix + item
        if col not in row.index:
            continue
        v = safe_float(row.get(col))
        if v is None:
            continue
        if transform:
            v = transform(v)
            if v is None:
                continue
        vals.append(v)
    return float(sum(vals)) if vals else None


def detect_cbcl_form(df: pd.DataFrame) -> str | None:
    cols = set(df.columns)
    if any(c.startswith("cbcl_1_5.") for c in cols):
        return "cbcl_1_5"
    if any(c.startswith("cbcl_6_18.") for c in cols):
        return "cbcl_6_18"
    return None


def load_cbcl(paths: list[str | Path]) -> pd.DataFrame:
    """
    Load CBCL 1-5 and/or 6-18 files. 6-18 takes priority over 1-5.
    Returns unified DataFrame with all domain columns (NaN for age-inappropriate subscales).
    """
    all_records = {}
    patient_form = {}

    for path in paths:
        df = read_csv_chunked(path)
        form_key = detect_cbcl_form(df)
        if form_key is None:
            print(f"Warning: could not detect CBCL form for {Path(path).name}")
            continue

        form = CBCL_MAP[form_key]
        prefix = form["prefix"]
        domain_map = form["domains"]

        for _, row in df.iterrows():
            pid = str(row["person_id"]).strip()
            # Priority: 6-18 over 1-5
            if patient_form.get(pid) == "cbcl_6_18" and form_key == "cbcl_1_5":
                continue
            if patient_form.get(pid) == "cbcl_1_5" and form_key == "cbcl_6_18":
                all_records[pid] = {}  # clear old 1-5 data
            if pid not in all_records:
                all_records[pid] = {}
            patient_form[pid] = form_key
            for domain, col_suffix in domain_map.items():
                col = prefix + col_suffix
                v = safe_float(row.get(col))
                all_records[pid][f"cbcl_{domain}"] = v
            # Preserve eval age for forward-gap analysis
            age_m = safe_float(row.get(prefix + "age_at_eval_months"))
            if age_m is None:
                age_y = safe_float(row.get(prefix + "age_at_eval_years"))
                age_m = age_y * 12.0 if age_y is not None else None
            if age_m is not None:
                all_records[pid]["cbcl_age_months"] = age_m

    if not all_records:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(all_records, orient="index").rename_axis("person_id")


def load_covariates(paths: list[str | Path]) -> pd.DataFrame:
    """
    Load covariate files (core_descriptive_variables, iq).
    Returns DataFrame with person_id + sex, age, fsiq, nviq, viq, etc.
    """
    all_records = {}

    def resolve(row, candidates):
        for col in candidates:
            if col in row.index and row[col] != "" and not (
                isinstance(row[col], float) and np.isnan(row[col])
            ):
                return row[col]
        return None

    for path in paths:
        df = read_csv_chunked(path)
        for _, row in df.iterrows():
            pid = str(row["person_id"]).strip()
            if pid not in all_records:
                all_records[pid] = {}
            for field, candidates in COVARIATE_FIELDS.items():
                if field in all_records[pid]:
                    continue  # already have it
                val = resolve(row, candidates)
                if val is None:
                    continue
                if field in NUMERIC_COVARIATES:
                    val = safe_float(val)
                all_records[pid][field] = val

    if not all_records:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(all_records, orient="index").rename_axis("person_id")


# ── Merge all scales ───────────────────────────────────────────────────────────

def merge_scales(scale_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all loaded scale DataFrames on person_id (outer join).
    scale_dfs: {"dcdq": df, "rbs": df, ...}
    Returns wide DataFrame with all domain columns.
    """
    dfs = [df for df in scale_dfs.values() if df is not None and not df.empty]
    if not dfs:
        return pd.DataFrame()
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="outer")
    return merged


def get_domain_columns(merged: pd.DataFrame) -> dict[str, list[str]]:
    """
    Returns dict mapping scale_key -> list of domain column names present in merged.
    """
    result = {}
    prefixes = {"dcdq": "dcdq_", "rbs": "rbs_", "scq": "scq_",
                "ados": "ados_", "cbcl": "cbcl_"}
    for scale, prefix in prefixes.items():
        cols = [c for c in merged.columns if c.startswith(prefix)]
        if cols:
            result[scale] = cols
    return result


def compute_stats(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and SD per domain column across all patients.
    Returns DataFrame with index=column_name, columns=['mean','sd','n'].
    Uses pd.to_numeric to handle Arrow-backed dtypes from parquet deserialization.
    """
    stats = {}
    for col in merged.columns:
        if col.startswith("_"):
            continue
        try:
            vals = pd.to_numeric(merged[col], errors="coerce").dropna()
        except Exception:
            continue
        if len(vals) == 0:
            continue
        stats[col] = {
            "mean": float(vals.mean()),
            "sd":   float(vals.std()) if len(vals) > 1 else 1.0,
            "n":    len(vals),
        }
    return pd.DataFrame(stats).T
