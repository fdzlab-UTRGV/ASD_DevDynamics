"""
modules/cohort_loader.py
─────────────────────────────────────────────────────────────────────────────
General "second cohort" loader for REPLICATION analyses (Option A).

Purpose
-------
Load an independent cohort (e.g. SSC) from its raw instrument files, score it
with the SAME definitions as the discovery sample, and produce a merged
DataFrame that can be run straight through the discovery-side analyses
(√ΔR² hubness, correlations, PCA). Because √ΔR² is a scale-free semi-partial
correlation, this within-cohort replication does not depend on the two cohorts
sharing a measurement scale — unlike out-of-sample ridge transfer.

What it produces
----------------
The same scored predictor / outcome columns as the discovery merge:
    dcdq_*, rbs_*, scq_*, cbcl_*   (via modules.holdout_loader scorers)
plus unified covariate columns matching discovery:
    sex, age_months, age_years, nviq, fsiq, viq

Header robustness
-----------------
Instrument items are matched via holdout_loader (which already handles the SSC
prefixes and item-name aliases). Covariates are matched with a candidate-list
resolver that tries many header spellings, and the loader RETURNS a mapping
report so the UI can show what matched and let the user hand-map anything that
did not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modules.loader import read_csv_chunked, safe_float
from modules.schema import NUMERIC_COVARIATES
from modules.holdout_loader import (
    _detect_instrument,
    _score_dcdq,
    _score_rbs,
    _score_scq,
    _score_cbcl,
    _merge_cbcl,
)


# ─────────────────────────────────────────────────────────────────────────────
# Covariate header candidates (discovery names first, then SSC, then bare)

# field → ordered list of candidate header names (substring-insensitive match
# is applied on top of exact match; see _resolve_covariate).
COVARIATE_CANDIDATES: dict[str, list[str]] = {
    "sex": [
        "core_descriptive_variables.sex",
        "ssc_core_descriptive.sex",
        "sex", "gender",
    ],
    "age_months": [
        "core_descriptive_variables.age_at_registration_months",
        "iq.age_test_date_months",
        "ssc_core_descriptive.age_at_ados",   # SSC: months at ADOS
        "age_at_eval_months", "age_months", "age_at_ados",
    ],
    "age_years": [
        "core_descriptive_variables.age_at_registration_years",
        "age_at_eval_years", "age_years",
    ],
    "nviq": [
        "core_descriptive_variables.nviq",
        "ssc_core_descriptive.ssc_diagnosis_nonverbal_iq",
        "iq.nviq_score", "iq.nviq", "nviq", "nonverbal_iq",
    ],
    "fsiq": [
        "core_descriptive_variables.fsiq",
        "ssc_core_descriptive.ssc_diagnosis_full_scale_iq",
        "iq.fsiq_score", "iq.fsiq", "fsiq", "full_scale_iq",
    ],
    "viq": [
        "core_descriptive_variables.viq",
        "ssc_core_descriptive.ssc_diagnosis_verbal_iq",
        "iq.viq_score", "iq.viq", "viq", "verbal_iq",
    ],
}

# The covariates most analyses adjust for. Used to decide "matched enough".
CORE_COVARIATES = ["sex", "age_months", "nviq"]


def _looks_like_covariate_file(df: pd.DataFrame) -> bool:
    """Heuristic: a file is a covariate/descriptive file if it has sex or IQ."""
    cols = " ".join(df.columns).lower()
    return any(k in cols for k in ("sex", "nonverbal_iq", "descriptive",
                                   "age_at", "fsiq", "nviq"))


def _resolve_covariate(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the actual column in df that best matches one of `candidates`.
    Tries exact match first, then case-insensitive exact, then case-insensitive
    substring (candidate contained in a column name).
    """
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    # exact
    for cand in candidates:
        if cand in df.columns:
            return cand
    # case-insensitive exact
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # case-insensitive substring (candidate's last token in a column).
    # Require the tail to be reasonably specific (len >= 3) and prefer the
    # column whose name contains the tail as a word-boundary-ish match to avoid
    # e.g. "verbal_iq" matching inside "nonverbal_iq".
    for cand in candidates:
        tail = cand.split(".")[-1].lower()
        if len(tail) < 3:
            continue
        for c in cols:
            cl = c.lower()
            # exact tail as a suffix or delimited token beats loose containment
            if cl.endswith(tail) or f"_{tail}" in cl or f".{tail}" in cl:
                # guard: don't let "verbal_iq" match "nonverbal_iq"
                if tail == "verbal_iq" and "nonverbal" in cl:
                    continue
                return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Covariate loading

def _load_covariates(
    df: pd.DataFrame,
    overrides: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Extract unified covariate columns from a descriptive/covariate DataFrame.

    Parameters
    ----------
    df        : the covariate/descriptive file
    overrides : {field: actual_column_name} manual mappings from the UI that
                take precedence over auto-detection.

    Returns
    -------
    (cov_df, report)
      cov_df : DataFrame indexed by person_id with unified covariate columns.
      report : {field: matched_column_or_None} for every covariate field.
    """
    overrides = overrides or {}
    if "person_id" not in df.columns:
        return pd.DataFrame(), {f: None for f in COVARIATE_CANDIDATES}

    report: dict[str, str | None] = {}
    resolved: dict[str, str] = {}
    for field, cands in COVARIATE_CANDIDATES.items():
        col = overrides.get(field) or _resolve_covariate(df, cands)
        report[field] = col
        if col is not None and col in df.columns:
            resolved[field] = col

    records = {}
    for _, row in df.iterrows():
        pid = str(row["person_id"]).strip()
        rec = records.setdefault(pid, {})
        for field, col in resolved.items():
            if field in rec:
                continue
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            if field in NUMERIC_COVARIATES:
                val = safe_float(val)
            rec[field] = val

    # Derive age_years from age_months if only months present
    if "age_months" in resolved and "age_years" not in resolved:
        for pid, rec in records.items():
            if "age_months" in rec and rec["age_months"] is not None:
                rec["age_years"] = rec["age_months"] / 12.0
        report["age_years"] = "(derived from age_months)"

    cov_df = (pd.DataFrame.from_dict(records, orient="index")
              .rename_axis("person_id")) if records else pd.DataFrame()
    return cov_df, report


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point

def load_cohort(
    paths: list[str | Path],
    covariate_overrides: dict[str, str] | None = None,
) -> dict:
    """
    Load and score a full second cohort from raw instrument + covariate files.

    Parameters
    ----------
    paths               : list of CSV/XLSX file paths (instruments + covariates)
    covariate_overrides : {field: column_name} manual mappings from the UI

    Returns
    -------
    dict with keys:
      merged        DataFrame (person_id index) of scored predictors, outcomes,
                    and unified covariates — ready for the discovery analyses.
      instrument_summary  {label: n} scored per instrument
      covariate_report    {field: matched_column_or_None}
      covariate_columns   list of headers in the covariate file (for UI mapping)
      unmatched_core      list of CORE_COVARIATES that did not match
      error               str, set only on fatal error
    """
    scored: dict[str, pd.DataFrame] = {}
    cbcl_frames: list[tuple[str, pd.DataFrame]] = []
    instrument_summary: dict[str, int] = {}
    covariate_report: dict = {}
    covariate_columns: list[str] = []

    cov_file_df = None

    for path in paths:
        try:
            suffix = Path(path).suffix.lower()
            if suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                df = read_csv_chunked(path)
        except Exception as e:
            print(f"[cohort] could not read {Path(path).name}: {e}")
            continue

        if "person_id" not in df.columns:
            print(f"[cohort] {Path(path).name} has no person_id — skipped")
            continue

        # Normalize person_id to a clean string so joins are robust to
        # int-vs-string dtype differences across files (a common cause of an
        # instrument silently failing to merge with the outcomes).
        df["person_id"] = df["person_id"].astype(str).str.strip()

        inst = _detect_instrument(df)

        if inst == "dcdq":
            s = _score_dcdq(df); scored["dcdq"] = s
            instrument_summary["DCDQ"] = len(s)
        elif inst == "rbs":
            s = _score_rbs(df); scored["rbs"] = s
            instrument_summary["RBS-R"] = len(s)
        elif inst == "scq":
            s = _score_scq(df); scored["scq"] = s
            instrument_summary["SCQ"] = len(s)
        elif inst in ("cbcl_6_18", "cbcl_2_5"):
            s = _score_cbcl(df, inst); cbcl_frames.append((inst, s))
            instrument_summary[inst.upper().replace("_", "-")] = len(s)
        elif _looks_like_covariate_file(df):
            cov_file_df = df
            covariate_columns = list(df.columns)
        else:
            print(f"[cohort] unrecognized file {Path(path).name} — skipped")

    if cbcl_frames:
        scored["cbcl"] = _merge_cbcl(cbcl_frames)

    if not scored:
        return {"error": ("No recognized instrument files. Expected DCDQ, "
                          "RBS-R, SCQ, or CBCL raw files.")}

    # ── person_id overlap diagnostics ────────────────────────────────────────
    # Zero overlap between an instrument and the outcome (CBCL) file means that
    # instrument's predictors will have 0 complete cases in every regression —
    # showing up as "0/0" in results. Surface this so the user can spot an ID
    # format mismatch (suffixes, leading zeros) rather than silently dropping.
    id_sets = {name: set(df.index.astype(str)) for name, df in scored.items()}
    overlap_report = {}
    ref_name = "cbcl" if "cbcl" in id_sets else next(iter(id_sets))
    ref_ids = id_sets.get(ref_name, set())
    for name, ids in id_sets.items():
        inter = len(ids & ref_ids)
        overlap_report[name] = {
            "n": len(ids),
            "overlap_with_outcomes": inter,
            "pct": (100.0 * inter / len(ids)) if ids else 0.0,
        }

    # Merge scored instruments
    merged = None
    for _name, df in scored.items():
        if merged is None:
            merged = df.copy()
        else:
            new_cols = [c for c in df.columns if c not in merged.columns]
            if new_cols:
                merged = merged.join(df[new_cols], how="outer")

    # Covariates
    unmatched_core = list(CORE_COVARIATES)
    if cov_file_df is not None:
        cov_df, covariate_report = _load_covariates(cov_file_df,
                                                    covariate_overrides)
        if not cov_df.empty:
            new_cols = [c for c in cov_df.columns if c not in merged.columns]
            merged = merged.join(cov_df[new_cols], how="left")
        unmatched_core = [c for c in CORE_COVARIATES
                          if not covariate_report.get(c)]

    return {
        "merged":             merged,
        "instrument_summary": instrument_summary,
        "covariate_report":   covariate_report,
        "covariate_columns":  covariate_columns,
        "unmatched_core":     unmatched_core,
        "overlap_report":     overlap_report,
    }
