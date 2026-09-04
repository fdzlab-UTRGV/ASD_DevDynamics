"""
modules/holdout_loader.py
─────────────────────────────────────────────────────────────────────────────
Loader for the out-of-sample HOLDOUT cohort (Figure 3C ridge regression).

The holdout (e.g., the SSC cohort) uses DIFFERENT column conventions than the
main SPARK discovery files:

  - Instrument prefixes differ:
        discovery            holdout (SSC)
        dcdq.                dcdq_raw.
        rbsr.                rbs_r_raw.
        scq.                 scq_current_raw.
        cbcl_6_18.           cbcl_6_18.        (same prefix)
        cbcl_1_5.            cbcl_2_5.         (different name)

  - A few RBS-R item names differ slightly / contain source typos:
        schema item                       holdout header
        q08_hits_self_against_object  →   q08_hits_self_object
        q09_hits_self_with_object     →   q09_hits_self_object
        q28_communication             →   q28_communicatiion
        q06_sensory                   →   q06_sensory          (same)

  - CBCL uses *_total (raw counts) instead of *_raw_score:
        schema suffix                     holdout suffix
        internalizing_problems_raw_score → internalizing_problems_total
        ...                                  ...

This module re-uses the SAME schema domain definitions and reverse-scoring as
the discovery loaders, so the scored output columns
(dcdq_Coordination, rbs_Sensory, scq_Social, cbcl_Internalizing, ...) are
IDENTICAL in name to the discovery side. That is what lets the trained ridge
model apply to the holdout without any retraining or column remapping.

The output is a single merged DataFrame indexed by person_id, ready to pass
straight into modules.ridge.run_oos_ridge as the `holdout` argument.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modules.schema import DCDQ, RBS, SCQ, SCQ_REVERSED
from modules.loader import read_csv_chunked, safe_float


# ─────────────────────────────────────────────────────────────────────────────
# Holdout column conventions

# Instrument prefixes used by the holdout (SSC) raw files.
HOLDOUT_PREFIX = {
    "dcdq": "dcdq_raw.",
    "rbs":  "rbs_r_raw.",
    "scq":  "scq_current_raw.",
}

# RBS-R item-name aliases: schema name → holdout header name.
# Only items that differ need entries; everything else matches verbatim.
RBS_ITEM_ALIASES = {
    "q08_hits_self_against_object": "q08_hits_self_object",
    "q09_hits_self_with_object":    "q09_hits_self_object",
    "q28_communication":            "q28_communicatiion",  # source typo preserved
}

# CBCL forms in the holdout. Maps unified domain → holdout *_total column suffix.
# We use *_total (raw symptom counts), matching the raw-score scale the
# discovery analysis used.
HOLDOUT_CBCL_FORMS = {
    "cbcl_6_18": {
        "prefix": "cbcl_6_18.",
        "domains": {
            "Internalizing": "internalizing_problems_total",
            "Externalizing": "externalizing_problems_total",
            "Anxious/Dep.":  "anxious_depressed_total",
            "Withdrawn":     "withdrawn_total",
            "Somatic":       "somatic_complaints_total",
            "Attention":     "attention_problems_total",
            "Aggressive":    "aggressive_behavior_total",
            "Rule-Breaking": "rule_breaking_total",
            "Social Prob.":  "social_problems_total",
            "Thought Prob.": "thought_problems_total",
            "ADHD":          "add_adhd_total",
        },
    },
    "cbcl_2_5": {
        "prefix": "cbcl_2_5.",
        "domains": {
            # 2-5 form: no Social Prob., Thought Prob., Rule-Breaking
            "Internalizing": "internalizing_problems_total",
            "Externalizing": "externalizing_problems_total",
            "Anxious/Dep.":  "anxious_depressed_total",
            "Withdrawn":     "withdrawn_total",
            "Somatic":       "somatic_complaints_total",
            "Attention":     "attention_problems_total",
            "Aggressive":    "aggressive_behavior_total",
            "ADHD":          "add_adhd_total",
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Instrument detection

def _detect_instrument(df: pd.DataFrame) -> str | None:
    """Identify which instrument a holdout file contains, from its columns."""
    cols = list(df.columns)
    col_str = " ".join(cols)
    if "dcdq_raw." in col_str:
        return "dcdq"
    if "rbs_r_raw." in col_str:
        return "rbs"
    if "scq_current_raw." in col_str:
        return "scq"
    if "cbcl_6_18." in col_str:
        return "cbcl_6_18"
    if "cbcl_2_5." in col_str:
        return "cbcl_2_5"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-instrument scorers (mirror discovery loaders, holdout column names)

def _score_dcdq(df: pd.DataFrame) -> pd.DataFrame:
    """Score DCDQ. Inverted: higher raw = better, so flip before averaging."""
    prefix = HOLDOUT_PREFIX["dcdq"]
    lo, hi = DCDQ["score_range"]
    records = []
    for _, row in df.iterrows():
        pid = str(row["person_id"]).strip()
        rec = {"person_id": pid}
        for domain, items in DCDQ["domains"].items():
            vals = []
            for item in items:
                v = safe_float(row.get(prefix + item))
                if v is not None:
                    vals.append(hi - v + lo)
            rec[f"dcdq_{domain}"] = float(np.mean(vals)) if vals else None
        records.append(rec)
    df_out = pd.DataFrame(records)
    if df_out.empty or "person_id" not in df_out.columns:
        return pd.DataFrame()
    return df_out.set_index("person_id")


def _score_rbs(df: pd.DataFrame) -> pd.DataFrame:
    """Score RBS-R using schema domains, applying item-name aliases."""
    prefix = HOLDOUT_PREFIX["rbs"]
    records = []
    for _, row in df.iterrows():
        pid = str(row["person_id"]).strip()
        rec = {"person_id": pid}
        for domain, items in RBS["domains"].items():
            vals = []
            for item in items:
                header_item = RBS_ITEM_ALIASES.get(item, item)
                v = safe_float(row.get(prefix + header_item))
                if v is not None:
                    vals.append(v)
            rec[f"rbs_{domain}"] = float(np.mean(vals)) if vals else None
        records.append(rec)
    df_out = pd.DataFrame(records)
    if df_out.empty or "person_id" not in df_out.columns:
        return pd.DataFrame()
    return df_out.set_index("person_id")


def _coerce_binary(val):
    """
    Coerce an SCQ response to 0/1, handling the codings seen across cohorts:
      - numeric 0/1                → as-is
      - numeric 1/2 (SSC 'current') → 1→0, 2→1  (2 = atypical/endorsed)
      - text 'yes'/'no', 'y'/'n'   → yes=1, no=0
      - text 'true'/'false'        → true=1, false=0
    Returns a float in {0.0, 1.0} or None if unrecognized/missing.

    NOTE on direction: SCQ items score 1 for the ASD-atypical response. For the
    1/2 coding, 2 is the atypical/endorsed level, so 2→1. If a specific SSC
    export uses the opposite convention this is the single place to adjust.
    """
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    # numeric path
    try:
        f = float(val)
        if np.isnan(f):
            return None
        if f in (0.0, 1.0):
            return f
        if f == 2.0:          # 1/2 coding → map 2 to 1
            return 1.0
        # any other numeric (e.g. already a proportion) — clamp to 0/1 sense
        return 1.0 if f >= 1.5 else 0.0
    except (ValueError, TypeError):
        pass
    # text path
    s = str(val).strip().lower()
    if s in ("1", "yes", "y", "true", "t"):
        return 1.0
    if s in ("0", "no", "n", "false", "f"):
        return 0.0
    return None


def _score_scq(df: pd.DataFrame) -> pd.DataFrame:
    """Score SCQ, applying reverse-scoring to reversed items (same as discovery).

    Robust to the different value codings SSC 'current' SCQ can use (0/1, 1/2,
    yes/no) via _coerce_binary, so string- or 1/2-coded files still score
    instead of collapsing to all-NaN.
    """
    prefix = HOLDOUT_PREFIX["scq"]
    records = []
    for _, row in df.iterrows():
        pid = str(row["person_id"]).strip()
        rec = {"person_id": pid}
        for domain, items in SCQ["domains"].items():
            vals = []
            for item in items:
                v = _coerce_binary(row.get(prefix + item))
                if v is None:
                    continue
                if item in SCQ_REVERSED:
                    v = 1.0 - v
                vals.append(v)
            rec[f"scq_{domain}"] = float(np.mean(vals)) if vals else None
        records.append(rec)
    df_out = pd.DataFrame(records)
    if df_out.empty or "person_id" not in df_out.columns:
        return pd.DataFrame()
    return df_out.set_index("person_id")


def _score_cbcl(df: pd.DataFrame, form_key: str) -> pd.DataFrame:
    """Score one CBCL form (6_18 or 2_5) using *_total raw counts."""
    form   = HOLDOUT_CBCL_FORMS[form_key]
    prefix = form["prefix"]
    records = []
    for _, row in df.iterrows():
        pid = str(row["person_id"]).strip()
        rec = {"person_id": pid, "_cbcl_form": form_key}
        for domain, suffix in form["domains"].items():
            v = safe_float(row.get(prefix + suffix))
            rec[f"cbcl_{domain}"] = v
        records.append(rec)
    df_out = pd.DataFrame(records)
    if df_out.empty or "person_id" not in df_out.columns:
        return pd.DataFrame()
    return df_out.set_index("person_id")


# ─────────────────────────────────────────────────────────────────────────────
# CBCL merge with 6-18 priority

def _merge_cbcl(frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Merge CBCL 6-18 and 2-5 frames, preferring 6-18 where a person has both
    (matching the discovery loader's priority rule).
    """
    by_form = {fk: f for fk, f in frames}
    out = {}

    # Start with 2-5 (lower priority), then overwrite with 6-18.
    for fk in ("cbcl_2_5", "cbcl_6_18"):
        if fk not in by_form:
            continue
        f = by_form[fk]
        for pid, row in f.iterrows():
            if fk == "cbcl_2_5" and pid in out and out[pid].get("_cbcl_form") == "cbcl_6_18":
                continue
            out[pid] = row.to_dict()

    if not out:
        return pd.DataFrame()
    merged = pd.DataFrame.from_dict(out, orient="index").rename_axis("person_id")
    if "_cbcl_form" in merged.columns:
        merged = merged.drop(columns=["_cbcl_form"])
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point

def load_holdout(paths: list[str | Path]) -> tuple[pd.DataFrame, dict]:
    """
    Load and score a set of raw holdout instrument files into a single merged
    DataFrame whose columns match the discovery scored columns.

    Parameters
    ----------
    paths : list of file paths (CSV/XLSX) for the holdout cohort.
            Any mix of DCDQ, RBS-R, SCQ, and CBCL (6-18 and/or 2-5) files.

    Returns
    -------
    (merged_df, summary)
      merged_df : DataFrame indexed by person_id with scored predictor and
                  outcome columns (dcdq_*, rbs_*, scq_*, cbcl_*).
      summary   : dict {instrument_label: n_patients} for status display.
    """
    scored: dict[str, pd.DataFrame] = {}
    cbcl_frames: list[tuple[str, pd.DataFrame]] = []
    summary: dict[str, int] = {}

    for path in paths:
        try:
            suffix = Path(path).suffix.lower()
            if suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                df = read_csv_chunked(path)
        except Exception as e:
            print(f"[holdout] could not read {Path(path).name}: {e}")
            continue

        if "person_id" not in df.columns:
            print(f"[holdout] {Path(path).name} has no person_id column — skipped")
            continue

        inst = _detect_instrument(df)
        if inst is None:
            print(f"[holdout] could not detect instrument for {Path(path).name}")
            continue

        if inst == "dcdq":
            s = _score_dcdq(df)
            scored["dcdq"] = s
            summary["DCDQ"] = len(s)
        elif inst == "rbs":
            s = _score_rbs(df)
            scored["rbs"] = s
            summary["RBS-R"] = len(s)
        elif inst == "scq":
            s = _score_scq(df)
            scored["scq"] = s
            summary["SCQ"] = len(s)
        elif inst in ("cbcl_6_18", "cbcl_2_5"):
            s = _score_cbcl(df, inst)
            cbcl_frames.append((inst, s))
            summary[inst.upper().replace("_", "-")] = len(s)

    # Merge CBCL forms (6-18 priority), then drop the helper column
    if cbcl_frames:
        scored["cbcl"] = _merge_cbcl(cbcl_frames)

    if not scored:
        return pd.DataFrame(), {}

    # Outer-join all scored instruments on person_id
    merged = None
    for name, df in scored.items():
        if merged is None:
            merged = df.copy()
        else:
            new_cols = [c for c in df.columns if c not in merged.columns]
            if new_cols:
                merged = merged.join(df[new_cols], how="outer")

    return merged, summary
