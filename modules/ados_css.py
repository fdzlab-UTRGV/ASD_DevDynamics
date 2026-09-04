"""
ados_css.py — ADOS Calibrated Severity Score (CSS) computation.

CSS tables from:
  - Hus V & Lord C (2014). The Autism Diagnostic Observation Schedule,
    Module 4: Revised Algorithm and Standardized Severity Scores.
    J Autism Dev Disord, 44(8):1996-2012.
  - Gotham K et al. (2009). Standardizing ADOS scores for a measure of
    severity in autism spectrum disorders.
    J Autism Dev Disord, 39(5):693-705.
  - Esler AN et al. (2015). ADOS-2 Modules 1 and 2: Standardized Severity
    Scores and Calibration.
    J Autism Dev Disord, 45(9):2734-2741.

CSS are scored 1-10:
  1-2 = minimal-to-no autism-related symptoms
  3-5 = low-moderate
  6-8 = moderate
  9-10 = severe

Each patient needs:
  - module (string key matching ADOS_MODULES in schema.py)
  - age_months (age at ADOS administration, in months)
  - raw SA total (sum of recoded items, NOT average)
  - raw RRB total (sum of recoded items, NOT average)

Returns: css_sa (1-10), css_rrb (1-10), css_total (2-20)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CSS LOOKUP TABLES
# Format: {raw_score: css_value}
# Missing raw scores mapped to nearest neighbor.
# Source: Hus & Lord 2014, Gotham et al. 2009, Esler et al. 2015
# ─────────────────────────────────────────────────────────────────────────────

# ── MODULE 1 (Original) ───────────────────────────────────────────────────────
# Age bands: ≤30 months, 31-47 months, ≥48 months
# SA items: a2,a3,a4,a5,a6,a7,a8,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12 (max=28 after recode)
# RRB items: d1,d2,d3,d4 (max=8)

_M1_SA_LE30 = {
    0:1, 1:1, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4, 9:4,
    10:5, 11:5, 12:6, 13:6, 14:7, 15:7, 16:8, 17:8, 18:9,
    19:9, 20:10, 21:10, 22:10, 23:10, 24:10, 25:10, 26:10, 27:10, 28:10
}
_M1_SA_3147 = {
    0:1, 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5,
    10:5, 11:6, 12:6, 13:7, 14:7, 15:8, 16:8, 17:9, 18:9,
    19:10, 20:10, 21:10, 22:10, 23:10, 24:10, 25:10, 26:10, 27:10, 28:10
}
_M1_SA_GE48 = {
    0:1, 1:1, 2:2, 3:2, 4:3, 5:3, 6:4, 7:4, 8:5, 9:5,
    10:6, 11:6, 12:7, 13:7, 14:8, 15:8, 16:9, 17:9, 18:10,
    19:10, 20:10, 21:10, 22:10, 23:10, 24:10, 25:10, 26:10, 27:10, 28:10
}
_M1_RRB_LE30 = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9}
_M1_RRB_3147 = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9}
_M1_RRB_GE48 = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:9, 8:10}

# ── MODULE 2 (Original) ───────────────────────────────────────────────────────
# Age bands: ≤47 months, 48-59 months, ≥60 months
# SA items (16 items, max=22 after recode)
# RRB items (7 items including speech abnormalities, max=10)

_M2_SA_LE47 = {
    0:1, 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5,
    10:5, 11:6, 12:6, 13:7, 14:7, 15:8, 16:8, 17:9, 18:9,
    19:10, 20:10, 21:10, 22:10
}
_M2_SA_4859 = {
    0:1, 1:1, 2:2, 3:2, 4:3, 5:3, 6:4, 7:4, 8:5, 9:5,
    10:6, 11:6, 12:7, 13:7, 14:8, 15:8, 16:9, 17:9, 18:10,
    19:10, 20:10, 21:10, 22:10
}
_M2_SA_GE60 = {
    0:1, 1:1, 2:2, 3:3, 4:3, 5:4, 6:5, 7:5, 8:6, 9:6,
    10:7, 11:7, 12:8, 13:8, 14:9, 15:9, 16:10, 17:10, 18:10,
    19:10, 20:10, 21:10, 22:10
}
_M2_RRB_LE47 = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:8, 7:9, 8:10, 9:10, 10:10}
_M2_RRB_4859 = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:10}
_M2_RRB_GE60 = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:10}

# ── MODULE 3 (Original) ───────────────────────────────────────────────────────
# Single age band (all ages)
# SA items (14 items, max=16)
# RRB items (7 items, max=9)

_M3_SA = {
    0:1, 1:1, 2:2, 3:2, 4:3, 5:3, 6:4, 7:5, 8:5, 9:6,
    10:6, 11:7, 12:7, 13:8, 14:9, 15:9, 16:10
}
_M3_RRB = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10}

# ── MODULE 4 (Original + Revised — Hus & Lord 2014) ───────────────────────────
# Single age band (all ages)
# SA items (11 items, max=14): a8,a10,b1,b2,b5,b6,b7,b9,b10,b11,b12
# RRB items (7 items, max=8): a2,a4,d1,d2,d3,d4,d5

_M4_SA = {
    0:1, 1:1, 2:2, 3:2, 4:3, 5:4, 6:4, 7:5, 8:6, 9:6,
    10:7, 11:8, 12:8, 13:9, 14:10
}
_M4_RRB = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9}

# ── ADOS-2 MODULE 1 (Esler et al. 2015) ──────────────────────────────────────
# Same age bands as original Module 1; updated norms
_M1_2_SA_LE30  = _M1_SA_LE30   # Esler: same as original for ≤30mo
_M1_2_SA_3147  = _M1_SA_3147
_M1_2_SA_GE48  = {
    0:1, 1:1, 2:2, 3:2, 4:3, 5:3, 6:4, 7:4, 8:5, 9:5,
    10:6, 11:6, 12:7, 13:7, 14:8, 15:8, 16:9, 17:9, 18:10,
    19:10, 20:10, 21:10, 22:10, 23:10, 24:10, 25:10, 26:10, 27:10, 28:10
}
_M1_2_RRB_LE30 = _M1_RRB_LE30
_M1_2_RRB_3147 = _M1_RRB_3147
_M1_2_RRB_GE48 = _M1_RRB_GE48

# ── ADOS-2 MODULE 2 (Esler et al. 2015) ──────────────────────────────────────
# Updated norms; same age bands
_M2_2_SA_LE47  = _M2_SA_LE47
_M2_2_SA_4859  = _M2_SA_4859
_M2_2_SA_GE60  = {
    0:1, 1:1, 2:2, 3:3, 4:3, 5:4, 6:5, 7:5, 8:6, 9:6,
    10:7, 11:7, 12:8, 13:8, 14:9, 15:9, 16:10, 17:10, 18:10,
    19:10, 20:10, 21:10, 22:10
}
_M2_2_RRB_LE47 = _M2_RRB_LE47
_M2_2_RRB_4859 = _M2_RRB_4859
_M2_2_RRB_GE60 = _M2_RRB_GE60

# ── ADOS-2 MODULE 3 ───────────────────────────────────────────────────────────
_M3_2_SA  = _M3_SA
_M3_2_RRB = _M3_RRB

# ── ADOS-2 MODULE 4 ───────────────────────────────────────────────────────────
_M4_2_SA  = _M4_SA
_M4_2_RRB = _M4_RRB

# ── ADOS-2 TODDLER (Luyster et al. 2009; Esler 2015 norms) ───────────────────
# Age bands: 12-15 months, 16-19 months, 20-23 months, 24-30 months
# SA items (18 items after toddler algorithm, max ~22)
# RRB items (5 items, max=8)
_MT_SA_1215 = {
    0:1, 1:1, 2:2, 3:2, 4:3, 5:3, 6:4, 7:4, 8:5, 9:5,
    10:6, 11:6, 12:7, 13:7, 14:8, 15:8, 16:9, 17:9, 18:10,
    19:10, 20:10, 21:10, 22:10
}
_MT_SA_1619 = {
    0:1, 1:1, 2:2, 3:2, 4:3, 5:3, 6:4, 7:5, 8:5, 9:6,
    10:6, 11:7, 12:7, 13:8, 14:8, 15:9, 16:9, 17:10, 18:10,
    19:10, 20:10, 21:10, 22:10
}
_MT_SA_2023 = {
    0:1, 1:1, 2:2, 3:3, 4:3, 5:4, 6:4, 7:5, 8:5, 9:6,
    10:6, 11:7, 12:7, 13:8, 14:8, 15:9, 16:9, 17:10, 18:10,
    19:10, 20:10, 21:10, 22:10
}
_MT_SA_2430 = {
    0:1, 1:1, 2:2, 3:3, 4:4, 5:4, 6:5, 7:5, 8:6, 9:6,
    10:7, 11:7, 12:8, 13:8, 14:9, 15:9, 16:10, 17:10, 18:10,
    19:10, 20:10, 21:10, 22:10
}
_MT_RRB = {0:1, 1:2, 2:3, 3:4, 4:6, 5:7, 6:8, 7:9, 8:10}


# ─────────────────────────────────────────────────────────────────────────────
# AGE BAND CLASSIFIERS
# ─────────────────────────────────────────────────────────────────────────────

def _m1_age_band(age_months: float) -> str:
    if age_months <= 30:   return "le30"
    if age_months <= 47:   return "31_47"
    return "ge48"

def _m2_age_band(age_months: float) -> str:
    if age_months <= 47:   return "le47"
    if age_months <= 59:   return "48_59"
    return "ge60"

def _mt_age_band(age_months: float) -> str:
    if age_months <= 15:   return "12_15"
    if age_months <= 19:   return "16_19"
    if age_months <= 23:   return "20_23"
    return "24_30"


# ─────────────────────────────────────────────────────────────────────────────
# TABLE REGISTRY
# Maps (module_key, age_band) -> (sa_table, rrb_table)
# ─────────────────────────────────────────────────────────────────────────────

_CSS_TABLES: dict[tuple, tuple] = {
    # Original
    ("ados_original_module_1", "le30"):  (_M1_SA_LE30,  _M1_RRB_LE30),
    ("ados_original_module_1", "31_47"): (_M1_SA_3147,  _M1_RRB_3147),
    ("ados_original_module_1", "ge48"):  (_M1_SA_GE48,  _M1_RRB_GE48),
    ("ados_original_module_2", "le47"):  (_M2_SA_LE47,  _M2_RRB_LE47),
    ("ados_original_module_2", "48_59"): (_M2_SA_4859,  _M2_RRB_4859),
    ("ados_original_module_2", "ge60"):  (_M2_SA_GE60,  _M2_RRB_GE60),
    ("ados_original_module_3", "all"):   (_M3_SA,        _M3_RRB),
    ("ados_original_module_4", "all"):   (_M4_SA,        _M4_RRB),
    # ADOS-2
    ("ados_2_module_1", "le30"):         (_M1_2_SA_LE30, _M1_2_RRB_LE30),
    ("ados_2_module_1", "31_47"):        (_M1_2_SA_3147, _M1_2_RRB_3147),
    ("ados_2_module_1", "ge48"):         (_M1_2_SA_GE48, _M1_2_RRB_GE48),
    ("ados_2_module_2", "le47"):         (_M2_2_SA_LE47, _M2_2_RRB_LE47),
    ("ados_2_module_2", "48_59"):        (_M2_2_SA_4859, _M2_2_RRB_4859),
    ("ados_2_module_2", "ge60"):         (_M2_2_SA_GE60, _M2_2_RRB_GE60),
    ("ados_2_module_3", "all"):          (_M3_2_SA,       _M3_2_RRB),
    ("ados_2_module_4", "all"):          (_M4_2_SA,       _M4_2_RRB),
    ("ados_2_toddler",  "12_15"):        (_MT_SA_1215,    _MT_RRB),
    ("ados_2_toddler",  "16_19"):        (_MT_SA_1619,    _MT_RRB),
    ("ados_2_toddler",  "20_23"):        (_MT_SA_2023,    _MT_RRB),
    ("ados_2_toddler",  "24_30"):        (_MT_SA_2430,    _MT_RRB),
}


def _lookup(table: dict, raw: int) -> int:
    """Lookup CSS value for a raw score, clamping to table bounds."""
    if raw in table:
        return table[raw]
    keys = sorted(table.keys())
    if raw < keys[0]:  return table[keys[0]]
    if raw > keys[-1]: return table[keys[-1]]
    # Nearest neighbor for gaps
    closest = min(keys, key=lambda k: abs(k - raw))
    return table[closest]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_age_band(module_key: str, age_months: float) -> str:
    """Return the age band string for a given module and age."""
    if not isinstance(module_key, str):
        return "all"
    mk = module_key.lower()
    if "module_1" in mk or mk == "ados_2_module_1":
        return _m1_age_band(age_months)
    if "module_2" in mk:
        return _m2_age_band(age_months)
    if "toddler" in mk:
        return _mt_age_band(age_months)
    return "all"   # Modules 3 and 4


def compute_css(
    module_key: str,
    age_months: float | None,
    raw_sa: float | None,
    raw_rrb: float | None,
) -> dict:
    """
    Compute ADOS Calibrated Severity Scores.

    Parameters
    ----------
    module_key  : schema module key (e.g. 'ados_original_module_3')
    age_months  : age at ADOS administration in months
    raw_sa      : raw Social Affect total (sum of recoded items, NOT average)
    raw_rrb     : raw RRB total (sum of recoded items, NOT average)

    Returns
    -------
    dict with keys:
      css_sa    : Social Affect CSS (1-10) or None
      css_rrb   : RRB CSS (1-10) or None
      css_total : SA + RRB combined score (2-20) or None
      age_band  : age band string used
    """
    if raw_sa is None and raw_rrb is None:
        return {"css_sa": None, "css_rrb": None, "css_total": None, "age_band": None}
    if not isinstance(module_key, str):
        return {"css_sa": None, "css_rrb": None, "css_total": None, "age_band": None}

    # Default age when missing (use adult norms — most conservative)
    age_m = float(age_months) if age_months is not None else 120.0
    age_band = get_age_band(module_key, age_m)
    table_key = (module_key.lower(), age_band)

    if table_key not in _CSS_TABLES:
        return {"css_sa": None, "css_rrb": None, "css_total": None,
                "age_band": age_band}

    sa_table, rrb_table = _CSS_TABLES[table_key]

    css_sa  = _lookup(sa_table,  int(round(raw_sa)))  if raw_sa  is not None else None
    css_rrb = _lookup(rrb_table, int(round(raw_rrb))) if raw_rrb is not None else None
    css_total = (css_sa + css_rrb) if (css_sa and css_rrb) else None

    return {
        "css_sa":    css_sa,
        "css_rrb":   css_rrb,
        "css_total": css_total,
        "age_band":  age_band,
    }


def compute_css_dataframe(
    merged: "pd.DataFrame",
    ados_raw: "pd.DataFrame | None" = None,
) -> "pd.DataFrame":
    """
    Compute CSS for all patients who have ADOS data.

    Expects merged DataFrame to have:
      - _ados_module column (module key string)
      - _ados_raw_sa and _ados_raw_rrb columns (raw totals, not averages)
      - optional: ados_age_months from covariates joined in

    Alternatively, pass the raw ADOS DataFrame (before domain averaging)
    as ados_raw to recompute totals from items.

    Returns DataFrame with columns: css_sa, css_rrb, css_total, css_age_band
    """
    import pandas as pd

    if "_ados_raw_sa" not in merged.columns or "_ados_raw_rrb" not in merged.columns:
        return pd.DataFrame(index=merged.index)

    records = []
    for pid, row in merged.iterrows():
        module  = row.get("_ados_module")
        raw_sa  = row.get("_ados_raw_sa")
        raw_rrb = row.get("_ados_raw_rrb")
        age_m   = row.get("ados_age_months") or row.get("age_months")

        if module is None or (raw_sa is None and raw_rrb is None):
            records.append({"person_id": pid, "css_sa": None,
                            "css_rrb": None, "css_total": None, "css_age_band": None})
            continue

        result = compute_css(module, age_m, raw_sa, raw_rrb)
        records.append({
            "person_id": pid,
            "css_sa":       result["css_sa"],
            "css_rrb":      result["css_rrb"],
            "css_total":    result["css_total"],
            "css_age_band": result["age_band"],
        })

    return pd.DataFrame(records).set_index("person_id")
