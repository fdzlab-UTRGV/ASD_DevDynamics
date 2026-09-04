"""
modules/stats.py
─────────────────────────────────────────────────────────────────────────────
Shared statistical utilities.

Only the functions used by the paper's analyses are retained:
  - fdr_bh                  Benjamini-Hochberg FDR correction
  - pearson_pairwise_matrix Figure 1 unadjusted Pearson r matrix

The covariate-adjusted √ΔR² framework lives in modules/mass_univariate.py.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values (q-values)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order  = np.argsort(pvals)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    adj   = pvals * n / ranked
    adj_s = adj[order]
    for i in range(len(adj_s) - 2, -1, -1):
        adj_s[i] = min(adj_s[i], adj_s[i + 1])
    adj[order] = adj_s
    return np.minimum(adj, 1.0)


def pearson_pairwise_matrix(merged, predictors, outcomes):
    """
    Unadjusted pairwise Pearson correlations for all predictor × outcome pairs
    (Figure 1). Uses all complete cases per pair. FDR-corrects across pairs.

    Returns a long DataFrame with columns:
      predictor, outcome, r, p_raw, n, p_fdr
    """
    rows = []
    for pred in predictors:
        if pred not in merged.columns:
            continue
        for out in outcomes:
            if out not in merged.columns or pred == out:
                continue
            pair = merged[[pred, out]].dropna()
            n = len(pair)
            if n < 5:
                rows.append({"predictor": pred, "outcome": out,
                             "r": np.nan, "p_raw": np.nan, "n": n})
                continue
            r, p = scipy_stats.pearsonr(pair[pred], pair[out])
            rows.append({"predictor": pred, "outcome": out,
                         "r": float(r), "p_raw": float(p), "n": n})

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    valid = ~result["p_raw"].isna()
    if valid.any():
        result.loc[valid, "p_fdr"] = fdr_bh(result.loc[valid, "p_raw"].values)
    return result


# ── ADOS Calibrated Severity Scores ──────────────────────────────────────────

def compute_css_for_merged(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ADOS Calibrated Severity Scores (CSS) for every patient in `merged`.

    Requires the raw-total columns produced by loader.load_ados:
      _ados_raw_sa, _ados_raw_rrb, _ados_module
    Uses ados_age_months (or age_months) for age-band selection.

    Returns a DataFrame indexed identically to `merged` with columns:
      css_sa, css_rrb, css_total, css_age_band

    These are the ADOS CSS outcomes used in Figures 1, 2, and 3C. If the raw
    ADOS columns are absent, returns an all-NaN frame so downstream merges and
    analyses degrade gracefully.
    """
    from modules.ados_css import compute_css

    needed = {"_ados_raw_sa", "_ados_raw_rrb", "_ados_module"}
    if not needed.issubset(merged.columns):
        return pd.DataFrame(
            columns=["css_sa", "css_rrb", "css_total", "css_age_band"],
            index=merged.index,
        )

    rows = []
    for _pid, row in merged.iterrows():
        module  = row.get("_ados_module")
        raw_sa  = row.get("_ados_raw_sa")
        raw_rrb = row.get("_ados_raw_rrb")
        age_m   = row.get("ados_age_months") or row.get("age_months")

        if module is None or (isinstance(module, float) and pd.isna(module)):
            rows.append({"css_sa": None, "css_rrb": None,
                         "css_total": None, "css_age_band": None})
            continue

        res = compute_css(module, age_m, raw_sa, raw_rrb)
        rows.append({
            "css_sa":       res["css_sa"],
            "css_rrb":      res["css_rrb"],
            "css_total":    res["css_total"],
            "css_age_band": res["age_band"],
        })

    return pd.DataFrame(rows, index=merged.index)
