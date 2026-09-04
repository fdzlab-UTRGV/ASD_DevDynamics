"""
modules/hubness.py
──────────────────────────────────────────────────────────────────────────────
Hubness Index module for the SPARK Behavioral Fingerprint app.

Computes the hubness index from a p2-results-store payload (output of
run_mass_univariate). Designed to slot directly into the Part 2 tab as
Step 4 without any recomputation of √ΔR².

Hubness Index Definition
────────────────────────
    Hubness(predictor) = Σ |√ΔR²|  across FDR-significant outcomes

This integrates both:
  - Magnitude  : larger effects contribute more to the index
  - Breadth    : predictors with more significant outcomes accumulate more

Theoretical justification (for paper Methods section)
──────────────────────────────────────────────────────
The hubness index operationalizes the hub claim by treating each
predictor's unique association profile as a weighted degree in a
bipartite predictor-outcome association graph. Unlike raw degree
(which counts significant connections equally), the weighted formulation
rewards both having many connections AND having strong ones — properties
that together define a hub dimension in the network psychopathology sense.

The signed square-root transformation preserves comparability across
instruments (all values are semi-partial correlations, −1 to +1), so
summing across outcomes from different instruments is mathematically valid.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from modules.domains import get_domain, DOMAIN_COLORS


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_hubness(
    payload: dict,
    fdr_thresh: float = 0.05,
    sig_only: bool = True,
) -> pd.DataFrame:
    """
    Compute the hubness index from a p2-results-store payload.

    Parameters
    ----------
    payload : dict
        Contents of p2-results-store. Must have keys:
        sqrt_dr2, pval_fdr, n_obs, predictors, outcomes.
    fdr_thresh : float
        FDR significance threshold (default 0.05).
    sig_only : bool
        If True, sum only over FDR-significant outcomes.
        If False, sum over all non-NaN pairs (sensitivity check).

    Returns
    -------
    pd.DataFrame with columns:
        predictor       — predictor name
        domain          — domain label (from modules.domains.get_domain)
        domain_color    — hex color for the domain
        hubness_index   — Σ|√ΔR²| across significant outcomes
        n_significant   — number of FDR-significant outcomes
        n_total         — total outcomes with valid data
        mean_abs_effect — mean |√ΔR²| across significant outcomes
        max_abs_effect  — max |√ΔR²| across significant outcomes
        rank            — rank (1 = highest hubness)
    Sorted by hubness_index descending.
    """
    sqrt_dr2 = pd.DataFrame(payload["sqrt_dr2"])
    pval_fdr = pd.DataFrame(payload["pval_fdr"])

    records = []
    for pred in sqrt_dr2.index:
        vals  = sqrt_dr2.loc[pred].astype(float)
        qs    = pval_fdr.loc[pred].astype(float)
        sig   = qs < fdr_thresh

        if sig_only:
            mask = sig & vals.notna()
        else:
            mask = vals.notna()

        abs_effects = vals[mask].abs()

        hubness     = float(abs_effects.sum())
        n_sig       = int(sig.sum())
        n_total     = int(vals.notna().sum())
        mean_abs    = float(abs_effects.mean()) if len(abs_effects) > 0 else np.nan
        max_abs     = float(abs_effects.max())  if len(abs_effects) > 0 else np.nan
        domain      = get_domain(pred)

        records.append({
            "predictor":       pred,
            "domain":          domain,
            "domain_color":    DOMAIN_COLORS.get(domain, "#94a3b8"),
            "hubness_index":   hubness,
            "n_significant":   n_sig,
            "n_total":         n_total,
            "mean_abs_effect": mean_abs,
            "max_abs_effect":  max_abs,
        })

    hub_df = (pd.DataFrame(records)
                .sort_values("hubness_index", ascending=False)
                .reset_index(drop=True))
    hub_df["rank"] = hub_df.index + 1

    return hub_df


# ──────────────────────────────────────────────────────────────────────────────
# Serialization helpers (for hub-results-store)
# ──────────────────────────────────────────────────────────────────────────────

def hubness_to_store(hub_df: pd.DataFrame) -> dict:
    """Serialize hub_df to a JSON-safe dict for dcc.Store."""
    return hub_df.to_dict(orient="records")


def hubness_from_store(data: list[dict]) -> pd.DataFrame:
    """Deserialize hub_df from a dcc.Store payload."""
    return pd.DataFrame(data)
