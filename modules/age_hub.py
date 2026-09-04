"""
modules/age_hub.py
──────────────────────────────────────────────────────────────────────────────
Age-stratified hubness index analysis.

Splits the sample into age bands, recomputes hubness rankings within each
band, and tests whether hub structure is stable across development.

Key design decisions
────────────────────
1. Age missingness is handled explicitly — participants without age data are
   excluded from stratified analyses but reported transparently.

2. Minimum N per band is enforced before running regressions. Bands with
   fewer than MIN_N participants are skipped and flagged in the output.

3. Hub ranking stability across bands is quantified as mean pairwise
   Spearman rank correlation — a single interpretable summary statistic
   for the paper's Results section.

4. Age is in months (age_months from cov-store). Bands are defined in
   years and converted internally.

Age bands (years → months)
──────────────────────────
    Early childhood : 4–8   (48–107 months)
    Middle childhood: 9–12  (108–155 months)
    Adolescence     : 13–17 (156–215 months)
    Young adult     : 18+   (216+ months)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from modules.mass_univariate import run_mass_univariate
from modules.hubness import compute_hubness


# ──────────────────────────────────────────────────────────────────────────────
# Band definitions
# ──────────────────────────────────────────────────────────────────────────────

AGE_BANDS = [
    {"label": "4–8 yrs",   "min_m":  48, "max_m": 107},
    {"label": "9–12 yrs",  "min_m": 108, "max_m": 155},
    {"label": "13–17 yrs", "min_m": 156, "max_m": 215},
    {"label": "18+ yrs",   "min_m": 216, "max_m": 9999},
]

MIN_N = 50   # minimum participants per band to run analysis


# ──────────────────────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────────────────────

def run_age_stratified_hubness(
    merged: pd.DataFrame,
    predictors: list[str],
    outcomes: list[str],
    cov_cols: list[str],
    age_col: str = "age_months",
    fdr_thresh: float = 0.05,
) -> dict:
    """
    Run hubness index analysis within each age band.

    Parameters
    ----------
    merged : pd.DataFrame
        Full merged SPARK DataFrame.
    predictors, outcomes, cov_cols : list[str]
        Column names as used by run_mass_univariate.
    age_col : str
        Column containing age in months (default: age_months).
    fdr_thresh : float
        FDR threshold for hubness computation.

    Returns
    -------
    dict with keys:
        band_results   — dict[band_label] → hub_df (ranked hubness per band)
        band_ns        — dict[band_label] → n participants in that band
        band_n_missing — int, participants excluded due to missing age
        stability      — dict with pairwise Spearman rs and mean_r
        rank_matrix    — DataFrame (predictors × bands) of hub ranks
        hub_matrix     — DataFrame (predictors × bands) of hubness values
        skipped_bands  — list of band labels skipped due to low N
        age_col_found  — bool, whether age_col was present in data
    """
    # ── Age column check ──────────────────────────────────────────────────
    if age_col not in merged.columns:
        # Try age_years as fallback, convert to months
        if "age_years" in merged.columns:
            merged = merged.copy()
            merged["age_months"] = merged["age_years"] * 12
            age_col = "age_months"
        else:
            return {"error": f"Age column '{age_col}' not found in data. "
                             "Load covariates file first."}

    age_series = merged[age_col]
    n_total = len(merged)
    n_missing_age = int(age_series.isna().sum())

    band_results: dict[str, pd.DataFrame] = {}
    band_ns: dict[str, int] = {}
    skipped: list[str] = []

    for band in AGE_BANDS:
        label = band["label"]
        mask  = (age_series >= band["min_m"]) & (age_series <= band["max_m"])
        sub   = merged[mask].copy()
        n     = len(sub)
        band_ns[label] = n

        if n < MIN_N:
            skipped.append(f"{label} (n={n}, below minimum {MIN_N})")
            continue

        # Remove age from covariates within-band — no variance to control
        # if everyone is the same age range. Keep sex and NVIQ.
        band_covs = [c for c in cov_cols if c != age_col and c in sub.columns]

        result = run_mass_univariate(sub, predictors, outcomes, band_covs)
        if "error" in result:
            skipped.append(f"{label} (error: {result['error']})")
            continue

        # Build a minimal payload compatible with compute_hubness
        payload = {
            "sqrt_dr2": result["sqrt_dr2"].to_dict(),
            "pval_fdr": result["pval_fdr"].to_dict(),
            "predictors": result["predictors"],
            "outcomes":   result["outcomes"],
        }

        hub_df = compute_hubness(payload, fdr_thresh=fdr_thresh, sig_only=True)
        band_results[label] = hub_df

    if len(band_results) < 2:
        return {
            "error": f"Fewer than 2 age bands had sufficient data (n≥{MIN_N}). "
                     f"Bands: {band_ns}. Skipped: {skipped}",
            "band_ns": band_ns,
            "n_missing_age": n_missing_age,
            "skipped_bands": skipped,
            "age_col_found": True,
        }

    # ── Build rank matrix and hubness matrix ─────────────────────────────
    all_preds = list(dict.fromkeys(
        p for hub_df in band_results.values()
        for p in hub_df["predictor"].tolist()
    ))

    rank_data = {}
    hub_data  = {}
    for label, hub_df in band_results.items():
        rank_map = dict(zip(hub_df["predictor"], hub_df["rank"]))
        hub_map  = dict(zip(hub_df["predictor"], hub_df["hubness_index"]))
        rank_data[label] = [rank_map.get(p, np.nan) for p in all_preds]
        hub_data[label]  = [hub_map.get(p, np.nan)  for p in all_preds]

    rank_matrix = pd.DataFrame(rank_data, index=all_preds)
    hub_matrix  = pd.DataFrame(hub_data,  index=all_preds)

    # ── Pairwise Spearman rank correlations across bands ─────────────────
    band_labels = list(band_results.keys())
    n_bands     = len(band_labels)
    pairs       = []
    rs_vals     = []

    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            la, lb = band_labels[i], band_labels[j]
            ra = rank_matrix[la].dropna()
            rb = rank_matrix[lb].dropna()
            common = ra.index.intersection(rb.index)
            if len(common) < 3:
                continue
            r, p = scipy_stats.spearmanr(ra[common], rb[common])
            pairs.append({"band_a": la, "band_b": lb,
                          "spearman_r": round(float(r), 3),
                          "p_value": round(float(p), 4),
                          "n_predictors": len(common)})
            rs_vals.append(float(r))

    mean_r = float(np.mean(rs_vals)) if rs_vals else np.nan

    return {
        "band_results":   band_results,
        "band_ns":        band_ns,
        "n_missing_age":  n_missing_age,
        "n_total":        n_total,
        "stability":      {"pairs": pairs, "mean_r": mean_r},
        "rank_matrix":    rank_matrix,
        "hub_matrix":     hub_matrix,
        "skipped_bands":  skipped,
        "age_col_found":  True,
        "fdr_thresh":     fdr_thresh,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Equal-N sensitivity analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_equaln_sensitivity(
    merged: pd.DataFrame,
    predictors: list[str],
    outcomes: list[str],
    cov_cols: list[str],
    age_col: str = "age_months",
    fdr_thresh: float = 0.05,
    seed: int = 42,
    n_iterations: int = 5,
) -> dict:
    """
    Equal-N sensitivity check for age-stratified hub stability.

    Addresses the concern that weaker concordance in later age bands may
    reflect smaller samples rather than genuine developmental shifts.

    Approach:
      1. Find the minimum N across all valid age bands.
      2. Subsample each band to that minimum N (repeated n_iterations times).
      3. Recompute hubness rankings and pairwise Spearman r in each iteration.
      4. Report mean ± SD Spearman r across iterations.

    If the concordance pattern from the full analysis holds in the
    equal-N subsamples, it reflects genuine developmental structure
    rather than sampling variability.

    Parameters
    ----------
    merged, predictors, outcomes, cov_cols, age_col, fdr_thresh :
        Same as run_age_stratified_hubness.
    seed : int
        Random seed for reproducibility.
    n_iterations : int
        Number of subsampling iterations (default 5; increase for publication).

    Returns
    -------
    dict with keys:
        equaln_n        — N used per band in each iteration
        iteration_rs    — list of dicts (one per iteration) with pairwise rs
        summary         — DataFrame with mean/SD Spearman r per band pair
        band_ns_full    — original unequal Ns for reference
    """
    rng = np.random.default_rng(seed)

    if age_col not in merged.columns:
        if "age_years" in merged.columns:
            merged = merged.copy()
            merged["age_months"] = merged["age_years"] * 12
            age_col = "age_months"
        else:
            return {"error": f"Age column '{age_col}' not found."}

    age_series = merged[age_col]

    # Build per-band index lists
    band_indices = {}
    for band in AGE_BANDS:
        mask = (age_series >= band["min_m"]) & (age_series <= band["max_m"])
        idx = merged.index[mask].tolist()
        if len(idx) >= MIN_N:
            band_indices[band["label"]] = idx

    if len(band_indices) < 2:
        return {"error": "Fewer than 2 bands with sufficient data for equal-N analysis."}

    # Use 80% of the minimum band size so even the smallest band
    # is genuinely subsampled in every iteration, giving meaningful SD.
    min_n_full   = min(len(v) for v in band_indices.values())
    equaln_n     = max(MIN_N, int(min_n_full * 0.80))
    band_ns_full = {k: len(v) for k, v in band_indices.items()}
    band_labels  = list(band_indices.keys())

    iteration_rs = []

    for it in range(n_iterations):
        iter_hub = {}
        for label, idx in band_indices.items():
            sampled = rng.choice(idx, size=equaln_n, replace=False)
            sub = merged.loc[sampled].copy()
            band_covs = [c for c in cov_cols
                         if c != age_col and c in sub.columns]
            result = run_mass_univariate(sub, predictors, outcomes, band_covs)
            if "error" in result:
                continue
            payload = {
                "sqrt_dr2":   result["sqrt_dr2"].to_dict(),
                "pval_fdr":   result["pval_fdr"].to_dict(),
                "predictors": result["predictors"],
                "outcomes":   result["outcomes"],
            }
            hub_df = compute_hubness(payload, fdr_thresh=fdr_thresh,
                                     sig_only=True)
            iter_hub[label] = dict(zip(hub_df["predictor"], hub_df["rank"]))

        # Pairwise Spearman rs for this iteration
        iter_pairs = {}
        for i in range(len(band_labels)):
            for j in range(i + 1, len(band_labels)):
                la, lb = band_labels[i], band_labels[j]
                if la not in iter_hub or lb not in iter_hub:
                    continue
                preds_common = list(
                    set(iter_hub[la]) & set(iter_hub[lb])
                )
                if len(preds_common) < 3:
                    continue
                ra = [iter_hub[la][p] for p in preds_common]
                rb = [iter_hub[lb][p] for p in preds_common]
                r, _ = scipy_stats.spearmanr(ra, rb)
                key = f"{la} vs {lb}"
                iter_pairs[key] = float(r)

        iteration_rs.append(iter_pairs)

    # Summarize across iterations
    all_keys = set(k for d in iteration_rs for k in d)
    summary_rows = []
    for key in sorted(all_keys):
        vals = [d[key] for d in iteration_rs if key in d]
        summary_rows.append({
            "comparison":  key,
            "mean_r":      round(float(np.mean(vals)), 3),
            "sd_r":        round(float(np.std(vals)), 3),
            "n_iter":      len(vals),
            "equaln_n":    equaln_n,
        })

    summary_df = pd.DataFrame(summary_rows)

    return {
        "equaln_n":     equaln_n,
        "n_iterations": n_iterations,
        "iteration_rs": iteration_rs,
        "summary":      summary_df,
        "band_ns_full": band_ns_full,
    }
