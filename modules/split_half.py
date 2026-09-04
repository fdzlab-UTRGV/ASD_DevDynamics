"""
modules/split_half.py
─────────────────────────────────────────────────────────────────────────────
Split-half replication analysis for Part 2.

Pipeline
--------
1. Split the sample into demographically matched discovery / replication halves
   (stratified by age quartile × sex so both halves have balanced demographics).

2. Run mass-univariate √ΔR² independently on each half.

3. Compute concordance r — Pearson correlation between discovery and replication
   √ΔR² vectors (all predictor × outcome cells stacked). This is the headline
   replication statistic (analogous to Macedo et al.'s r = 0.93).

4. Ridge regression — trained on discovery, tested on replication.
   α is cross-validated (k=5) within the discovery half.
   Reports out-of-sample Pearson r per outcome.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from modules.mass_univariate import run_mass_univariate


# ─────────────────────────────────────────────────────────────────────────────
# Stratified split
# ─────────────────────────────────────────────────────────────────────────────

def split_sample_matched(
    df: pd.DataFrame,
    age_col:  str = "age_months",
    sex_col:  str = "sex",
    seed:     int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into two demographically matched halves.

    Stratifies by age quartile × sex so both halves have similar
    age and sex distributions. Returns (discovery, replication).
    """
    rng = np.random.default_rng(seed)

    # Build stratum labels
    if age_col in df.columns and df[age_col].notna().any():
        age_q = pd.qcut(df[age_col].fillna(df[age_col].median()),
                        q=4, labels=False, duplicates="drop")
    else:
        age_q = pd.Series(0, index=df.index)

    if sex_col in df.columns:
        sex_b = df[sex_col].fillna(0).astype(int)
    else:
        sex_b = pd.Series(0, index=df.index)

    strata = age_q.astype(str) + "_" + sex_b.astype(str)

    disc_idx, rep_idx = [], []

    for stratum in strata.unique():
        s_idx = df.index[strata == stratum].tolist()
        if len(s_idx) < 2:
            # Too few in stratum — put all in discovery
            disc_idx.extend(s_idx)
            continue
        arr = np.array(s_idx, dtype=object)
        rng.shuffle(arr)
        mid = len(arr) // 2
        disc_idx.extend(arr[:mid].tolist())
        rep_idx.extend(arr[mid:].tolist())

    disc = df.loc[df.index.isin(disc_idx)]
    rep  = df.loc[df.index.isin(rep_idx)]
    return disc, rep


# ─────────────────────────────────────────────────────────────────────────────
# Ridge regression helpers
# ─────────────────────────────────────────────────────────────────────────────

def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise columns (z-score). Returns (Z, mean, std)."""
    mu  = X.mean(axis=0)
    sig = X.std(axis=0, ddof=1)
    sig[sig == 0] = 1.0
    return (X - mu) / sig, mu, sig


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge weights: w = (X'X + αI)⁻¹ X'y."""
    p   = X.shape[1]
    XtX = X.T @ X
    w   = np.linalg.solve(XtX + alpha * np.eye(p), X.T @ y)
    return w


def _cv_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list[float],
    k: int = 5,
    seed: int = 42,
) -> float:
    """K-fold cross-validation to select the best ridge α."""
    n = len(y)
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(n)
    folds = np.array_split(idx, k)

    best_alpha = alphas[0]
    best_mse   = np.inf

    for alpha in alphas:
        fold_mse = []
        for fi in range(k):
            val   = folds[fi]
            train = np.concatenate([folds[j] for j in range(k) if j != fi])
            if len(train) < X.shape[1] + 2:
                continue
            try:
                w    = _ridge_fit(X[train], y[train], alpha)
                pred = X[val] @ w
                fold_mse.append(float(np.mean((y[val] - pred) ** 2)))
            except np.linalg.LinAlgError:
                continue
        if fold_mse:
            mse = float(np.mean(fold_mse))
            if mse < best_mse:
                best_mse   = mse
                best_alpha = alpha

    return best_alpha


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def run_split_half(
    merged:     pd.DataFrame,
    predictors: list[str],
    outcomes:   list[str],
    cov_cols:   list[str] | None = None,
    seed:       int = 42,
) -> dict:
    """
    Full split-half replication analysis.

    Parameters
    ----------
    merged     : merged SPARK DataFrame
    predictors : list of predictor column names
    outcomes   : list of outcome column names
    cov_cols   : covariate column names
    seed       : random seed for the split

    Returns
    -------
    dict with:
      concordance_r      float     Pearson r(disc √ΔR², rep √ΔR²)
      concordance_p      float     two-sided p-value
      n_pairs            int       number of non-NaN cell pairs in concordance
      disc_dr2           DataFrame discovery √ΔR² matrix
      rep_dr2            DataFrame replication √ΔR² matrix
      n_disc             int
      n_rep              int
      ridge              list[dict] per-outcome ridge results
      best_alphas        dict      predictor→chosen α
      cov_cols           list
    """
    cov_cols   = [c for c in (cov_cols or []) if c in merged.columns]
    predictors = [p for p in predictors if p in merged.columns]
    outcomes   = [o for o in outcomes   if o in merged.columns]

    if not predictors or not outcomes:
        return {"error": "No valid predictors or outcomes."}

    # ── Split ────────────────────────────────────────────────────────────────
    disc, rep = split_sample_matched(merged, seed=seed)
    n_disc, n_rep = len(disc), len(rep)

    if n_disc < len(predictors) + 5 or n_rep < len(predictors) + 5:
        return {"error": f"Halves too small: discovery={n_disc}, replication={n_rep}."}

    # ── √ΔR² on each half ────────────────────────────────────────────────────
    disc_result = run_mass_univariate(disc, predictors, outcomes, cov_cols)
    rep_result  = run_mass_univariate(rep,  predictors, outcomes, cov_cols)

    if "error" in disc_result or "error" in rep_result:
        return {"error": "Mass-univariate failed on one or both halves."}

    disc_dr2     = disc_result["sqrt_dr2"]
    rep_dr2      = rep_result["sqrt_dr2"]
    disc_pval_fdr = disc_result["pval_fdr"]
    rep_pval_fdr  = rep_result["pval_fdr"]
    disc_n_obs    = disc_result["n_obs"]
    rep_n_obs     = rep_result["n_obs"]

    # ── Concordance r ─────────────────────────────────────────────────────────
    d_vec = disc_dr2.values.flatten().astype(float)
    r_vec = rep_dr2.values.flatten().astype(float)
    mask  = ~(np.isnan(d_vec) | np.isnan(r_vec))
    n_pairs = int(mask.sum())

    if n_pairs >= 3:
        conc_r, conc_p = scipy_stats.pearsonr(d_vec[mask], r_vec[mask])
    else:
        conc_r, conc_p = np.nan, np.nan

    # ── Ridge regression: train on discovery, test on replication ────────────
    # Build complete-case matrices for discovery and replication
    all_cols = predictors + outcomes + cov_cols
    disc_cc  = disc[all_cols].dropna()
    rep_cc   = rep[all_cols].dropna()

    ridge_results = []
    best_alphas   = {}

    if len(disc_cc) >= len(predictors) + 3 and len(rep_cc) >= 3:
        X_disc_raw = disc_cc[predictors].values
        X_rep_raw  = rep_cc[predictors].values

        # Standardise using discovery statistics
        X_disc, mu, sig = _standardise(X_disc_raw)
        X_rep  = (X_rep_raw - mu) / sig

        # Remove constant columns
        active = np.std(X_disc, axis=0) > 0
        X_disc = X_disc[:, active]
        X_rep  = X_rep[:, active]
        active_preds = [p for p, a in zip(predictors, active) if a]

        for out in outcomes:
            y_disc = disc_cc[out].values.astype(float)
            y_rep  = rep_cc[out].values.astype(float)

            if np.std(y_disc) < 1e-9 or np.std(y_rep) < 1e-9:
                continue

            # Choose α via CV on discovery
            alpha = _cv_alpha(X_disc, y_disc, ALPHAS, k=5, seed=seed)
            best_alphas[out] = alpha

            # Fit on full discovery, predict on replication
            try:
                w    = _ridge_fit(X_disc, y_disc, alpha)
                pred = X_rep @ w

                r_test, p_test = scipy_stats.pearsonr(y_rep, pred)
                r2_test = float(r_test ** 2)

                # Null baseline: predict discovery mean
                null_pred = np.full_like(pred, y_disc.mean())
                r_null = float(np.corrcoef(y_rep, null_pred)[0, 1])

                ridge_results.append({
                    "outcome":   out,
                    "r_test":    float(r_test),
                    "r2_test":   float(r2_test),
                    "p_test":    float(p_test),
                    "alpha":     alpha,
                    "n_disc":    len(y_disc),
                    "n_rep":     len(y_rep),
                    "n_preds":   int(active.sum()),
                })
            except (np.linalg.LinAlgError, ValueError):
                continue

    return {
        "concordance_r":   float(conc_r),
        "concordance_p":   float(conc_p),
        "n_pairs":         n_pairs,
        "disc_dr2":        disc_dr2,
        "rep_dr2":         rep_dr2,
        "disc_pval_fdr":   disc_pval_fdr,
        "rep_pval_fdr":    rep_pval_fdr,
        "disc_n_obs":      disc_n_obs,
        "rep_n_obs":       rep_n_obs,
        "n_disc":          n_disc,
        "n_rep":           n_rep,
        "ridge":           ridge_results,
        "best_alphas":     best_alphas,
        "cov_cols":        cov_cols,
        "predictors":      predictors,
        "outcomes":        outcomes,
    }
