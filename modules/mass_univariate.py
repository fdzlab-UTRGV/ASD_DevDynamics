"""
modules/mass_univariate.py
─────────────────────────────────────────────────────────────────────────────
Mass-univariate linear regressions with signed √ΔR² effect sizes.

Core function: run_mass_univariate()

For each predictor × outcome pair:
  1. Fit base model:  outcome ~ covariates
  2. Fit full model:  outcome ~ predictor + covariates
  3. ΔR²  = R²_full − R²_base   (unique variance explained by predictor)
  4. sign  = sign(β_predictor in full model)
  5. √ΔR² = sign × √(max(0, ΔR²))     ← signed semi-partial correlation

Effect size interpretation:
  |√ΔR²| ~ 0.10  small
  |√ΔR²| ~ 0.30  medium
  |√ΔR²| ~ 0.50  large

p-values are FDR-corrected (Benjamini-Hochberg) across all tests.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────────────────────────────────────
# OLS helper (minimal, no dependency on stats.py)
# ─────────────────────────────────────────────────────────────────────────────

def _ols(y: np.ndarray, *xs: np.ndarray) -> dict | None:
    """Fit OLS and return betas, R², p-value for last predictor, se."""
    n = len(y)
    X = np.column_stack([np.ones(n)] + list(xs))
    try:
        XtX    = X.T @ X
        XtXinv = np.linalg.inv(XtX)
        betas  = XtXinv @ (X.T @ y)
        resid  = y - X @ betas
        sse    = float(resid @ resid)
        sst    = float(np.sum((y - y.mean()) ** 2))
        r2     = 1.0 - sse / sst if sst > 0 else 0.0
        p      = len(xs) + 1              # number of params incl intercept
        mse    = sse / max(n - p, 1)
        se     = np.sqrt(np.maximum(0.0, np.diag(XtXinv) * mse))
        return {"betas": betas, "r2": r2, "se": se, "n": n}
    except np.linalg.LinAlgError:
        return None


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction over finite tests only.

    NaN/inf entries are not tests and therefore must not contribute to the
    BH family size or ranks. Their adjusted values remain NaN.
    """
    pvals = np.asarray(pvals, dtype=float)
    adj = np.full(pvals.shape, np.nan, dtype=float)

    valid = np.isfinite(pvals)
    valid_idx = np.flatnonzero(valid)
    m = len(valid_idx)
    if m == 0:
        return adj

    order = valid_idx[np.argsort(pvals[valid_idx])]
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        q = pvals[i] * m / (rank + 1)
        prev = min(q, prev)
        adj[i] = prev

    return np.clip(adj, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def run_mass_univariate(
    merged:     pd.DataFrame,
    predictors: list[str],
    outcomes:   list[str],
    cov_cols:   list[str] | None = None,
) -> dict:
    """
    Mass-univariate linear regressions with signed √ΔR² effect sizes.

    Parameters
    ----------
    merged     : merged SPARK DataFrame (person_id index)
    predictors : list of predictor column names
    outcomes   : list of outcome column names
    cov_cols   : covariate column names (partialled from both X and Y)

    Returns
    -------
    dict with keys:
      sqrt_dr2   : DataFrame (predictors × outcomes) of signed √ΔR²
      pval_raw   : DataFrame (predictors × outcomes) of raw p-values
      pval_fdr   : DataFrame (predictors × outcomes) of FDR-corrected p-values
      n_obs      : DataFrame (predictors × outcomes) of sample sizes
      beta       : DataFrame (predictors × outcomes) of raw β coefficients
      r2_base    : Series (outcomes) of base-model R² (covariates only)
      r2_full    : DataFrame (predictors × outcomes) of full-model R²
      cov_cols   : list of covariates actually used
      predictors : list of predictors actually run
      outcomes   : list of outcomes actually run
      n_tests    : total number of tests run
    """
    cov_cols   = [c for c in (cov_cols or []) if c in merged.columns]
    predictors = [p for p in predictors if p in merged.columns]
    outcomes   = [o for o in outcomes   if o in merged.columns]

    if not predictors or not outcomes:
        return {"error": "No valid predictors or outcomes in data."}

    # Pre-compute base-model R² per outcome (covariates only)
    r2_base: dict[str, float] = {}
    for out in outcomes:
        cols = [out] + cov_cols
        sub  = merged[cols].dropna()
        if len(sub) < len(cov_cols) + 3:
            r2_base[out] = np.nan
            continue
        y = sub[out].values
        if cov_cols:
            covs = [sub[c].values for c in cov_cols]
            res  = _ols(y, *covs)
            r2_base[out] = res["r2"] if res else np.nan
        else:
            r2_base[out] = 0.0     # no covariates → base R² = 0

    # Storage
    sqrt_dr2  = pd.DataFrame(index=predictors, columns=outcomes, dtype=float)
    pval_raw  = pd.DataFrame(index=predictors, columns=outcomes, dtype=float)
    n_obs     = pd.DataFrame(index=predictors, columns=outcomes, dtype=float)
    beta      = pd.DataFrame(index=predictors, columns=outcomes, dtype=float)
    r2_full_m = pd.DataFrame(index=predictors, columns=outcomes, dtype=float)

    for pred in predictors:
        for out in outcomes:
            cols = [pred, out] + cov_cols
            sub  = merged[cols].dropna()
            n    = len(sub)
            min_n = len(cov_cols) + 4      # predictor + intercept + covariates + 2
            if n < min_n:
                continue

            y    = sub[out].values
            x    = sub[pred].values
            covs = [sub[c].values for c in cov_cols]

            res = _ols(y, x, *covs)
            if res is None:
                continue

            b_pred   = float(res["betas"][1])
            se_pred  = float(res["se"][1])
            df_resid = n - len(cov_cols) - 2
            t_stat   = b_pred / se_pred if se_pred > 0 else 0.0
            p        = float(2 * scipy_stats.t.sf(abs(t_stat), df=max(df_resid, 1)))
            r2_f     = res["r2"]
            r2_b     = r2_base.get(out, 0.0)
            delta_r2 = max(0.0, r2_f - (r2_b or 0.0))
            sign     = np.sign(b_pred) if b_pred != 0 else 1.0
            sr2      = sign * np.sqrt(delta_r2)

            sqrt_dr2.loc[pred, out]  = sr2
            pval_raw.loc[pred, out]  = p
            n_obs.loc[pred, out]     = n
            beta.loc[pred, out]      = b_pred
            r2_full_m.loc[pred, out] = r2_f

    # FDR correction across all tests simultaneously
    flat_p   = pval_raw.values.flatten().astype(float)
    flat_fdr = _fdr_bh(flat_p)
    pval_fdr = pd.DataFrame(
        flat_fdr.reshape(pval_raw.shape),
        index=predictors, columns=outcomes,
    )

    return {
        "sqrt_dr2":   sqrt_dr2.astype(float),
        "pval_raw":   pval_raw.astype(float),
        "pval_fdr":   pval_fdr.astype(float),
        "n_obs":      n_obs.astype(float),
        "beta":       beta.astype(float),
        "r2_base":    pd.Series(r2_base),
        "r2_full":    r2_full_m.astype(float),
        "cov_cols":   cov_cols,
        "predictors": predictors,
        "outcomes":   outcomes,
        "n_tests":    int((~sqrt_dr2.isna()).values.sum()),
    }
