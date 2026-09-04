"""
modules/ridge.py
─────────────────────────────────────────────────────────────────────────────
Out-of-sample ridge regression: Figure 3C (Fernandez et al.).

Population-level design (matches the main √ΔR² analysis)
-------------------------------------------------------
The analysis works at the population level: every individual contributes with
whatever data they have available, rather than being dropped for missing any
single predictor. Concretely, for each outcome we fit one ridge model on all
discovery participants who have that outcome, standardizing each predictor by
its discovery mean/SD (computed over that predictor's available cases) and
mean-imputing any missing predictor to the population mean. The trained model
is then applied — without retraining — to all holdout participants who have
that outcome. This is the multivariate analog of the per-pair complete-case
logic used in run_mass_univariate, and it keeps the sample size at the
population level instead of collapsing to listwise-complete cases.

Pipeline (per outcome)
----------------------
1. Discovery cases with the outcome → standardize + impute predictors.
2. Select ridge α by 5-fold cross-validation within discovery.
3. Fit ridge once on all those discovery cases.
4. Apply to all holdout cases with the outcome (no retraining).
5. Report out-of-sample Pearson r and a permutation p-value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────────────────────────────────────
# Ridge helpers (no sklearn dependency)

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """
    Closed-form ridge with an (unpenalized) intercept.

    Predictors are assumed already standardized (mean ~0). We fit
      y = X w + b
    by centering y and solving the penalized normal equations for w, then
    recovering the intercept b = mean(y) - mean(X) @ w. The intercept is NOT
    penalized, which is standard practice — penalizing it would bias predictions
    toward zero and can distort out-of-sample fit.

    Returns (w, b).
    """
    p      = X.shape[1]
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc     = X - x_mean
    yc     = y - y_mean
    XtX    = Xc.T @ Xc
    w      = np.linalg.solve(XtX + alpha * np.eye(p), Xc.T @ yc)
    b      = y_mean - x_mean @ w
    return w, b


def _cv_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list[float] = ALPHAS,
    k: int = 5,
    seed: int = 42,
) -> float:
    """Select regularization α by k-fold cross-validation (minimize MSE)."""
    n   = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
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
                w, b = _ridge_fit(X[train], y[train], alpha)
                pred = X[val] @ w + b
                fold_mse.append(float(np.mean((y[val] - pred) ** 2)))
            except np.linalg.LinAlgError:
                continue
        if fold_mse and np.mean(fold_mse) < best_mse:
            best_mse   = float(np.mean(fold_mse))
            best_alpha = alpha

    return best_alpha


# ─────────────────────────────────────────────────────────────────────────────
# Main function

def run_oos_ridge(
    discovery:  pd.DataFrame,
    holdout:    pd.DataFrame,
    predictors: list[str],
    outcomes:   list[str],
    n_perm:     int = 1000,
    seed:       int = 42,
) -> dict:
    """
    Train ridge on discovery, evaluate on the holdout cohort.

    Parameters
    ----------
    discovery  : SPARK discovery sample (large)
    holdout    : holdout cohort (e.g. SSC), scored to match discovery
    predictors : behavioral predictor column names (same in both DataFrames)
    outcomes   : CBCL / ADOS CSS outcome column names
    n_perm     : permutations for p-value estimation
    seed       : random seed

    Returns
    -------
    dict with keys:
      results   list[dict]  per-outcome ridge performance
      error     str         set only if a fatal error occurred
    """
    rng = np.random.default_rng(seed)

    # Validate columns
    pred_disc = [c for c in predictors if c in discovery.columns]
    pred_hold = [c for c in predictors if c in holdout.columns]
    shared_pred = [c for c in pred_disc if c in pred_hold]

    if len(shared_pred) < 2:
        return {"error": (
            f"Too few shared predictor columns between discovery and holdout. "
            f"Discovery has: {pred_disc}; Holdout has: {pred_hold}."
        )}

    outcomes_disc = [c for c in outcomes if c in discovery.columns]
    outcomes_hold = [c for c in outcomes if c in holdout.columns]
    shared_out = [c for c in outcomes_disc if c in outcomes_hold]

    if not shared_out:
        return {"error": (
            "No shared outcome columns between discovery and holdout. "
            "Check that the holdout file contains CBCL columns."
        )}

    # ── Population-level standardization ─────────────────────────────────────
    # Following the original analysis: we work at the population level and each
    # individual contributes with whatever data they have available. Rather than
    # listwise-deleting anyone missing any single predictor (which discards most
    # of the sample), we standardize each predictor using the discovery column
    # mean/SD computed over its own available cases, and mean-impute missing
    # predictor values to the (standardized) population mean of 0. Every person
    # who has an outcome value therefore contributes to that outcome's model,
    # using their available predictors — the multivariate analog of the
    # per-pair complete-case logic in run_mass_univariate.

    # Per-column discovery mean/SD over available (non-missing) cases
    disc_mu, disc_sig = {}, {}
    for c in shared_pred:
        col = discovery[c].astype(float)
        vals = col.dropna().values
        if len(vals) < 3 or np.std(vals, ddof=1) == 0:
            continue
        disc_mu[c]  = float(np.mean(vals))
        disc_sig[c] = float(np.std(vals, ddof=1))

    active_pred = [c for c in shared_pred if c in disc_mu]
    if len(active_pred) < 2:
        return {"error": (
            "Fewer than 2 predictors have usable variance in the discovery "
            "sample. Check predictor scaling."
        )}

    def _standardize_impute(df: pd.DataFrame) -> np.ndarray:
        """Z-score each predictor by discovery stats; impute missing → 0 (mean)."""
        cols = []
        for c in active_pred:
            z = (df[c].astype(float) - disc_mu[c]) / disc_sig[c]
            cols.append(z.values)
        Z = np.column_stack(cols)
        Z = np.where(np.isnan(Z), 0.0, Z)   # mean-impute missing predictors
        return Z

    results = []
    n_holdout_overall = 0

    for out in shared_out:
        # ── Discovery: everyone with this outcome contributes ────────────────
        disc_sub = discovery[discovery[out].notna()]
        if len(disc_sub) < len(active_pred) + 3:
            continue

        X_d = _standardize_impute(disc_sub)
        y_d = disc_sub[out].values.astype(float)
        if np.std(y_d) < 1e-9:
            continue

        # Select α by 5-fold CV, fit ridge on all discovery cases for this outcome
        alpha = _cv_alpha(X_d, y_d, k=5, seed=seed)
        try:
            w, b = _ridge_fit(X_d, y_d, alpha)
        except np.linalg.LinAlgError:
            continue

        # ── Holdout: everyone with this outcome contributes ──────────────────
        hold_sub = holdout[holdout[out].notna()]
        if len(hold_sub) < 3:
            continue

        X_h     = _standardize_impute(hold_sub)
        y_hold  = hold_sub[out].values.astype(float)
        y_pred  = X_h @ w + b

        if np.std(y_hold) < 1e-9 or np.std(y_pred) < 1e-9:
            continue

        r_obs, _ = scipy_stats.pearsonr(y_hold, y_pred)
        r2_obs   = float(r_obs ** 2)

        # Permutation p-value
        perm_r = np.empty(n_perm)
        for i in range(n_perm):
            perm_r[i] = scipy_stats.pearsonr(rng.permutation(y_hold), y_pred)[0]
        p_perm = float((np.abs(perm_r) >= np.abs(r_obs)).mean())
        if p_perm == 0.0:
            p_perm = 1.0 / (n_perm + 1)

        n_holdout_overall = max(n_holdout_overall, len(hold_sub))

        results.append({
            "outcome":      out,
            "r":            float(r_obs),
            "r2":           r2_obs,
            "p_perm":       p_perm,
            "alpha":        float(alpha),
            "n_discovery":  int(len(disc_sub)),
            "n_holdout":    int(len(hold_sub)),
            "n_predictors": int(len(active_pred)),
        })

    if not results:
        return {"error": (
            "No outcomes could be evaluated. Check that the holdout file "
            "contains outcome (CBCL) columns with non-missing values."
        )}

    return {"results": results}
