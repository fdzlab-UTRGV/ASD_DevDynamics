"""
modules/dev_age_analysis.py
────────────────────────────────────────────────────────────────────────────
Developmental age-band analysis.

Computes √ΔR² per domain × T1 age band cell using the same nested
regression framework as the Domain √ΔR² tab, with CBCL Psychopathology
composite as the outcome. T2 is fixed at 12–18y (144–216 months).

Three domains: Sensory-Repetitive, Motor, Social
Four T1 bands: 0–4y, 4–8y, 8–12y, 12–18y (concurrent)

Then runs five analyses on the 12 cells:
  1. Two-way ANOVA: domain × band effects on √ΔR²
  2. Tukey HSD post-hoc comparisons
  3. Segmented regression on rank-ordered √ΔR²
  4. Cell-level Bayesian hierarchical model (12 cell estimates)
  5. Individual-level Bayesian hierarchical model (raw observations)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from itertools import combinations

# ── Domain definitions (same as Domain √ΔR² tab) ─────────────────────────────

from modules.domains import DOMAIN_COLORS as _PART2_COLORS

# Use exact same colors as Part 2 √ΔR² tab
DOMAINS = {
    "Sensory-Repetitive": {
        "features": [
            "rbs_Sensory","rbs_Obsessive","rbs_Ritualistic",
            "rbs_Stereotyped","rbs_Compulsive","rbs_Self-Injurious",
            "sp_low_reg","sp_sensitivity","sp_avoiding","sp_seeking",
            "seq_hyper","seq_hypo","seq_enhanced",
            "isq_noticing","isq_interpreting","isq_acting",
            "scq_Sensory","ados_RRB",
        ],
        "age_cols": ["rbs_age_months","scq_age_months",
                     "sensory_age_months","ados_age_months"],
        "color": _PART2_COLORS.get("Sensory", "#fb923c"),
    },
    "Motor": {
        "features": [
            "dcdq_Fine Motor","dcdq_Gross Motor","dcdq_Coordination",
        ],
        "age_cols": ["dcdq_age_months"],
        "color": _PART2_COLORS.get("Motor", "#38bdf8"),
    },
    "Social": {
        "features": [
            "scq_Social","scq_Communication","ados_Social Affect",
        ],
        "age_cols": ["scq_age_months","ados_age_months"],
        "color": _PART2_COLORS.get("Social", "#34d399"),
    },
}

CBCL_FEATURES = [
    "cbcl_Internalizing","cbcl_Externalizing",
    "cbcl_Anxious/Dep.","cbcl_Social Prob.",
    "cbcl_Attention","cbcl_ADHD",
]
CBCL_AGE_COL = "cbcl_age_months"

T1_BANDS = {
    "0–4y":   (0,    48),
    "4–8y":   (48,   96),
    "8–12y":  (96,  144),
    "12–18y": (144, 216),
}
T2_WINDOW = (144, 216)   # 12–18y CBCL
DOMAIN_ORDER = ["Sensory-Repetitive", "Motor", "Social"]
BASE_COVS    = ["sex", "nviq"]

# Structurally missing cells — no data collected for these domain×band
# combinations. Never display or report slopes for these cells.
MISSING_CELLS = {
    ("Motor", "0\u20134y"),   # DCDQ not administered under 4y
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zscore_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index)
    z = pd.DataFrame(index=df.index)
    for col in present:
        vals  = pd.to_numeric(df[col], errors="coerce")
        mu, s = float(vals.mean()), float(vals.std(ddof=1))
        z[col] = (vals - mu) / s if s > 1e-9 else 0.0
    comp = z.mean(axis=1, skipna=True)
    comp.loc[df[present].isna().all(axis=1)] = np.nan
    return comp


def _domain_age(merged: pd.DataFrame, domain: str) -> pd.Series:
    cols  = [c for c in DOMAINS[domain]["age_cols"] if c in merged.columns]
    if not cols:
        return pd.Series(np.nan, index=merged.index)
    return merged[cols].apply(pd.to_numeric, errors="coerce").min(axis=1)


def _ols(y: np.ndarray, *xs: np.ndarray):
    n  = len(y)
    X  = np.column_stack([np.ones(n)] + list(xs))
    try:
        b  = np.linalg.lstsq(X, y, rcond=None)[0]
        r  = y - X @ b
        ss = float(r @ r)
        st = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss / st if st > 0 else 0.0
        p  = X.shape[1]
        mse = ss / max(n - p, 1)
        se  = np.sqrt(np.maximum(0, np.diag(np.linalg.inv(X.T @ X)) * mse))
        t   = b[-1] / se[-1] if se[-1] > 0 else 0.0
        pv  = float(2 * scipy_stats.t.sf(abs(t), max(n - p, 1)))
        return {"betas": b, "r2": r2, "se": se, "pval": pv, "n": n}
    except Exception:
        return None


def _sqrt_dr2(sub: pd.DataFrame, x_col: str,
              y_col: str, cov_cols: list[str]) -> dict | None:
    needed = [x_col, y_col] + cov_cols
    avail  = [c for c in needed if c in sub.columns]
    df     = sub[avail].dropna()
    n      = len(df)
    if n < len(avail) + 5:
        return None
    y      = df[y_col].values.astype(float)
    covs   = [df[c].values.astype(float) for c in cov_cols if c in df.columns]
    x      = df[x_col].values.astype(float)
    base   = _ols(y, *covs) if covs else {"r2": 0.0}
    full   = _ols(y, *covs, x)
    if full is None:
        return None
    dr2    = max(0.0, full["r2"] - (base["r2"] if base else 0.0))
    sign   = np.sign(full["betas"][-1])
    return {
        "sqrt_dr2": float(sign * np.sqrt(dr2)),
        "delta_r2": dr2,
        "beta":     float(full["betas"][-1]),
        "se_beta":  float(full["se"][-1]),
        "pval":     full["pval"],
        "n":        n,
        # SE of √ΔR² via delta method: se_sqrt_dr2 ≈ se_beta / (2*sqrt_dr2)
        # use a safe approximation
        "se_sqrt_dr2": float(full["se"][-1] / (2 * max(abs(sign * np.sqrt(dr2)), 1e-6))),
    }


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    n    = len(pvals)
    idx  = np.argsort(pvals)
    adj  = np.full(n, np.nan)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i    = idx[rank]
        q    = pvals[i] * n / (rank + 1)
        prev = min(q, prev)
        adj[i] = prev
    return np.clip(adj, 0.0, 1.0)


# ── Core: √ΔR² per cell ───────────────────────────────────────────────────────

def compute_cells(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute √ΔR² for each domain × T1 age band cell.

    Population-level design: for each cell, take everyone whose domain
    was measured in that age band. Of those, whoever also has CBCL at
    any age contributes to the regression. Age (domain eval age) and sex
    are covariates. No T2 window restriction — CBCL from any age.

    This treats each age band as a population-level cross-section.
    The signal is the domain-psychopathology association in that
    developmental slice, not a within-person prospective prediction.
    """
    # Build CBCL composite — everyone with any CBCL data
    psych    = _zscore_mean(merged, CBCL_FEATURES)
    cbcl_age = pd.to_numeric(
        merged.get(CBCL_AGE_COL, pd.Series(np.nan, index=merged.index)),
        errors="coerce")

    rows = []
    for dom in DOMAIN_ORDER:
        dom_comp = _zscore_mean(merged, DOMAINS[dom]["features"])
        dom_age  = _domain_age(merged, dom)
        color    = DOMAINS[dom]["color"]

        for band_name, (lo, hi) in T1_BANDS.items():
            # Skip structurally missing cells
            if (dom, band_name) in MISSING_CELLS:
                continue
            # Everyone with domain measured in this age band
            band_mask = (dom_age >= lo) & (dom_age < hi) & dom_comp.notna()

            sub = pd.DataFrame({
                "_domain": dom_comp[band_mask],
                "_psych":  psych[band_mask],
                "_age":    dom_age[band_mask],
                "sex":     pd.to_numeric(
                    merged.loc[band_mask, "sex"], errors="coerce")
                           if "sex" in merged.columns
                           else pd.Series(np.nan, index=merged.index[band_mask]),
            }).dropna(subset=["_domain", "_age"])
            # _psych NaN → those people just don't contribute to regression

            cov_cols = ["_age"] + (
                ["sex"] if "sex" in sub.columns and
                sub["sex"].notna().sum() > 10 else [])

            r = _sqrt_dr2(sub, "_domain", "_psych", cov_cols)

            if r:
                rows.append({
                    "domain":      dom,
                    "band":        band_name,
                    "n":           r["n"],
                    "sqrt_dr2":    round(r["sqrt_dr2"], 4),
                    "delta_r2":    round(r["delta_r2"], 4),
                    "beta":        round(r["beta"], 4),
                    "se_beta":     round(r["se_beta"], 4),
                    "se_sqrt_dr2": round(r["se_sqrt_dr2"], 4),
                    "pval":        r["pval"],
                    "color":       color,
                    # store individual data for individual-level Bayes
                    "_sub":        sub,
                })

    if not rows:
        return pd.DataFrame()

    # FDR correction
    pvals = np.array([r["pval"] for r in rows])
    pfdr  = _fdr_bh(pvals)
    for r, q in zip(rows, pfdr):
        r["pval_fdr"] = round(float(q), 5)
        r["sig"]      = ("***" if q < 0.001 else "**" if q < 0.01
                         else "*" if q < 0.05 else "ns")

    return pd.DataFrame(rows)


# ── 1. Two-way ANOVA ──────────────────────────────────────────────────────────

def run_anova(cells: pd.DataFrame) -> dict:
    """
    Three-part ANOVA on the 11 observed sqrt(DeltaR2) cell estimates.

    Part 1 — Domain main effect (one-way ANOVA):
        Do the three domain trajectories operate at different levels?
        df_between=2, df_within=8 (pooling all 11 cells within domains).
        This is the primary confirmatory test.

    Part 2 — Per-domain linear age trend (one per domain):
        Within each domain, does sqrt(DeltaR2) change linearly across bands?
        Sensory-Repetitive: 4 bands (slope across 0-4y, 4-8y, 8-12y, 12-18y)
        Motor:              3 bands (4-8y, 8-12y, 12-18y — 0-4y structurally absent)
        Social:             4 bands
        Estimated as weighted simple regression: sqrt(DeltaR2) ~ band_index,
        weighted by 1/SE^2. Tests whether the domain line has a significant slope.

    Part 3 — Interaction note:
        Whether the age slope DIFFERS between domains is tested by the
        GLM on individual data (score x Domain x Band), not here.
        Cell-level ANOVA has df_error=5 for the interaction — too low to interpret.
    """
    if cells.empty or len(cells) < 4:
        return {"error": "Not enough cells for ANOVA."}

    # Exclude structurally missing cells
    cells = cells[~cells.apply(
        lambda r: (r["domain"], r["band"]) in MISSING_CELLS, axis=1
    )].copy()

    y   = cells["sqrt_dr2"].values.astype(float)
    w   = (1.0 / (cells["se_sqrt_dr2"].values.astype(float)**2 + 1e-9))
    dom = cells["domain"].values
    bnd = cells["band"].values
    n   = len(y)

    # ── Part 1: Domain main effect (one-way ANOVA) ────────────────────────
    dom_labels = DOMAIN_ORDER
    k_dom      = len(dom_labels)
    grand_mean = float(np.average(y, weights=w))

    ss_bet = sum(
        float(np.sum(w[dom==d])) *
        (float(np.average(y[dom==d], weights=w[dom==d])) - grand_mean)**2
        for d in dom_labels if (dom==d).any()
    )
    ss_wit = sum(
        float(np.sum(w[dom==d] * (y[dom==d] -
              float(np.average(y[dom==d], weights=w[dom==d])))**2))
        for d in dom_labels if (dom==d).any()
    )
    df1_dom = k_dom - 1
    df2_dom = n - k_dom
    if df2_dom > 0 and ss_wit > 0:
        F_dom = round((ss_bet/df1_dom) / (ss_wit/df2_dom), 3)
        p_dom = round(float(scipy_stats.f.sf(F_dom, df1_dom, df2_dom)), 4)
    else:
        F_dom, p_dom = np.nan, np.nan

    # ── Part 2: Per-domain linear age trend ───────────────────────────────
    band_order = list(T1_BANDS.keys())   # 0-4y, 4-8y, 8-12y, 12-18y
    band_index = {b: i for i, b in enumerate(band_order)}

    domain_trends = []
    for d in DOMAIN_ORDER:
        mask  = dom == d
        if mask.sum() < 3:
            domain_trends.append({
                "domain": d,
                "n_bands": int(mask.sum()),
                "slope": np.nan, "se_slope": np.nan,
                "F": np.nan, "p": np.nan,
                "note": "< 3 bands — trend not estimable",
            })
            continue

        yi = y[mask]
        wi = w[mask]
        xi = np.array([band_index[b] for b in bnd[mask]], dtype=float)

        # Weighted linear regression: y ~ b0 + b1*x, weights=wi
        sw   = wi.sum()
        sxw  = (wi * xi).sum()
        syw  = (wi * yi).sum()
        sx2w = (wi * xi**2).sum()
        sxyw = (wi * xi * yi).sum()
        denom = sw * sx2w - sxw**2
        if abs(denom) < 1e-12:
            domain_trends.append({"domain": d, "n_bands": int(mask.sum()),
                                   "slope": np.nan, "F": np.nan, "p": np.nan,
                                   "note": "collinear"})
            continue

        b1     = (sw * sxyw - sxw * syw) / denom
        b0     = (syw - b1 * sxw) / sw
        y_hat  = b0 + b1 * xi
        resid  = yi - y_hat
        df_reg = 1
        df_res = int(mask.sum()) - 2
        ss_reg = float(np.sum(wi * (y_hat - float(np.average(yi,weights=wi)))**2))
        ss_res = float(np.sum(wi * resid**2))
        if df_res > 0 and ss_res > 0:
            F_t = round((ss_reg/df_reg) / (ss_res/df_res), 3)
            p_t = round(float(scipy_stats.f.sf(F_t, df_reg, df_res)), 4)
        else:
            F_t, p_t = np.nan, np.nan
        se_slope = round(float(np.sqrt(sw / (denom + 1e-12))), 6)
        domain_trends.append({
            "domain":   d,
            "n_bands":  int(mask.sum()),
            "slope":    round(float(b1), 4),
            "se_slope": se_slope,
            "F":        F_t,
            "df1":      df_reg,
            "df2":      df_res,
            "p":        p_t,
            "sig":      ("***" if not np.isnan(p_t) and p_t<0.001 else
                         "**"  if not np.isnan(p_t) and p_t<0.01  else
                         "*"   if not np.isnan(p_t) and p_t<0.05  else "ns"),
        })

    return {
        "domain": {"F": F_dom, "df1": df1_dom, "df2": df2_dom, "p": p_dom,
                   "sig": ("***" if not np.isnan(p_dom) and p_dom<0.001 else
                           "**"  if not np.isnan(p_dom) and p_dom<0.01  else
                           "*"   if not np.isnan(p_dom) and p_dom<0.05  else "ns")},
        "domain_trends": domain_trends,
        "n_cells":  n,
        "note": ("Part 1: one-way ANOVA — do domain trajectories differ in level? "
                 "Part 2: per-domain weighted linear regression across age bands — "
                 "does each domain line have a significant age trend? "
                 "Motor x 0-4y excluded. "
                 "Interaction (does slope differ by domain?) tested by GLM on "
                 "individual data (score x Domain x Band)."),
    }


# ── 2. Tukey HSD post-hoc ────────────────────────────────────────────────────

def run_tukey(cells: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Tukey-style comparisons between all domain × band cells.
    Uses SE of difference = sqrt(se_i² + se_j²).
    """
    if cells.empty:
        return pd.DataFrame()

    rows = []
    cell_list = cells.to_dict("records")
    k = len(cell_list)

    for i, j in combinations(range(k), 2):
        ci, cj = cell_list[i], cell_list[j]
        diff   = ci["sqrt_dr2"] - cj["sqrt_dr2"]
        se_d   = np.sqrt(ci["se_sqrt_dr2"]**2 + cj["se_sqrt_dr2"]**2)
        q_stat = abs(diff) / se_d if se_d > 0 else 0.0
        # Approximate p via t-distribution (conservative)
        df_    = min(ci["n"], cj["n"]) - 2
        pval   = float(2 * scipy_stats.t.sf(q_stat, max(df_, 1)))
        rows.append({
            "cell_A":  f"{ci['domain']} · {ci['band']}",
            "cell_B":  f"{cj['domain']} · {cj['band']}",
            "diff":    round(diff, 4),
            "se_diff": round(se_d, 4),
            "q":       round(q_stat, 3),
            "p_raw":   round(pval, 4),
        })

    if not rows:
        return pd.DataFrame()

    df_tukey = pd.DataFrame(rows)
    # Bonferroni correction for number of comparisons
    m     = len(df_tukey)
    df_tukey["p_adj"] = np.clip(df_tukey["p_raw"] * m, 0, 1).round(4)
    df_tukey["sig"]   = df_tukey["p_adj"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01
                  else "*" if p < 0.05 else "ns")
    return df_tukey.sort_values("p_adj").reset_index(drop=True)


# ── 3. Segmented regression ──────────────────────────────────────────────────

def run_segmented(cells: pd.DataFrame) -> dict:
    """
    Rank-order cells by √ΔR² and fit piecewise linear regression.
    Identifies breakpoints in the ranked effect-size profile.
    """
    if cells.empty or len(cells) < 4:
        return {"error": "Not enough cells."}

    ranked = cells.sort_values("sqrt_dr2", ascending=False).reset_index(drop=True)
    x      = ranked.index.values.astype(float)
    y      = ranked["sqrt_dr2"].values.astype(float)
    labels = (ranked["domain"] + " · " + ranked["band"]).tolist()

    # Try all possible single breakpoints and pick the one minimising RSS
    best_bp, best_rss, best_segs = None, np.inf, None
    for bp in range(1, len(x) - 1):
        x1, y1 = x[:bp+1], y[:bp+1]
        x2, y2 = x[bp:],   y[bp:]
        b1 = np.polyfit(x1, y1, 1) if len(x1) >= 2 else None
        b2 = np.polyfit(x2, y2, 1) if len(x2) >= 2 else None
        if b1 is None or b2 is None:
            continue
        rss = (np.sum((y1 - np.polyval(b1, x1))**2) +
               np.sum((y2 - np.polyval(b2, x2))**2))
        if rss < best_rss:
            best_rss = rss
            best_bp  = bp
            best_segs = (b1, b2)

    if best_bp is None:
        return {"error": "Could not fit segmented regression."}

    # F-test: piecewise vs linear
    b_lin = np.polyfit(x, y, 1)
    rss_lin = float(np.sum((y - np.polyval(b_lin, x))**2))
    df1     = 2  # extra params in piecewise
    df2     = len(x) - 4
    f_stat  = ((rss_lin - best_rss) / df1) / (best_rss / max(df2, 1))
    p_val   = float(scipy_stats.f.sf(f_stat, df1, max(df2, 1)))

    seg1_labels = labels[:best_bp+1]
    seg2_labels = labels[best_bp:]

    # Tag each ranked row with its segment
    ranked_records = ranked[["domain","band","sqrt_dr2","se_sqrt_dr2","n"]].copy()
    ranked_records.insert(0, "rank", range(1, len(ranked_records)+1))
    ranked_records["segment"] = ["Segment 1" if i <= best_bp else "Segment 2"
                                  for i in range(len(ranked_records))]

    return {
        "ranked":      ranked_records.to_dict("records"),
        "breakpoint":  best_bp,
        "bp_label":    labels[best_bp],
        "seg1_labels": seg1_labels,
        "seg2_labels": seg2_labels,
        "seg1_slope":  round(float(best_segs[0][0]), 4),
        "seg2_slope":  round(float(best_segs[1][0]), 4),
        "seg1_mean":   round(float(np.mean(y[:best_bp+1])), 4),
        "seg2_mean":   round(float(np.mean(y[best_bp:])), 4),
        "F":           round(f_stat, 3),
        "df1":         df1,
        "df2":         max(df2, 1),
        "p":           round(p_val, 4),
        "x":           x.tolist(),
        "y":           y.tolist(),
        "b_lin":       b_lin.tolist(),
        "b_seg1":      best_segs[0].tolist(),
        "b_seg2":      best_segs[1].tolist(),
    }


# ── 4. Cell-level Bayesian model ─────────────────────────────────────────────

def run_bayes_cell(cells: pd.DataFrame) -> dict:
    """
    Cell-level Bayesian hierarchical model.
    12 data points (√ΔR² per cell), weighted by 1/SE².
    Fixed effects: domain (3 levels), band (4 levels).
    Uses analytical conjugate normal-normal model.
    """
    if cells.empty or len(cells) < 4:
        return {"error": "Not enough cells."}

    y  = cells["sqrt_dr2"].values.astype(float)
    w  = 1.0 / (cells["se_sqrt_dr2"].values.astype(float) ** 2 + 1e-9)

    # Encode domain and band as dummy variables
    doms = pd.Categorical(cells["domain"], categories=DOMAIN_ORDER)
    bnds = pd.Categorical(cells["band"],   categories=list(T1_BANDS.keys()))
    D    = pd.get_dummies(doms, drop_first=True).values.astype(float)
    B    = pd.get_dummies(bnds, drop_first=True).values.astype(float)
    X    = np.column_stack([np.ones(len(y)), D, B])

    # Weighted least squares as Bayesian posterior mode (normal likelihood)
    W   = np.diag(w)
    XtW = X.T @ W
    try:
        cov_post = np.linalg.inv(XtW @ X)
        mean_post = cov_post @ (XtW @ y)
    except Exception:
        return {"error": "Matrix inversion failed in cell-level Bayes."}

    se_post  = np.sqrt(np.diag(cov_post))
    param_names = (["intercept"]
                   + [f"domain_{d}" for d in DOMAIN_ORDER[1:]]
                   + [f"band_{b}"   for b in list(T1_BANDS.keys())[1:]])

    params = []
    for name, mean, se in zip(param_names, mean_post, se_post):
        # 95% credible interval
        ci_lo = mean - 1.96 * se
        ci_hi = mean + 1.96 * se
        # Posterior probability of direction (P > 0 or P < 0)
        z    = mean / se if se > 0 else 0.0
        prob = float(scipy_stats.norm.cdf(abs(z)))
        params.append({
            "parameter": name,
            "mean":      round(float(mean), 4),
            "se":        round(float(se),   4),
            "ci_lo":     round(float(ci_lo),4),
            "ci_hi":     round(float(ci_hi),4),
            "P_direction": round(prob, 3),
        })

    return {
        "params":    params,
        "n_cells":   len(cells),
        "ref_domain": DOMAIN_ORDER[0],
        "ref_band":   list(T1_BANDS.keys())[0],
    }


# ── 4b. Within-domain Bayesian age trend (Option A) ──────────────────────────

def run_bayes_domain_trends(cells: pd.DataFrame) -> dict:
    """
    Fit separate Bayesian normal-normal conjugate models within each domain.
    Each model: sqrt(DeltaR2)_cell ~ 1 + band, weighted by 1/SE^2.
    Gives domain-specific band effects (age trends).

    Sensory-Repetitive: 4 cells (0-4y, 4-8y, 8-12y, 12-18y), ref=0-4y
    Motor:              3 cells (4-8y, 8-12y, 12-18y — 0-4y absent), ref=4-8y
    Social:             4 cells (0-4y, 4-8y, 8-12y, 12-18y), ref=0-4y
    """
    if cells.empty:
        return {"error": "No cells."}

    # Exclude structurally missing
    cells = cells[~cells.apply(
        lambda r: (r["domain"], r["band"]) in MISSING_CELLS, axis=1
    )].copy()

    results = {}
    band_order = list(T1_BANDS.keys())

    for dom in DOMAIN_ORDER:
        sub = cells[cells["domain"] == dom].copy()
        sub = sub.sort_values("band", key=lambda s: s.map(
            {b: i for i, b in enumerate(band_order)}))

        if len(sub) < 2:
            results[dom] = {"error": f"Only {len(sub)} cell(s) — not enough."}
            continue

        y  = sub["sqrt_dr2"].values.astype(float)
        w  = 1.0 / (sub["se_sqrt_dr2"].values.astype(float)**2 + 1e-9)
        bands = sub["band"].values

        # Reference band: first observed band for this domain
        ref_band = bands[0]

        # Build design matrix: intercept + band dummies (drop first = ref)
        other_bands = [b for b in bands if b != ref_band]
        # One-hot for non-reference bands
        X_cols = [np.ones(len(y))]
        for ob in other_bands:
            X_cols.append((bands == ob).astype(float))
        X = np.column_stack(X_cols)
        p = X.shape[1]

        # Weighted analytical posterior
        W    = np.diag(w)
        XtW  = X.T @ W
        try:
            cov_post  = np.linalg.inv(XtW @ X)
            mean_post = cov_post @ (XtW @ y)
        except Exception as e:
            results[dom] = {"error": str(e)}
            continue

        se_post = np.sqrt(np.maximum(0, np.diag(cov_post)))

        param_names = [f"intercept_{ref_band}"] +                       [f"band_{ob}_vs_{ref_band}" for ob in other_bands]

        params = []
        for name, mean, se in zip(param_names, mean_post, se_post):
            ci_lo = mean - 1.96 * se
            ci_hi = mean + 1.96 * se
            z     = mean / se if se > 0 else 0.0
            prob  = float(scipy_stats.norm.cdf(abs(z)))
            params.append({
                "parameter":   name,
                "band":        name.split("band_")[1].split("_vs_")[0]
                               if "band_" in name else ref_band,
                "mean":        round(float(mean), 4),
                "se":          round(float(se),   4),
                "ci_lo":       round(float(ci_lo), 4),
                "ci_hi":       round(float(ci_hi), 4),
                "P_direction": round(prob, 3),
            })

        results[dom] = {
            "params":   params,
            "n_cells":  len(sub),
            "ref_band": ref_band,
            "bands":    list(bands),
        }

    return {
        "by_domain": results,
        "note": ("Separate Bayesian models per domain. "
                 "Band effects = change in sqrt(DeltaR2) relative to "
                 "reference band within each domain. "
                 "Motor ref = 4-8y (0-4y structurally absent). "
                 "Wide CIs for Motor (n=3 cells, df=1)."),
    }


# ── 5. Individual-level Bayesian model ───────────────────────────────────────

def run_bayes_individual(cells: pd.DataFrame) -> dict:
    """
    Individual-level Bayesian hierarchical model.
    Stack all individual observations from all cells.
    Fixed effects: domain, band, domain × band interaction.
    Random effect: instrument combination (via cell identity as proxy).
    Uses analytical WLS posterior (normal likelihood).
    """
    if cells.empty:
        return {"error": "No cells to build individual model."}

    # Stack individual data from each cell
    frames = []
    for _, row in cells.iterrows():
        sub = row.get("_sub")
        if sub is None or not isinstance(sub, pd.DataFrame) or sub.empty:
            continue
        sub = sub.copy()
        sub["_domain"] = row["domain"]
        sub["_band"]   = row["band"]
        frames.append(sub[["_domain","_psych","_band","_age","sex"]])

    if not frames:
        return {"error": "No individual data available. Re-run cell computation."}

    df   = pd.concat(frames, ignore_index=True).dropna(
        subset=["_domain","_psych","_band","_age"])
    n    = len(df)
    if n < 20:
        return {"error": f"Only {n} observations after stacking."}

    y    = df["_psych"].values.astype(float)
    doms = pd.Categorical(df["_domain"], categories=DOMAIN_ORDER)
    bnds = pd.Categorical(df["_band"],   categories=list(T1_BANDS.keys()))
    D    = pd.get_dummies(doms, drop_first=True).values.astype(float)
    B    = pd.get_dummies(bnds, drop_first=True).values.astype(float)
    age  = df["_age"].values.astype(float)
    age  = (age - age.mean()) / (age.std() + 1e-9)

    # Include sex if available
    use_sex = "sex" in df.columns and df["sex"].notna().sum() > 100
    if use_sex:
        sex = pd.to_numeric(df["sex"], errors="coerce").fillna(
            df["sex"].mode()[0] if not df["sex"].mode().empty else 1).values.astype(float)
        X = np.column_stack([np.ones(n), D, B, age, sex])
        covs = ["age_z", "sex"]
    else:
        X = np.column_stack([np.ones(n), D, B, age])
        covs = ["age_z"]

    try:
        XtX     = X.T @ X
        cov_post = np.linalg.inv(XtX)
        mean_post = cov_post @ (X.T @ y)
    except Exception:
        return {"error": "Matrix inversion failed in individual-level Bayes."}

    resid   = y - X @ mean_post
    sigma2  = float(np.var(resid, ddof=X.shape[1]))
    se_post = np.sqrt(np.diag(cov_post) * sigma2)

    param_names = (["intercept"]
                   + [f"domain_{d}" for d in DOMAIN_ORDER[1:]]
                   + [f"band_{b}"   for b in list(T1_BANDS.keys())[1:]]
                   + covs)

    params = []
    for name, mean, se in zip(param_names, mean_post, se_post):
        ci_lo = mean - 1.96 * se
        ci_hi = mean + 1.96 * se
        z     = mean / se if se > 0 else 0.0
        prob  = float(scipy_stats.norm.cdf(abs(z)))
        params.append({
            "parameter":   name,
            "mean":        round(float(mean), 4),
            "se":          round(float(se),   4),
            "ci_lo":       round(float(ci_lo),4),
            "ci_hi":       round(float(ci_hi),4),
            "P_direction": round(prob, 3),
        })

    return {
        "params":     params,
        "n_obs":      n,
        "ref_domain": DOMAIN_ORDER[0],
        "ref_band":   list(T1_BANDS.keys())[0],
        "sigma":      round(float(np.sqrt(sigma2)), 4),
    }


# ── Master run ────────────────────────────────────────────────────────────────

def run_all(merged: pd.DataFrame) -> dict:
    cells = compute_cells(merged)
    if cells.empty:
        return {"error": "No cells computed — check CBCL and domain data."}

    # Store _sub separately before serialisation
    sub_store = {}
    for _, row in cells.iterrows():
        key = (row["domain"], row["band"])
        sub_store[key] = row.get("_sub")
    cells_clean = cells.drop(columns=["_sub"], errors="ignore")

    # Re-attach for individual Bayes
    cells_with_sub = cells_clean.copy()
    cells_with_sub["_sub"] = [
        sub_store.get((r["domain"], r["band"]))
        for _, r in cells_with_sub.iterrows()
    ]

    return {
        "cells":      cells_clean,
        "anova":      run_anova(cells_clean),
        "tukey":      run_tukey(cells_clean),
        "segmented":  run_segmented(cells_clean),
        "bayes_cell": run_bayes_cell(cells_clean),
        "bayes_ind":  run_bayes_individual(cells_with_sub),
        "bayes_domain_trends": run_bayes_domain_trends(cells_clean),
        "glm":        run_glm(cells_with_sub),
        "glm_logistic": run_glm_logistic(cells_with_sub),
        "mixed":      run_mixed_models(cells_with_sub),
    }


def run_glm(cells: pd.DataFrame) -> dict:
    """
    Individual-level GLM on all stacked observations (same data as
    run_bayes_individual). Each row is one person with a domain score
    and a CBCL score. Domain and Band are categorical moderators.

    Model: CBCL ~ domain_score * Domain * Band + age + sex

    Type III SS tests:
      domain_score            : overall domain-CBCL association
      score × Domain          : does slope differ by domain?
      score × Band            : does slope change across development?
      score × Domain × Band   : do developmental trajectories differ by domain?

    Thousands of observations -> proper df_resid.
    """
    if cells.empty:
        return {"error": "No cells available."}

    frames = []
    for _, row in cells.iterrows():
        sub = row.get("_sub")
        if sub is None or not isinstance(sub, pd.DataFrame) or sub.empty:
            continue
        sub = sub.copy()
        sub["_domain_label"] = row["domain"]
        sub["_band_label"]   = row["band"]
        frames.append(sub[["_domain","_psych","_domain_label",
                            "_band_label","_age","sex"]])

    if not frames:
        return {"error": "No individual data. Re-run cell computation."}

    df = pd.concat(frames, ignore_index=True).dropna(
        subset=["_domain","_psych","_band_label","_age"])
    n  = len(df)
    if n < 50:
        return {"error": f"Only {n} observations — need ≥ 50."}

    y = df["_psych"].values.astype(float)
    x = df["_domain"].values.astype(float)
    x = (x - x.mean()) / (x.std() + 1e-9)

    doms = pd.Categorical(df["_domain_label"], categories=DOMAIN_ORDER)
    bnds = pd.Categorical(df["_band_label"],   categories=list(T1_BANDS.keys()))
    D    = pd.get_dummies(doms, drop_first=True).values.astype(float)
    B    = pd.get_dummies(bnds, drop_first=True).values.astype(float)

    xD  = np.column_stack([x.reshape(-1,1) * D[:,i:i+1]
                           for i in range(D.shape[1])])
    xB  = np.column_stack([x.reshape(-1,1) * B[:,j:j+1]
                           for j in range(B.shape[1])])
    xDB = np.column_stack([x.reshape(-1,1) * D[:,i:i+1] * B[:,j:j+1]
                           for i in range(D.shape[1])
                           for j in range(B.shape[1])])

    age = df["_age"].values.astype(float)
    age = (age - age.mean()) / (age.std() + 1e-9)

    use_sex = "sex" in df.columns and df["sex"].notna().sum() > 100
    if use_sex:
        sex  = pd.to_numeric(df["sex"],errors="coerce").fillna(1).values.astype(float)
        covs = np.column_stack([age, sex]); cov_names = ["age_z","sex"]
    else:
        covs = age.reshape(-1,1); cov_names = ["age_z"]

    X_full = np.column_stack([np.ones(n), x, D, B, xD, xB, xDB, covs])
    p_full = X_full.shape[1]

    def _ols(X):
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        ss = float(np.sum((y - X @ b)**2))
        return b, ss

    b_full, ss_full = _ols(X_full)
    df_err = n - p_full
    ms_err = ss_full / df_err

    cx_s,  cx_e   = 1,        2
    cD_s,  cD_e   = 2,        2+D.shape[1]
    cB_s,  cB_e   = cD_e,     cD_e+B.shape[1]
    cxD_s, cxD_e  = cB_e,     cB_e+xD.shape[1]
    cxB_s, cxB_e  = cxD_e,    cxD_e+xB.shape[1]
    cxDB_s,cxDB_e = cxB_e,    cxB_e+xDB.shape[1]

    def _type3(s, e):
        keep = list(range(s)) + list(range(e, p_full))
        _, ss_red = _ols(X_full[:, keep])
        ss_eff = max(ss_red - ss_full, 0.0)
        df_eff = e - s
        F = (ss_eff / df_eff) / ms_err
        p = float(scipy_stats.f.sf(F, df_eff, df_err))
        return {"F": round(F,3), "df1": df_eff, "df2": df_err,
                "p": round(p,4),
                "sig": ("***" if p<0.001 else "**" if p<0.01
                        else "*" if p<0.05 else "ns")}

    other_doms = [d for d in DOMAIN_ORDER    if d != DOMAIN_ORDER[0]]
    other_bnds = [b for b in T1_BANDS.keys() if b != list(T1_BANDS.keys())[0]]

    coef_names = (["intercept","domain_score"]
                  + [f"domain_{d}"    for d in other_doms]
                  + [f"band_{b}"      for b in other_bnds]
                  + [f"score×{d}"     for d in other_doms]
                  + [f"score×{b}"     for b in other_bnds]
                  + [f"score×{d}×{b}" for d in other_doms for b in other_bnds]
                  + cov_names)

    se_arr = np.sqrt(np.maximum(0,
               np.diag(np.linalg.pinv(X_full.T @ X_full))) * ms_err)
    t_arr  = b_full / (se_arr + 1e-12)
    p_coef = [float(2*scipy_stats.t.sf(abs(t), df_err)) for t in t_arr]

    coef_rows = [
        {"parameter": coef_names[i],
         "estimate":  round(float(b_full[i]),4),
         "se":        round(float(se_arr[i]),4),
         "t":         round(float(t_arr[i]),3),
         "p":         round(float(p_coef[i]),4),
         "sig":       ("***" if p_coef[i]<0.001 else "**" if p_coef[i]<0.01
                        else "*" if p_coef[i]<0.05 else "ns")}
        for i in range(len(b_full))
    ]

    return {
        "domain_score":      _type3(cx_s,   cx_e),
        "domain_main":       _type3(cD_s,   cD_e),
        "band_main":         _type3(cB_s,   cB_e),
        "score×domain":      _type3(cxD_s,  cxD_e),
        "score×band":        _type3(cxB_s,  cxB_e),
        "score×domain×band": _type3(cxDB_s, cxDB_e),
        "coefficients":      coef_rows,
        "n_obs":             n,
        "df_resid":          df_err,
        "ref_domain":        DOMAIN_ORDER[0],
        "ref_band":          list(T1_BANDS.keys())[0],
        "model":             "CBCL ~ domain_score × Domain × Band + age + sex",
    }


# ── 7. Logistic GLM: P(late_CBCL elevated) ~ z_score × Domain × Band ─────────


def run_glm_logistic(cells: pd.DataFrame) -> dict:
    """
    Population-level logistic GLM.

    Outcome: P(late_CBCL elevated)
      elevated = 1 if a person's CBCL composite (any age) ≥
                 band_mean_CBCL + 1 SD of the late CBCL distribution.
      Threshold is population-level: defined from the distribution of
      late (12–18y) CBCL scores, shifted by the band-level mean.
      This asks: given childhood domain severity and the typical
      psychopathology level at that developmental stage, what is the
      probability of ending up elevated in adolescence?

    Model:
      logit P(elevated) ~ z_score × Domain × Band
                        + band_mean_CBCL   (population baseline)
                        + age_z            (age within band, standardized)
                        + sex

    Data: same stacked _sub frames as run_glm / run_bayes_individual.
    Population assumption: each (domain_score, CBCL) pair is a
    population-level draw — no within-person pairing required.
    """
    if cells.empty:
        return {"error": "No cells available."}

    frames = []
    for _, row in cells.iterrows():
        sub = row.get("_sub")
        if sub is None or not isinstance(sub, pd.DataFrame) or sub.empty:
            continue
        sub = sub.copy()
        sub["_domain_label"] = row["domain"]
        sub["_band_label"]   = row["band"]
        frames.append(sub[["_domain","_psych","_domain_label",
                            "_band_label","_age","sex"]])

    if not frames:
        return {"error": "No individual data. Re-run cell computation."}

    df = pd.concat(frames, ignore_index=True).dropna(
        subset=["_domain","_psych","_band_label","_age"])
    n  = len(df)
    if n < 50:
        return {"error": f"Only {n} observations — need ≥ 50."}

    # ── Band-level mean CBCL (population baseline per band) ───────────────
    band_mean_cbcl = df.groupby("_band_label")["_psych"].mean().to_dict()
    df["_band_mean_cbcl"] = df["_band_label"].map(band_mean_cbcl)

    # ── Late CBCL threshold: band_mean + 1 SD of full CBCL distribution ──
    cbcl_sd = float(df["_psych"].std(ddof=1))
    df["_elevated"] = (df["_psych"] >= df["_band_mean_cbcl"] + cbcl_sd
                       ).astype(float)
    n_elevated = int(df["_elevated"].sum())
    prev_overall = round(float(df["_elevated"].mean()) * 100, 1)

    # ── Predictors ────────────────────────────────────────────────────────
    y = df["_elevated"].values.astype(float)
    x = df["_domain"].values.astype(float)
    x = (x - x.mean()) / (x.std() + 1e-9)          # standardized domain score

    doms = pd.Categorical(df["_domain_label"], categories=DOMAIN_ORDER)
    bnds = pd.Categorical(df["_band_label"],   categories=list(T1_BANDS.keys()))
    D    = pd.get_dummies(doms, drop_first=True).values.astype(float)
    B    = pd.get_dummies(bnds, drop_first=True).values.astype(float)

    xD  = np.column_stack([x.reshape(-1,1) * D[:,i:i+1]
                           for i in range(D.shape[1])])
    xB  = np.column_stack([x.reshape(-1,1) * B[:,j:j+1]
                           for j in range(B.shape[1])])
    xDB = np.column_stack([x.reshape(-1,1) * D[:,i:i+1] * B[:,j:j+1]
                           for i in range(D.shape[1])
                           for j in range(B.shape[1])])

    age  = df["_age"].values.astype(float)
    age  = (age - age.mean()) / (age.std() + 1e-9)
    bmc  = df["_band_mean_cbcl"].values.astype(float)
    bmc  = (bmc - bmc.mean()) / (bmc.std() + 1e-9)  # standardize baseline

    use_sex = "sex" in df.columns and df["sex"].notna().sum() > 100
    if use_sex:
        sex  = pd.to_numeric(df["sex"],errors="coerce").fillna(1).values.astype(float)
        covs = np.column_stack([bmc, age, sex])
        cov_names = ["band_mean_CBCL_z","age_z","sex"]
    else:
        covs = np.column_stack([bmc, age])
        cov_names = ["band_mean_CBCL_z","age_z"]

    X = np.column_stack([np.ones(n), x, D, B, xD, xB, xDB, covs])

    # ── Logistic regression via IRLS ──────────────────────────────────────
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _irls(X, y, max_iter=200, tol=1e-7):
        b = np.zeros(X.shape[1])
        for _ in range(max_iter):
            mu   = _sigmoid(X @ b)
            mu   = np.clip(mu, 1e-9, 1 - 1e-9)
            W    = mu * (1 - mu)
            z    = X @ b + (y - mu) / W
            Xw   = X * np.sqrt(W)[:,None]
            zw   = z * np.sqrt(W)
            b_new, _, _, _ = np.linalg.lstsq(Xw, zw, rcond=None)
            if np.max(np.abs(b_new - b)) < tol:
                b = b_new
                break
            b = b_new
        mu   = _sigmoid(X @ b)
        mu   = np.clip(mu, 1e-9, 1 - 1e-9)
        W    = mu * (1 - mu)
        Xw   = X * np.sqrt(W)[:,None]
        cov_b = np.linalg.pinv(Xw.T @ Xw)
        se_b  = np.sqrt(np.maximum(0, np.diag(cov_b)))
        return b, se_b, mu

    b, se_b, mu_hat = _irls(X, y)

    # ── Coefficients ──────────────────────────────────────────────────────
    other_doms = [d for d in DOMAIN_ORDER    if d != DOMAIN_ORDER[0]]
    other_bnds = [b_ for b_ in T1_BANDS.keys() if b_ != list(T1_BANDS.keys())[0]]

    coef_names = (["intercept", "domain_score"]
                  + [f"domain_{d}"    for d in other_doms]
                  + [f"band_{b_}"     for b_ in other_bnds]
                  + [f"score×{d}"     for d in other_doms]
                  + [f"score×{b_}"    for b_ in other_bnds]
                  + [f"score×{d}×{b_}"for d in other_doms for b_ in other_bnds]
                  + cov_names)

    z_arr  = b / (se_b + 1e-12)
    p_arr  = [float(2 * scipy_stats.norm.sf(abs(z))) for z in z_arr]
    or_arr = np.exp(b)

    coef_rows = [
        {"parameter":  coef_names[i],
         "log_OR":     round(float(b[i]),    4),
         "OR":         round(float(or_arr[i]),4),
         "se":         round(float(se_b[i]), 4),
         "z":          round(float(z_arr[i]),3),
         "p":          round(float(p_arr[i]),4),
         "sig":        ("***" if p_arr[i]<0.001 else "**" if p_arr[i]<0.01
                         else "*" if p_arr[i]<0.05 else "ns")}
        for i in range(len(b))
    ]

    # ── Predicted probabilities at z_score = -1, 0, +1 SD ────────────────
    # for each domain × band combination (observed cells only)
    pred_rows = []
    ref_dom   = DOMAIN_ORDER[0]
    ref_band  = list(T1_BANDS.keys())[0]
    b_map     = dict(zip(coef_names, b))

    for dom in DOMAIN_ORDER:
        for band in T1_BANDS.keys():
            if (dom, band) in MISSING_CELLS:
                continue   # Motor × 0–4y structurally absent
            for z_val in [-1.0, 0.0, 1.0, 2.0]:
                # Build linear predictor for this cell
                lp = b_map.get("intercept", 0)
                lp += z_val * b_map.get("domain_score", 0)
                if dom != ref_dom:
                    lp += b_map.get(f"domain_{dom}", 0)
                    lp += z_val * b_map.get(f"score×{dom}", 0)
                if band != ref_band:
                    lp += b_map.get(f"band_{band}", 0)
                    lp += z_val * b_map.get(f"score×{band}", 0)
                if dom != ref_dom and band != ref_band:
                    lp += z_val * b_map.get(f"score×{dom}×{band}", 0)
                # band_mean_CBCL_z at 0 (population average baseline)
                # age_z at 0, sex at population mean (~1.5)
                prob = float(_sigmoid(lp))
                pred_rows.append({
                    "domain":     dom,
                    "band":       band,
                    "z_score":    z_val,
                    "prob_elevated": round(prob, 4),
                    "pct_elevated":  round(prob * 100, 1),
                })

    return {
        "coefficients":    coef_rows,
        "predictions":     pred_rows,
        "n_obs":           n,
        "n_elevated":      n_elevated,
        "prevalence_pct":  prev_overall,
        "cbcl_sd":         round(cbcl_sd, 4),
        "threshold_note":  (f"Elevated = CBCL ≥ band_mean + {cbcl_sd:.3f} "
                            f"(1 SD). Prevalence = {prev_overall}% "
                            f"({n_elevated:,}/{n:,})."),
        "model":           ("logit P(late_CBCL elevated) ~ "
                            "z_score × Domain × Band + "
                            "band_mean_CBCL + age + sex"),
    }


# ── 8. Population GLM + Mixed-effects model ──────────────────────────────────

def run_mixed_models(cells: pd.DataFrame) -> dict:
    """
    Population GLM and Mixed-effects LMM on the 11 OBSERVED cells only.

    Motor × 0-4y is structurally absent (DCDQ not administered under 4y)
    and is EXCLUDED from the data stack entirely. This prevents the
    factorial parameterization from extrapolating an unobserved cell and
    influencing the estimation of the other Motor coefficients.

    Cell-level slopes are also estimated directly within each cell
    (CBCL ~ z_score + z_age + sex) as a robustness check.

    Model 1: Population GLM  CBCL ~ z_score * Domain * Band + z_age + sex
    Model 2: Mixed LMM       same + (1|child_id)  via statsmodels REML
    Both fitted on the 11-cell stack (Motor x 0-4y excluded).
    """
    if cells.empty:
        return {"error": "No cells available."}

    frames = []
    cell_slopes = []   # direct within-cell slope estimates

    for _, row in cells.iterrows():
        dom  = row["domain"]
        band = row["band"]

        # Skip structurally missing cells
        if (dom, band) in MISSING_CELLS:
            continue

        sub = row.get("_sub")
        if sub is None or not isinstance(sub, pd.DataFrame) or sub.empty:
            continue

        s = sub.copy()
        s["_domain_label"] = dom
        s["_band_label"]   = band
        s["_child_id"]     = s.index.astype(str)
        frames.append(s[["_domain","_psych","_domain_label",
                          "_band_label","_age","sex","_child_id"]])

        # ── Direct within-cell slope ──────────────────────────────────────
        sc = s.dropna(subset=["_domain","_psych","_age"])
        if len(sc) > 10:
            xc   = sc["_domain"].values.astype(float)
            xc   = (xc - xc.mean()) / (xc.std() + 1e-9)
            yc   = sc["_psych"].values.astype(float)
            ac   = sc["_age"].values.astype(float)
            ac   = (ac - ac.mean()) / (ac.std() + 1e-9)
            use_sex_c = "sex" in sc.columns and sc["sex"].notna().sum() > 10
            if use_sex_c:
                sx = pd.to_numeric(sc["sex"],errors="coerce").fillna(1).values.astype(float)
                Xc = np.column_stack([np.ones(len(xc)), xc, ac, sx])
            else:
                Xc = np.column_stack([np.ones(len(xc)), xc, ac])
            bc, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
            rc   = yc - Xc @ bc
            dfc  = len(xc) - Xc.shape[1]
            msc  = float(np.sum(rc**2)) / max(dfc, 1)
            cvc  = np.linalg.pinv(Xc.T @ Xc) * msc
            sec  = np.sqrt(max(0, cvc[1,1]))
            cell_slopes.append({
                "domain":        dom,
                "band":          band,
                "slope_per_1SD": round(float(bc[1]),4),
                "se":            round(sec,4),
                "ci_lo":         round(float(bc[1])-1.96*sec,4),
                "ci_hi":         round(float(bc[1])+1.96*sec,4),
                "n":             len(sc),
            })

    if not frames:
        return {"error": "No individual data. Re-run cell computation."}

    df = pd.concat(frames, ignore_index=False).reset_index(drop=True)
    df = df.dropna(subset=["_domain","_psych","_band_label","_age"])
    n        = len(df)
    n_people = df["_child_id"].nunique()
    if n < 50:
        return {"error": f"Only {n} observations — need >= 50."}

    # Guard: sequential integer IDs = reset index, not participant IDs
    try:
        sample_ids = df["_child_id"].unique()[:200]
        id_ints    = sorted([int(x) for x in sample_ids])
        if id_ints == list(range(len(id_ints))):
            return {"error": (
                "child_id appears to be sequential row numbers. "
                "The merged DataFrame index may have been reset.")}
    except (ValueError, TypeError):
        pass

    # ── Standardize predictors ────────────────────────────────────────────
    df = df.copy()
    df["z_score"] = ((df["_domain"] - df["_domain"].mean()) /
                     (df["_domain"].std() + 1e-9))
    df["z_age"]   = ((df["_age"] - df["_age"].mean()) /
                     (df["_age"].std() + 1e-9))

    # Sanitize labels for statsmodels formula
    band_clean = {
        "0–4y": "b0_4y", "4–8y": "b4_8y",
        "8–12y": "b8_12y", "12–18y": "b12_18y",
    }
    dom_clean = {
        "Sensory-Repetitive": "Sensory_Rep",
        "Motor": "Motor", "Social": "Social",
    }
    df["domain_f"] = df["_domain_label"].map(dom_clean).fillna(
        df["_domain_label"].str.replace(r"[^A-Za-z0-9]","_",regex=True))
    df["band_f"]   = df["_band_label"].map(band_clean).fillna(
        df["_band_label"].str.replace(r"[^A-Za-z0-9]","_",regex=True))

    # Observed domain×band combinations only (no Motor×b0_4y)
    obs_doms  = df["domain_f"].unique().tolist()
    obs_bands = df["band_f"].unique().tolist()
    df["domain_f"] = pd.Categorical(df["domain_f"], categories=obs_doms)
    df["band_f"]   = pd.Categorical(df["band_f"],   categories=obs_bands)

    use_sex = "sex" in df.columns and df["sex"].notna().sum() > 100
    if use_sex:
        df["sex"] = pd.to_numeric(df["sex"], errors="coerce").fillna(1)

    # Reference: Sensory_Rep x b0_4y (first alphabetically after Categorical sort)
    ref_dom_f  = dom_clean[DOMAIN_ORDER[0]]
    ref_band_f = band_clean[list(T1_BANDS.keys())[0]]

    # Observed other levels (Motor b0_4y excluded from band for Motor rows)
    other_doms_f  = [dom_clean[d] for d in DOMAIN_ORDER
                     if d != DOMAIN_ORDER[0]]
    other_bands_f = [band_clean[b] for b in T1_BANDS.keys()
                     if b != list(T1_BANDS.keys())[0]]
    other_doms    = [d for d in DOMAIN_ORDER if d != DOMAIN_ORDER[0]]
    other_bands   = [b for b in T1_BANDS.keys() if b != list(T1_BANDS.keys())[0]]

    coef_names_base = (["intercept","z_score"]
                       + [f"domain_{d}"         for d in other_doms]
                       + [f"band_{b}"            for b in other_bands]
                       + [f"score×{d}"      for d in other_doms]
                       + [f"score×{b}"      for b in other_bands]
                       + [f"score×{d}×{b}" for d in other_doms
                                                   for b in other_bands]
                       + (["z_age","sex"] if use_sex else ["z_age"]))

    # ── Design matrix ─────────────────────────────────────────────────────
    D    = pd.get_dummies(df["domain_f"], drop_first=True).values.astype(float)
    B    = pd.get_dummies(df["band_f"],   drop_first=True).values.astype(float)
    x    = df["z_score"].values.astype(float)
    y    = df["_psych"].values.astype(float)
    age  = df["z_age"].values.astype(float)

    xD   = np.column_stack([x.reshape(-1,1)*D[:,i:i+1] for i in range(D.shape[1])])
    xB   = np.column_stack([x.reshape(-1,1)*B[:,j:j+1] for j in range(B.shape[1])])
    xDB  = np.column_stack([x.reshape(-1,1)*D[:,i:i+1]*B[:,j:j+1]
                            for i in range(D.shape[1])
                            for j in range(B.shape[1])])
    if use_sex:
        covs = np.column_stack([age, df["sex"].values.astype(float)])
    else:
        covs = age.reshape(-1,1)

    X  = np.column_stack([np.ones(n), x, D, B, xD, xB, xDB, covs])
    p  = X.shape[1]

    b_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid_ols = y - X @ b_ols
    df_ols    = n - p
    ms_ols    = float(np.sum(resid_ols**2)) / df_ols
    cov_ols   = np.linalg.pinv(X.T @ X) * ms_ols
    se_ols    = np.sqrt(np.maximum(0, np.diag(cov_ols)))
    t_ols     = b_ols / (se_ols + 1e-12)
    p_ols     = [float(2*scipy_stats.t.sf(abs(t), df_ols)) for t in t_ols]
    ll_ols    = float(-0.5*n*np.log(2*np.pi*ms_ols) - n/2)
    aic_ols   = -2*ll_ols + 2*p

    def _coef_list(b, se, t_arr, p_arr, label):
        ci_lo = b - 1.96*se; ci_hi = b + 1.96*se
        return [
            {"parameter": coef_names_base[i] if i<len(coef_names_base)
                          else f"param_{i}",
             "estimate":  round(float(b[i]),4),
             "se":        round(float(se[i]),4),
             "ci_lo":     round(float(ci_lo[i]),4),
             "ci_hi":     round(float(ci_hi[i]),4),
             "t":         round(float(t_arr[i]),3),
             "p":         round(float(p_arr[i]),4),
             "sig":       ("***" if p_arr[i]<0.001 else
                           "**"  if p_arr[i]<0.01  else
                           "*"   if p_arr[i]<0.05  else "ns"),
             "model": label}
            for i in range(len(b))
        ]

    def _model_slopes(b_vec, se_vec):
        """Reconstruct slopes for 11 observed cells only."""
        b_map  = dict(zip(coef_names_base, b_vec))
        se_map = dict(zip(coef_names_base, se_vec))
        ref_dom  = DOMAIN_ORDER[0]
        ref_band = list(T1_BANDS.keys())[0]
        rows = []
        for dom in DOMAIN_ORDER:
            for band in T1_BANDS.keys():
                if (dom, band) in MISSING_CELLS:
                    continue   # exclude Motor x 0-4y entirely
                slope = b_map.get("z_score", 0)
                keys  = ["z_score"]
                if dom != ref_dom:
                    slope += b_map.get(f"score×{dom}", 0)
                    keys.append(f"score×{dom}")
                if band != ref_band:
                    slope += b_map.get(f"score×{band}", 0)
                    keys.append(f"score×{band}")
                if dom != ref_dom and band != ref_band:
                    slope += b_map.get(f"score×{dom}×{band}", 0)
                    keys.append(f"score×{dom}×{band}")
                se_slope = float(np.sqrt(
                    sum(se_map.get(k,0)**2 for k in keys)))
                rows.append({
                    "domain":        dom, "band": band,
                    "slope_per_1SD": round(float(slope),4),
                    "se":            round(se_slope,4),
                    "ci_lo":         round(float(slope)-1.96*se_slope,4),
                    "ci_hi":         round(float(slope)+1.96*se_slope,4),
                })
        return rows

    m1_coefs = _coef_list(b_ols, se_ols, t_ols, p_ols, "Population GLM")

    # ── Model 2: statsmodels mixedlm (simplified for ICC) ─────────────────
    try:
        import statsmodels.formula.api as smf
        import threading

        sex_term   = "+ sex" if use_sex else ""
        formula_vc = (f"_psych ~ z_score + domain_f + band_f "
                      f"+ z_age {sex_term}")

        fit_result = {}
        fit_error  = {}

        def _fit():
            try:
                lme    = smf.mixedlm(formula_vc, data=df,
                                     groups=df["_child_id"])
                result = lme.fit(reml=True, method="lbfgs", maxiter=200)
                fit_result["fit"] = result
            except Exception as e:
                fit_error["err"] = str(e)

        t = threading.Thread(target=_fit, daemon=True)
        t.start()
        t.join(timeout=90)

        if t.is_alive():
            raise TimeoutError("Mixed model timed out after 90s.")
        if "err" in fit_error:
            raise RuntimeError(fit_error["err"])

        lme_fit  = fit_result["fit"]
        vc       = lme_fit.cov_re
        sigma2_u = float(vc.iloc[0,0]) if hasattr(vc,"iloc") else float(vc)
        sigma2_e = float(lme_fit.scale)
        icc      = sigma2_u/(sigma2_u+sigma2_e) if (sigma2_u+sigma2_e)>0 else 0
        n_bar    = n/n_people if n_people>0 else 1
        deff     = max(1.0+(n_bar-1)*icc, 1.0)

        se_lme = se_ols * np.sqrt(deff)
        t_lme  = b_ols / (se_lme + 1e-12)
        p_lme  = [float(2*scipy_stats.t.sf(abs(t), df_ols)) for t in t_lme]

        m2 = {
            "coefficients": _coef_list(b_ols, se_lme, t_lme, p_lme,
                                       "Mixed-effects (DEFF-corrected)"),
            "slopes":       _model_slopes(b_ols, se_lme),
            "ll":           round(float(lme_fit.llf), 2),
            "aic":          round(float(lme_fit.aic), 2),
            "sigma2_e":     round(sigma2_e, 4),
            "sigma2_u":     round(sigma2_u, 4),
            "icc":          round(icc, 4),
            "deff":         round(deff, 4),
            "n_bar":        round(n_bar, 2),
            "method":       (f"ICC from statsmodels mixedlm REML. "
                             f"Fixed effects from OLS, SEs x sqrt(DEFF). "
                             f"DEFF={deff:.3f}, ICC={icc:.3f}, n_bar={n_bar:.1f}. "
                             f"Motor x 0-4y excluded (structurally missing)."),
        }

    except ImportError:
        m2 = {"error": "statsmodels not installed. Run: pip install statsmodels",
              "coefficients": [], "slopes": []}
    except Exception as e:
        import traceback
        print(f"[Mixed model error]\n{traceback.format_exc()}")
        m2 = {"error": f"Mixed model failed: {str(e)[:400]}",
              "coefficients": [], "slopes": []}

    return {
        "glm":         {"coefficients": m1_coefs,
                        "slopes":       _model_slopes(b_ols, se_ols),
                        "cell_slopes":  cell_slopes,
                        "ll":           round(ll_ols,2),
                        "aic":          round(aic_ols,2),
                        "sigma2_e":     round(ms_ols,4),
                        "sigma2_u":     None,
                        "icc":          None},
        "mixed":       m2,
        "cell_slopes": cell_slopes,
        "n_obs":       n,
        "n_people":    n_people,
        "model":       ("CBCL ~ z_score * Domain * Band + z_age + sex "
                        "[+ (1|child_id)]. 11 observed cells; "
                        "Motor x 0-4y excluded."),
    }
