"""
modules/domains.py
─────────────────────────────────────────────────────────────────────────────
Cross-instrument domain groupings for the √ΔR² fingerprint matrix.

Because √ΔR² is scale-invariant (all values are semi-partial correlations,
range −1 to +1), features from different instruments measuring the same
construct can be placed together in the same domain band and compared
directly. This is the key advantage over raw regression coefficients.

Domain structure
----------------
Sensory         : SP quadrants, SEQ patterns, ISQ subscales, RBS-R Sensory
Motor           : DCDQ subscales
Social          : SCQ Social/Communication, ADOS Social Affect, CSS-SA
Repetitive/RRB  : RBS-R subscales (Obsessive, Sameness, Ritualistic, etc.),
                  ADOS RRB, CSS-RRB
Anxiety         : CBCL Anxious/Dep, Somatic, Withdrawn, Internalizing
Externalizing   : CBCL Externalising, Aggression, Rule-Breaking
ADHD            : CBCL Attention
Social Problems : CBCL Social Problems, Thought Problems
ASD Severity    : CSS-Total, SCQ total (if used as outcome)
Other           : Anything not matched above
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Domain → colour mapping
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_COLORS: dict[str, str] = {
    "Sensory":          "#fb923c",   # orange
    "Motor":            "#38bdf8",   # sky blue
    "Social":           "#34d399",   # emerald
    "Repetitive/RRB":   "#f472b6",   # pink
    "Anxiety":          "#a78bfa",   # violet
    "Externalising":    "#f87171",   # red
    "ADHD":             "#fbbf24",   # amber
    "Social Problems":  "#67e8f9",   # cyan
    "ASD Severity":     "#94a3b8",   # slate
    "Other":            "#64748b",   # muted
}

# Display order for domains in the grouped heatmap
DOMAIN_ORDER: list[str] = [
    "Sensory",
    "Motor",
    "Social",
    "Repetitive/RRB",
    "Anxiety",
    "Externalising",
    "ADHD",
    "Social Problems",
    "ASD Severity",
    "Other",
]

# ─────────────────────────────────────────────────────────────────────────────
# Feature → domain map
# Keys are the column names that appear in the merged DataFrame.
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DOMAIN: dict[str, str] = {

    # ── Sensory ──────────────────────────────────────────────────────────────
    # RBS-R Sensory subscale
    "rbs_Sensory":           "Sensory",

    # SP / AASP quadrants (stored as sp_* in merged)
    "sp_low_reg":            "Sensory",
    "sp_sensitivity":        "Sensory",
    "sp_avoiding":           "Sensory",
    "sp_seeking":            "Sensory",

    # SEQ-3 patterns (stored as seq_* in sensory-store)
    "seq_hyper":             "Sensory",
    "seq_hypo":              "Sensory",
    "seq_enhanced":          "Sensory",
    "seq_seeking":           "Sensory",

    # ISQ subscales (stored as isq_* in sensory-store)
    "isq_noticing":          "Sensory",
    "isq_interpreting":      "Sensory",
    "isq_acting":            "Sensory",

    # SCQ domains — split across constructs
    "scq_Social":            "Social",
    "scq_Communication":     "Social",
    "scq_Sensory":           "Sensory",   # sensory behaviors in ASD context

    # ── Motor ─────────────────────────────────────────────────────────────────
    "dcdq_Fine Motor":       "Motor",
    "dcdq_Gross Motor":      "Motor",
    "dcdq_Coordination":     "Motor",

    # ── Social ───────────────────────────────────────────────────────────────
    "scq_Social":            "Social",
    "scq_Communication":     "Social",
    "ados_Social Affect":    "Social",
    "css_sa":                "Social",

    # ── Repetitive / RRB ─────────────────────────────────────────────────────
    "rbs_Obsessive":         "Repetitive/RRB",
    "rbs_Sameness":          "Repetitive/RRB",
    "rbs_Ritualistic":       "Repetitive/RRB",
    "rbs_Stereotyped":       "Repetitive/RRB",
    "rbs_SIB":               "Repetitive/RRB",
    "ados_RRB":              "Repetitive/RRB",
    "css_rrb":               "Repetitive/RRB",

    # ── Anxiety / Internalising ───────────────────────────────────────────────
    "cbcl_Anxious/Dep.":     "Anxiety",
    "cbcl_Somatic":          "Anxiety",
    "cbcl_Withdrawn":        "Anxiety",
    "cbcl_Internalizing":    "Anxiety",

    # ── Externalising ─────────────────────────────────────────────────────────
    "cbcl_Externalizing":    "Externalising",
    "cbcl_Aggression":       "Externalising",
    "cbcl_Rule-Breaking":    "Externalising",
    "cbcl_Delinquent":       "Externalising",

    # ── ADHD ─────────────────────────────────────────────────────────────────
    "cbcl_Attention":        "ADHD",

    # ── Social Problems ──────────────────────────────────────────────────────
    "cbcl_Social Prob.":     "Social Problems",
    "cbcl_Thought Prob.":    "Social Problems",

    # ── ASD Severity ─────────────────────────────────────────────────────────
    "css_total":             "ASD Severity",
    "cbcl_Total":            "ASD Severity",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def get_domain(feature: str) -> str:
    """Return domain label for a feature column name, falling back to 'Other'."""
    return FEATURE_DOMAIN.get(feature, "Other")


def sort_by_domain(features: list[str]) -> list[str]:
    """Sort feature list by DOMAIN_ORDER, then alphabetically within domain."""
    order = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    return sorted(
        features,
        key=lambda f: (order.get(get_domain(f), len(DOMAIN_ORDER)), f),
    )


def domain_band_shapes(
    features: list[str],
    axis: str = "y",            # "y" for row bands, "x" for column bands
    dark: bool = True,
) -> list[dict]:
    """
    Build Plotly shape dicts for domain colour bands on an axis.

    Each domain gets a semi-transparent filled rectangle behind the tick labels.
    axis="y"  → horizontal bands (for row grouping)
    axis="x"  → vertical bands (for column grouping)
    """
    shapes = []
    if not features:
        return shapes

    current_domain = get_domain(features[0])
    band_start = 0

    def _add(start, end, domain):
        col   = DOMAIN_COLORS.get(domain, DOMAIN_COLORS["Other"])
        alpha = 0.12 if dark else 0.14
        # Convert #rrggbb to rgba() — Plotly doesn't accept 8-digit hex
        r = int(col[1:3], 16)
        g = int(col[3:5], 16)
        b = int(col[5:7], 16)
        fill = f"rgba({r},{g},{b},{alpha})"

        if axis == "y":
            shapes.append({
                "type": "rect",
                "xref": "paper", "yref": "y",
                "x0": 0, "x1": 1,
                "y0": start - 0.5, "y1": end - 0.5,
                "fillcolor": fill, "line": {"width": 0},
                "layer": "below",
            })
        else:
            shapes.append({
                "type": "rect",
                "xref": "x", "yref": "paper",
                "x0": start - 0.5, "x1": end - 0.5,
                "y0": 0, "y1": 1,
                "fillcolor": fill, "line": {"width": 0},
                "layer": "below",
            })

    for i, feat in enumerate(features):
        d = get_domain(feat)
        if d != current_domain:
            _add(band_start, i, current_domain)
            current_domain = d
            band_start = i
    _add(band_start, len(features), current_domain)

    return shapes


def domain_tick_colors(features: list[str]) -> list[str]:
    """Return a list of hex colours — one per feature — for axis tick colouring."""
    return [DOMAIN_COLORS.get(get_domain(f), DOMAIN_COLORS["Other"])
            for f in features]


def make_legend_traces() -> list[dict]:
    """Return invisible scatter traces that create a domain colour legend."""
    traces = []
    for domain in DOMAIN_ORDER:
        col = DOMAIN_COLORS[domain]
        traces.append({
            "type": "scatter",
            "x": [None], "y": [None],
            "mode": "markers",
            "marker": {"size": 10, "color": col, "symbol": "square"},
            "name": domain,
            "showlegend": True,
        })
    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Predictor domain → constituent feature columns
# ─────────────────────────────────────────────────────────────────────────────

PREDICTOR_DOMAINS: dict[str, list[str]] = {
    "Sensory": [
        # RBS-R sensory subscale
        "rbs_Sensory",
        # SP / AASP quadrants
        "sp_low_reg", "sp_sensitivity", "sp_avoiding", "sp_seeking",
        # SEQ-3 patterns (hyper + hypo + enhanced; seeking excluded)
        "seq_hyper", "seq_hypo", "seq_enhanced",
        # ISQ subscales
        "isq_noticing", "isq_interpreting", "isq_acting",
        # SCQ Sensory (sensory behaviors from ASD screening tool)
        "scq_Sensory",
    ],
    "Motor": [
        "dcdq_Fine Motor", "dcdq_Gross Motor", "dcdq_Coordination",
    ],
    "Social": [
        "scq_Social", "scq_Communication",
    ],
    "Repetitive/RRB": [
        "rbs_Obsessive", "rbs_Sameness", "rbs_Ritualistic",
        "rbs_Stereotyped", "rbs_SIB",
    ],
}

# Display order for predictor domains
PREDICTOR_DOMAIN_ORDER: list[str] = [
    "Sensory", "Motor", "Social", "Repetitive/RRB",
]


# ─────────────────────────────────────────────────────────────────────────────
# Composite computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_domain_composites(
    df: "pd.DataFrame",
    domains: list[str] | None = None,
) -> "pd.DataFrame":
    """
    Compute a composite score per predictor domain per patient.

    Method
    ------
    For each domain:
      1. Identify constituent features that exist in df.
      2. Z-score each feature (using the full-sample mean and SD).
      3. Take the row-wise mean of available z-scores (skipna=True),
         so patients with only partial instrument coverage still get a score.
      4. Patients with *no* data for any constituent feature receive NaN.

    Parameters
    ----------
    df      : merged DataFrame (patients × features)
    domains : subset of PREDICTOR_DOMAIN_ORDER to compute (default = all)

    Returns
    -------
    DataFrame with one column per domain, indexed like df.
    """
    import pandas as pd
    import numpy as np

    if domains is None:
        domains = PREDICTOR_DOMAIN_ORDER

    result = pd.DataFrame(index=df.index)

    for domain in domains:
        feat_cols = [f for f in PREDICTOR_DOMAINS.get(domain, [])
                     if f in df.columns]
        if not feat_cols:
            continue

        z_frame = pd.DataFrame(index=df.index)
        for col in feat_cols:
            vals  = df[col]
            mu    = float(vals.mean())
            sigma = float(vals.std(ddof=1))
            if sigma > 1e-9:
                z_frame[col] = (vals - mu) / sigma
            else:
                z_frame[col] = 0.0

        # Row-wise mean: patients with some data get a partial composite
        composite = z_frame.mean(axis=1, skipna=True)

        # NaN where ALL constituent features are missing
        all_missing = df[feat_cols].isna().all(axis=1)
        composite.loc[all_missing] = np.nan

        result[domain] = composite

    return result
