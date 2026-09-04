"""
layouts/part2.py
─────────────────────────────────────────────────────────────────────────────
Part 2 tab — Formal analysis following Macedo et al. approach.

Step 1: Mass-univariate √ΔR² fingerprint matrix
  Each input domain → each outcome, covariate-adjusted.
  Effect size = signed semi-partial correlation (√ΔR²).

Step 2: PCA of the effect-size matrix  [planned]
Step 3: Split-half ridge regression    [planned]
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def part2_layout():
    return html.Div([

        # ── Header ────────────────────────────────────────────────────────
        html.Div("Part 2 — Formal Analysis", style={
            "fontSize": "14px", "fontWeight": "700", "marginBottom": "2px"}),
        html.Div(
            "Mass-univariate linear regressions with signed √ΔR² effect sizes "
            "(covariate-adjusted). FDR-corrected across all predictor × outcome pairs.",
            style={"fontSize": "11px", "color": "var(--text-muted)",
                   "marginBottom": "16px"}),

        # ── Step 1: Mass-univariate √ΔR² ─────────────────────────────────
        html.Div("Step 1 — √ΔR² Fingerprint Matrix",
                 style={"fontSize": "13px", "fontWeight": "700",
                        "marginBottom": "4px"}),
        html.Div(
            "For each predictor × outcome pair: fit outcome ~ predictor + covariates, "
            "compare to base model without predictor. "
            "√ΔR² = sign(β) × √(unique variance explained). "
            "Comparable across all instruments and outcomes.",
            style={"fontSize": "11px", "color": "var(--text-muted)",
                   "marginBottom": "12px"}),

        dbc.Row([
            # Predictors
            dbc.Col([
                dbc.Label("Predictors (inputs)", className="form-label"),
                dbc.Checklist(
                    id="p2-predictors",
                    options=[],   # populated on tab activate
                    value=[],
                    inline=False,
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"fontSize": "11px", "display": "block",
                                "marginBottom": "2px"},
                ),
            ], width=4),

            # Outcomes
            dbc.Col([
                dbc.Label("Outcomes", className="form-label"),
                dbc.Checklist(
                    id="p2-outcomes",
                    options=[],   # populated on tab activate
                    value=[],
                    inline=False,
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"fontSize": "11px", "display": "block",
                                "marginBottom": "2px"},
                ),
            ], width=3),

            # Covariates + options
            dbc.Col([
                dbc.Label("Covariates", className="form-label"),
                dbc.Checklist(
                    id="p2-covariates",
                    options=[
                        {"label": " Age",  "value": "age_months"},
                        {"label": " Sex",  "value": "sex"},
                        {"label": " NVIQ", "value": "nviq"},
                    ],
                    value=["age_months", "sex", "nviq"],
                    inline=False,
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"fontSize": "11px", "display": "block",
                                "marginBottom": "2px"},
                ),
                html.Hr(style={"borderColor": "var(--border)",
                               "margin": "10px 0"}),
                dbc.Label("FDR threshold", className="form-label"),
                dbc.Select(
                    id="p2-fdr-thresh",
                    options=[
                        {"label": "q < .05", "value": 0.05},
                        {"label": "q < .10", "value": 0.10},
                        {"label": "q < .20", "value": 0.20},
                    ],
                    value=0.05, size="sm",
                    style={"maxWidth": "120px"},
                ),
                html.Hr(style={"borderColor": "var(--border)",
                               "margin": "10px 0"}),
                dbc.Label("Effect size display", className="form-label"),
                dbc.RadioItems(
                    id="p2-display-mode",
                    options=[
                        {"label": " √ΔR² (semi-partial r)", "value": "sqrt_dr2"},
                        {"label": " Raw β coefficient",     "value": "beta"},
                        {"label": " R² (full model)",       "value": "r2_full"},
                    ],
                    value="sqrt_dr2",
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"fontSize": "11px"},
                ),
            ], width=3),
        ], style={"marginBottom": "12px"}),

        dbc.Row([
            dbc.Col([
                dbc.Button("Run √ΔR² Analysis", id="btn-p2-run",
                           color="primary", size="sm"),
                dbc.Button("Export CSV", id="btn-p2-export",
                           color="secondary", outline=True, size="sm",
                           style={"marginLeft": "8px"}, disabled=True),
            ], width=6),
            dbc.Col([
                dbc.Checklist(
                    id="p2-group-domain",
                    options=[{"label": " Group by domain (cross-instrument)",
                              "value": "group"}],
                    value=["group"],
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"fontSize": "11px",
                                "color": "var(--accent)",
                                "fontWeight": "600"},
                ),
            ], width=6),
        ], style={"marginBottom": "12px"}),
        dcc.Download(id="p2-download"),

        dcc.Loading(type="circle",
                    children=html.Div(id="p2-content")),


        html.Hr(style={"borderColor": "var(--border)", "margin": "28px 0"}),

        # ── Step 2: PCA of effect-size matrix (placeholder) ───────────────
        html.Div("Step 2 — PCA of Effect-Size Matrix",
                 style={"fontSize": "13px", "fontWeight": "700",
                        "marginBottom": "4px"}),
        html.Div(
            "Run Step 1 first. PCA applied to the √ΔR² matrix identifies "
            "the dominant axis of sensorimotor-to-outcome co-variation.",
            style={"fontSize": "11px", "color": "var(--text-muted)",
                   "marginBottom": "8px"}),

        dbc.Button("Run PCA on √ΔR² matrix", id="btn-p2-pca",
                   color="primary", size="sm", disabled=True),
        dbc.Button("⬇ Export CSV", id="btn-p2-pca-export",
                   color="secondary", outline=True, size="sm",
                   disabled=True, style={"marginLeft": "8px"}),
        dcc.Download(id="p2-pca-download"),

        dcc.Loading(type="circle",
                    children=html.Div(id="p2-pca-content",
                                      style={"marginTop": "12px"})),

        html.Hr(style={"borderColor": "var(--border)", "margin": "28px 0"}),

        # ── Step 3: Split-half ridge regression (placeholder) ─────────────
        html.Div("Step 3 — Split-Half Ridge Regression",
                 style={"fontSize": "13px", "fontWeight": "700",
                        "marginBottom": "4px"}),
        html.Div(
            "Demographically matched discovery / replication halves (stratified "
            "by age quartile × sex). Concordance r between discovery and replication "
            "√ΔR² vectors is the headline replication statistic. Ridge regression "
            "trained on discovery, tested on replication (α chosen by 5-fold CV).",
            style={"fontSize": "11px", "color": "var(--text-muted)",
                   "marginBottom": "8px"}),

        dbc.Row([
            dbc.Col([
                dbc.Label("Random seed", className="form-label"),
                dbc.Input(id="p2-split-seed", type="number",
                          value=42, min=0, max=99999, size="sm",
                          style={"maxWidth": "100px"}),
            ], width=3),
        ], style={"marginBottom": "8px"}),

        dbc.Button("Run Split-Half", id="btn-p2-split",
                   color="primary", size="sm", disabled=True),
        dbc.Button("⬇ Export CSV", id="btn-p2-split-export",
                   color="secondary", outline=True, size="sm",
                   disabled=True, style={"marginLeft": "8px"}),
        dcc.Download(id="p2-split-download"),

        dcc.Loading(type="circle",
                    children=html.Div(id="p2-split-content",
                                      style={"marginTop": "12px"})),

        html.Hr(style={"borderColor": "var(--border)", "margin": "28px 0"}),

        # ── Step 4: Suppressor with √ΔR² ─────────────────────────────────
        html.Div("Step 4 — Suppressor Analysis (√ΔR²)",
                 style={"fontSize": "13px", "fontWeight": "700",
                        "marginBottom": "4px"}),
        html.Div(
            "Adds a suppressor variable as an additional covariate and re-runs "
            "the mass-univariate models. Shows how much of each predictor's unique "
            "variance in each outcome is concealed (or revealed) by the suppressor. "
            "Negative Δ√ΔR² = suppressor masks the association; "
            "positive = suppressor reveals it.",
            style={"fontSize": "11px", "color": "var(--text-muted)",
                   "marginBottom": "10px"}),

        dbc.Row([
            dbc.Col([
                dbc.Label("Suppressor variable", className="form-label"),
                dbc.Select(
                    id="p2-suppressor-var",
                    options=[],   # populated on tab activate
                    value=None,
                    size="sm",
                ),
            ], width=4),
        ], style={"marginBottom": "10px"}),

        dbc.Button("Run Suppressor Analysis", id="btn-p2-suppress",
                   color="primary", size="sm", disabled=True),
        dbc.Button("⬇ Export CSV", id="btn-p2-suppress-export",
                   color="secondary", outline=True, size="sm",
                   disabled=True, style={"marginLeft": "8px"}),
        dcc.Download(id="p2-suppress-download"),

        dcc.Loading(type="circle",
                    children=html.Div(id="p2-suppress-content",
                                      style={"marginTop": "12px"})),

        html.Hr(style={"borderColor": "var(--border)", "margin": "28px 0"}),

        # ── Step 5: Hubness Index ─────────────────────────────────────────
        html.Div("Step 5 — Hubness Index",
                 style={"fontSize": "13px", "fontWeight": "700",
                        "marginBottom": "4px"}),
        html.Div(
            "Quantifies which predictors occupy hub-like positions in ASD "
            "behavioral organization. Hubness = \u03a3|\u221a\u0394R\u00b2| across FDR-significant "
            "outcomes. Integrates both the magnitude and breadth of each "
            "predictor's unique cross-domain associations. Requires Step 1.",
            style={"fontSize": "11px", "color": "var(--text-muted)",
                   "marginBottom": "10px"}),

        dbc.Row([
            dbc.Col([
                dbc.Label("FDR threshold", className="form-label"),
                dbc.Select(
                    id="p2-hub-fdr-thresh",
                    options=[
                        {"label": "q < .05", "value": 0.05},
                        {"label": "q < .10", "value": 0.10},
                        {"label": "q < .20", "value": 0.20},
                    ],
                    value=0.05, size="sm",
                    style={"maxWidth": "120px"},
                ),
            ], width=2),
            dbc.Col([
                dbc.Label("Include", className="form-label"),
                dbc.RadioItems(
                    id="p2-hub-sig-only",
                    options=[
                        {"label": " Significant outcomes only", "value": "sig"},
                        {"label": " All outcomes (sensitivity check)", "value": "all"},
                    ],
                    value="sig",
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"fontSize": "11px", "display": "block",
                                "marginBottom": "2px"},
                ),
            ], width=4),
        ], style={"marginBottom": "10px"}),

        dbc.Button("Compute Hubness Index", id="btn-p2-hub",
                   color="primary", size="sm", disabled=True),
        dbc.Button("\u2b07 Export CSV", id="btn-p2-hub-export",
                   color="secondary", outline=True, size="sm",
                   disabled=True, style={"marginLeft": "8px"}),
        dcc.Download(id="p2-hub-download"),

        dcc.Loading(type="circle",
                    children=html.Div(id="p2-hub-content",
                                      style={"marginTop": "12px"})),

        html.Hr(style={"borderColor": "var(--border)", "margin": "28px 0"}),

        # ── Step 6: Age-stratified hub stability ──────────────────────────
        html.Div("Step 6 — Age-Stratified Hub Stability",
                 style={"fontSize": "13px", "fontWeight": "700",
                        "marginBottom": "4px"}),
        html.Div(
            "Tests whether hub rankings are stable across development. "
            "Splits participants into age bands, recomputes hubness within "
            "each band, and reports Spearman rank concordance. "
            "Participants without age data are excluded and reported. "
            "Requires Step 1 and a loaded covariates file.",
            style={"fontSize": "11px", "color": "var(--text-muted)",
                   "marginBottom": "10px"}),

        dbc.Row([
            dbc.Col([
                dbc.Label("Age bands", className="form-label"),
                html.Div([
                    html.Span("4–8 yrs  ·  9–12 yrs  ·  13–17 yrs  ·  18+ yrs",
                              style={"fontSize": "11px",
                                     "color": "var(--text-muted)"}),
                    html.Div("Minimum 50 participants per band required.",
                             style={"fontSize": "10px",
                                    "color": "var(--text-muted)",
                                    "marginTop": "2px"}),
                ]),
            ], width=4),
            dbc.Col([
                dbc.Label("FDR threshold", className="form-label"),
                dbc.Select(
                    id="p2-agehub-fdr-thresh",
                    options=[
                        {"label": "q < .05", "value": 0.05},
                        {"label": "q < .10", "value": 0.10},
                        {"label": "q < .20", "value": 0.20},
                    ],
                    value=0.05, size="sm",
                    style={"maxWidth": "120px"},
                ),
            ], width=2),
        ], style={"marginBottom": "10px"}),

        dbc.Button("Run Age-Stratified Analysis", id="btn-p2-agehub",
                   color="primary", size="sm", disabled=True),
        dbc.Button("\u2b07 Export CSV", id="btn-p2-agehub-export",
                   color="secondary", outline=True, size="sm",
                   disabled=True, style={"marginLeft": "8px"}),
        dcc.Download(id="p2-agehub-download"),

        dcc.Loading(type="circle",
                    children=html.Div(id="p2-agehub-content",
                                      style={"marginTop": "12px"})),

        html.Hr(style={"borderColor": "var(--border)", "margin": "28px 0"}),

        # ── Step 7: Equal-N sensitivity ───────────────────────────────────
        html.Div("Step 7 — Equal-N Sensitivity Check",
                 style={"fontSize": "13px", "fontWeight": "700",
                        "marginBottom": "4px"}),
        html.Div(
            "Subsamples each age band to the same N as the smallest band, "
            "then re-runs the hub ranking analysis. Tests whether weaker "
            "concordance in later bands reflects smaller samples or genuine "
            "developmental shifts. Requires Step 6.",
            style={"fontSize": "11px", "color": "var(--text-muted)",
                   "marginBottom": "10px"}),

        dbc.Row([
            dbc.Col([
                dbc.Label("Iterations", className="form-label"),
                dbc.Input(id="p2-equaln-iters", type="number",
                          value=5, min=1, max=20, size="sm",
                          style={"maxWidth": "80px"}),
                html.Div("More iterations = more stable estimates. "
                         "5 is fast; 20 for publication.",
                         style={"fontSize": "10px",
                                "color": "var(--text-muted)",
                                "marginTop": "3px"}),
            ], width=3),
            dbc.Col([
                dbc.Label("Random seed", className="form-label"),
                dbc.Input(id="p2-equaln-seed", type="number",
                          value=42, min=0, max=99999, size="sm",
                          style={"maxWidth": "100px"}),
            ], width=2),
        ], style={"marginBottom": "10px"}),

        dbc.Button("Run Equal-N Sensitivity", id="btn-p2-equaln",
                   color="primary", size="sm", disabled=True),
        dbc.Button("\u2b07 Export CSV", id="btn-p2-equaln-export",
                   color="secondary", outline=True, size="sm",
                   disabled=True, style={"marginLeft": "8px"}),
        dcc.Download(id="p2-equaln-download"),

        dcc.Loading(type="circle",
                    children=html.Div(id="p2-equaln-content",
                                      style={"marginTop": "12px"})),

        html.Hr(style={"borderColor": "var(--border)", "margin": "28px 0"}),

        # ── Network Analysis Export ───────────────────────────────────────
        html.Div("Network Analysis — Data Export",
                 style={"fontSize": "13px", "fontWeight": "700",
                        "marginBottom": "4px"}),
        html.Div([
            html.Span(
                "Exports data for the supplementary partial correlation network "
                "analysis (R: qgraph + networktools + bootnet). Produces two files: ",
            ),
            html.Span("(1) ", style={"fontWeight": "600"}),
            html.Span("raw_data.csv — participant-level predictor + outcome scores "
                      "for network estimation, "),
            html.Span("(2) ", style={"fontWeight": "600"}),
            html.Span("hubness_index.csv — predictor hub rankings for convergence test. "
                      "Run Step 1 and Step 5 first."),
        ], style={"fontSize": "11px", "color": "var(--text-muted)",
                  "marginBottom": "10px"}),

        dbc.Button("\u2b07 Export for Network Analysis (R)",
                   id="btn-p2-network-export",
                   color="success", outline=True, size="sm", disabled=True),
        dcc.Download(id="p2-network-download"),

    ], style={"padding": "16px"})
