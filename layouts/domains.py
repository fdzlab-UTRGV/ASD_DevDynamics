"""
layouts/domains.py
─────────────────────────────────────────────────────────────────────────────
Domain √ΔR² tab — cross-instrument domain-level analysis.

Identical pipeline to Part 2 but predictors are domain composites
(z-score mean across constituent features) rather than individual columns.

Domains
-------
Sensory       : SP + SEQ + ISQ + RBS-R Sensory + SCQ Sensory
Motor         : DCDQ all subscales
Social        : SCQ Social + SCQ Communication
Repetitive    : RBS-R Obsessive/Sameness/Ritualistic/Stereotyped/SIB
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from modules.domains import PREDICTOR_DOMAIN_ORDER, DOMAIN_COLORS


def domain_layout():
    domain_opts = [
        {"label": html.Span(
            d,
            style={"color": DOMAIN_COLORS.get(d, "#94a3b8"),
                   "fontWeight": "600"}
        ), "value": d}
        for d in PREDICTOR_DOMAIN_ORDER
    ]

    return html.Div([

        html.Div("Domain √ΔR² — Cross-instrument fingerprint by construct",
                 style={"fontSize": "14px", "fontWeight": "700",
                        "marginBottom": "2px"}),
        html.Div(
            "Computes a composite score per predictor domain (z-score mean across "
            "all constituent features from all instruments), then runs "
            "covariate-adjusted √ΔR² regressions. Because composites are on the "
            "same z-score scale, the resulting √ΔR² values are directly comparable "
            "across Sensory, Motor, Social, and Repetitive domains regardless of "
            "which instruments contributed.",
            style={"fontSize": "11px", "color": "var(--text-muted)",
                   "marginBottom": "16px"}),

        # ── Step 1: Domain √ΔR² ───────────────────────────────────────────
        html.Div("Step 1 — Domain composite √ΔR²",
                 style={"fontSize": "13px", "fontWeight": "700",
                        "marginBottom": "4px"}),

        dbc.Row([
            # Predictor domains
            dbc.Col([
                dbc.Label("Predictor domains", className="form-label"),
                dbc.Checklist(
                    id="dom-predictors",
                    options=domain_opts,
                    value=PREDICTOR_DOMAIN_ORDER,
                    inline=False,
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"fontSize": "12px", "display": "block",
                                "marginBottom": "4px"},
                ),
                html.Hr(style={"borderColor": "var(--border)",
                               "margin": "10px 0"}),
                html.Div("Composite features loaded per domain:",
                         style={"fontSize": "10px",
                                "color": "var(--text-muted)",
                                "marginBottom": "4px"}),
                html.Div(id="dom-feature-counts",
                         style={"fontSize": "10px",
                                "color": "var(--text-muted)"}),
            ], width=3),

            # Outcomes
            dbc.Col([
                dbc.Label("Outcomes", className="form-label"),
                dbc.Checklist(
                    id="dom-outcomes",
                    options=[],    # populated on tab activate
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
                    id="dom-covariates",
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
                    id="dom-fdr-thresh",
                    options=[
                        {"label": "q < .05", "value": 0.05},
                        {"label": "q < .10", "value": 0.10},
                        {"label": "q < .20", "value": 0.20},
                    ],
                    value=0.05, size="sm",
                    style={"maxWidth": "120px"},
                ),
            ], width=3),

            # Composite info
            dbc.Col([
                dbc.Label("Composite method", className="form-label"),
                html.Div([
                    html.Div("Z-score mean", style={"fontWeight": "700",
                                                    "fontSize": "11px"}),
                    html.Div(
                        "Each constituent feature is z-scored (full-sample "
                        "mean and SD). The domain composite = row-wise mean "
                        "of available z-scores. Patients with partial "
                        "instrument coverage still receive a composite.",
                        style={"fontSize": "10px",
                               "color": "var(--text-muted)",
                               "marginTop": "4px"}),
                ]),
            ], width=3),
        ], style={"marginBottom": "12px"}),

        dbc.Row([
            dbc.Col([
                dbc.Button("Run Domain Analysis", id="btn-dom-run",
                           color="primary", size="sm"),
                dbc.Button("Export CSV", id="btn-dom-export",
                           color="secondary", outline=True, size="sm",
                           style={"marginLeft": "8px"}, disabled=True),
            ]),
        ], style={"marginBottom": "12px"}),

        dcc.Download(id="dom-download"),

        dcc.Loading(type="circle",
                    children=html.Div(id="dom-content")),

        html.Hr(style={"borderColor": "var(--border)", "margin": "28px 0"}),

        # ── Step 2: PCA ───────────────────────────────────────────────────
        html.Div("Step 2 — PCA of domain √ΔR² matrix",
                 style={"fontSize": "13px", "fontWeight": "700",
                        "marginBottom": "4px"}),
        html.Div(
            "With only 4 predictor rows, the PCA is simple and interpretable: "
            "PC1 shows which domains co-vary in their association with outcomes. "
            "Run Step 1 first.",
            style={"fontSize": "11px", "color": "var(--text-muted)",
                   "marginBottom": "8px"}),
        dbc.Button("Run PCA", id="btn-dom-pca",
                   color="primary", size="sm", disabled=True),
        dbc.Button("⬇ Export CSV", id="btn-dom-pca-export",
                   color="secondary", outline=True, size="sm",
                   disabled=True, style={"marginLeft": "8px"}),
        dcc.Download(id="dom-pca-download"),
        dcc.Loading(type="circle",
                    children=html.Div(id="dom-pca-content",
                                      style={"marginTop": "12px"})),

        html.Hr(style={"borderColor": "var(--border)", "margin": "28px 0"}),

        # ── Step 3: Split-half ────────────────────────────────────────────
        html.Div("Step 3 — Split-half replication",
                 style={"fontSize": "13px", "fontWeight": "700",
                        "marginBottom": "4px"}),
        html.Div(
            "Demographically matched discovery / replication halves. "
            "Composites computed independently in each half before running √ΔR². "
            "Concordance r tests whether the domain-level fingerprint replicates.",
            style={"fontSize": "11px", "color": "var(--text-muted)",
                   "marginBottom": "8px"}),
        dbc.Row([
            dbc.Col([
                dbc.Label("Seed", className="form-label"),
                dbc.Input(id="dom-split-seed", type="number",
                          value=42, min=0, max=99999, size="sm",
                          style={"maxWidth": "100px"}),
            ], width=2),
        ], style={"marginBottom": "8px"}),
        dbc.Button("Run Split-Half", id="btn-dom-split",
                   color="primary", size="sm", disabled=True),
        dbc.Button("⬇ Export CSV", id="btn-dom-split-export",
                   color="secondary", outline=True, size="sm",
                   disabled=True, style={"marginLeft": "8px"}),
        dcc.Download(id="dom-split-download"),
        dcc.Loading(type="circle",
                    children=html.Div(id="dom-split-content",
                                      style={"marginTop": "12px"})),

    ], style={"padding": "16px"})
