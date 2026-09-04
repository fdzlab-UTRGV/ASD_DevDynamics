"""
layouts/ridge.py
─────────────────────────────────────────────────────────────────────────────
Ridge Regression tab layout — Figure 3C (Fernandez et al.).

Out-of-sample generalization via ridge regression:
  - Model trained on SPARK discovery sample (all 10 behavioral predictors)
  - Applied to the held-out cohort (e.g. SSC), scored to match discovery
  - Reports Pearson r between predicted and observed CBCL / ADOS CSS scores
  - Permutation p-values (1,000 permutations)

The holdout instrument files are uploaded here (separate from the main SPARK data).
Expected columns: the same behavioral predictors used in the main analysis,
plus CBCL outcome columns.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def ridge_layout() -> html.Div:
    return html.Div([

        # ── Header ────────────────────────────────────────────────────────
        html.Div("Out-of-Sample Generalization — Ridge Regression",
                 style={"fontSize": "14px", "fontWeight": "700",
                        "marginBottom": "2px"}),
        html.Div(
            "A multivariate ridge model is trained on the SPARK discovery sample "
            "using all 10 behavioral predictors simultaneously (α selected by "
            "5-fold CV). The trained model is applied without retraining to the "
            "held-out cohort. Performance is reported as Pearson r with "
            "permutation p-values (1,000 permutations). The holdout's raw "
            "instrument files are scored automatically to match the discovery "
            "predictors and outcomes.",
            style={"fontSize": "11px", "color": "var(--text-muted)",
                   "marginBottom": "20px"},
        ),

        dbc.Row([

            # ── Left: holdout upload + controls ───────────────────────────
            dbc.Col([
                html.Div("Step 1 — Upload holdout dataset",
                         className="section-label"),
                html.Div(
                    "Upload the holdout cohort's raw instrument files (DCDQ, "
                    "RBS-R, SCQ, and CBCL 6-18 and/or 2-5). Drag all files in "
                    "at once. They are scored with the same definitions as the "
                    "discovery sample and kept separate from the main SPARK upload.",
                    style={"fontSize": "10px", "color": "var(--text-muted)",
                           "marginBottom": "8px"},
                ),
                dcc.Upload(
                    id="upload-ridge-holdout",
                    children=html.Div([
                        html.Div("Holdout instrument files", style={
                            "fontSize": "11px", "fontWeight": "700",
                            "color": "var(--accent)",
                        }),
                        html.Div("DCDQ · RBS-R · SCQ · CBCL (raw .csv/.xlsx)", style={
                            "fontSize": "10px", "color": "var(--text-muted)",
                            "marginTop": "2px",
                        }),
                    ]),
                    className="dash-upload",
                    multiple=True,
                    accept=".csv,.xlsx",
                    style={"marginBottom": "6px"},
                ),
                html.Div(id="status-ridge-holdout",
                         className="status-muted",
                         style={"fontSize": "10px", "marginBottom": "16px",
                                "minHeight": "14px"}),

                html.Hr(style={"borderColor": "var(--border)",
                               "margin": "8px 0 16px"}),

                html.Div("Step 2 — Run analysis", className="section-label"),
                dbc.Button("Run ridge regression", id="ridge-run",
                           color="primary", size="sm",
                           style={"width": "100%", "marginBottom": "6px"}),
                dbc.Button("Export results", id="ridge-export",
                           color="secondary", size="sm", outline=True,
                           style={"width": "100%"}),
                dcc.Download(id="ridge-download"),

                html.Hr(style={"borderColor": "var(--border)",
                               "margin": "16px 0 12px"}),

                html.Div("Options", className="section-label"),
                dbc.Label("Permutations", style={"fontSize": "11px"}),
                dbc.Select(
                    id="ridge-n-perm",
                    options=[
                        {"label": "100 (fast)", "value": "100"},
                        {"label": "1,000 (paper default)", "value": "1000"},
                        {"label": "5,000 (precise)", "value": "5000"},
                    ],
                    value="1000",
                    size="sm",
                    style={"marginBottom": "12px"},
                ),

                html.Div("Significance threshold (α)", style={
                    "fontSize": "11px", "color": "var(--text-muted)",
                    "marginBottom": "4px",
                }),
                dbc.Select(
                    id="ridge-alpha-thresh",
                    options=[
                        {"label": "p < .05", "value": "0.05"},
                        {"label": "p < .01", "value": "0.01"},
                        {"label": "p < .001", "value": "0.001"},
                    ],
                    value="0.05",
                    size="sm",
                    style={"marginBottom": "12px"},
                ),

            ], width=3, style={"borderRight": "1px solid var(--border)",
                               "paddingRight": "16px"}),

            # ── Right: results ─────────────────────────────────────────────
            dbc.Col([
                dcc.Loading(
                    type="circle",
                    children=html.Div(id="ridge-results"),
                ),
            ], width=9),

        ]),
    ], style={"padding": "16px"})
