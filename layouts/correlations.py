"""
layouts/correlations.py
─────────────────────────────────────────────────────────────────────────────
Correlations tab — pairwise Pearson r between predictor and outcome domains.
FDR-corrected. Optional RM0035 sensory-only filter.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def corr_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Predictor scales", className="form-label"),
                dbc.Checklist(id="corr-predictors", inline=True,
                              inputStyle={"marginRight": "4px"},
                              labelStyle={"marginRight": "12px",
                                          "fontSize": "11px"}),
            ], width=6),
            dbc.Col([
                dbc.Label("Outcome scales", className="form-label"),
                dbc.Checklist(id="corr-outcomes", inline=True,
                              inputStyle={"marginRight": "4px"},
                              labelStyle={"marginRight": "12px",
                                          "fontSize": "11px"}),
            ], width=6),
        ], style={"marginBottom": "12px"}),
        dbc.Row([
            dbc.Col([
                dbc.Button("Run Correlations", id="btn-corr",
                           color="primary", size="sm"),
                dbc.Button("Save run", id="btn-corr-save",
                           color="success", size="sm",
                           style={"marginLeft": "8px"}, disabled=True),
                dbc.Button("Export CSV", id="btn-corr-export",
                           color="secondary", outline=True, size="sm",
                           style={"marginLeft": "4px"}, disabled=True),
                dcc.Download(id="download-corr"),
            ], width=8),
            dbc.Col([
                dbc.Checklist(
                    id="corr-rm0035",
                    options=[{"label": " RM0035 sensory only", "value": "rm0035"}],
                    value=[], inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"fontSize": "11px"},
                ),
                html.Div(id="corr-rm0035-note",
                         className="status-muted",
                         style={"fontSize": "10px", "marginTop": "2px"}),
            ], width=4),
        ], style={"marginBottom": "16px"}),
        html.Div(id="corr-save-status",
                 style={"fontSize": "11px", "color": "var(--success)",
                        "marginBottom": "8px", "minHeight": "16px"}),
        dcc.Loading(type="circle", children=html.Div(id="corr-content")),
        # Per-tab results store - holds last run for export/save without polluting source data
    ], style={"padding": "16px"})
