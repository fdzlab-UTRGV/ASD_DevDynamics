"""
layouts/sidebar.py
─────────────────────────────────────────────────────────────────────────────
Persistent sidebar — instrument upload slots and action buttons.

Upload SPARK instrument files here. The holdout cohort used for
out-of-sample ridge regression is uploaded on the Ridge Regression tab.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


# (key, label, accept_types, description)
INSTRUMENT_SLOTS = [
    ("dcdq", "DCDQ",       ".csv,.xlsx", "Developmental Coordination Disorder Q"),
    ("rbs",  "RBS-R",      ".csv,.xlsx", "Repetitive Behavior Scale – Revised"),
    ("scq",  "SCQ",        ".csv,.xlsx", "Social Communication Questionnaire"),
    ("ados", "ADOS",       ".csv,.xlsx", "Autism Diagnostic Observation Schedule"),
    ("cbcl", "CBCL",       ".csv,.xlsx", "Child Behavior Checklist"),
    ("cov",  "Covariates", ".csv,.xlsx", "core_descriptive_variables.csv"),
]


def _upload_slot(key: str, label: str, accept: str, info: str) -> html.Div:
    return html.Div([
        dcc.Upload(
            id=f"upload-{key}",
            children=html.Div([
                html.Div(label, style={
                    "fontSize": "11px", "fontWeight": "700",
                    "color": "var(--accent)",
                }),
                html.Div(info, style={
                    "fontSize": "10px", "color": "var(--text-muted)",
                    "marginTop": "2px",
                }),
            ]),
            className="dash-upload",
            multiple=True,
            accept=accept,
            style={"marginBottom": "4px"},
        ),
        html.Div(id=f"status-{key}",
                 className="status-muted",
                 style={"fontSize": "10px", "minHeight": "12px"}),
    ], style={"marginBottom": "10px"})


def sidebar_layout() -> html.Div:
    return html.Div([
        # Header
        html.Div([
            html.Div("ASD Phenotypic Architecture", style={
                "fontSize": "12px", "fontWeight": "700",
                "color": "var(--text)",
            }),
            html.Div("Fernandez et al.", style={
                "fontSize": "10px", "color": "var(--text-muted)",
            }),
        ], style={"marginBottom": "16px"}),

        # Instrument upload
        html.Div("Upload Instruments", className="section-label"),
        *[_upload_slot(*slot) for slot in INSTRUMENT_SLOTS],

        # Actions
        html.Div("Actions", className="section-label"),
        dbc.Button("Clear all", id="btn-clear",
                   color="danger", outline=True, size="sm",
                   className="w-100"),

        # Loaded data summary
        html.Div("Loaded data", className="section-label"),
        html.Div(id="load-summary",
                 style={"fontSize": "11px", "color": "var(--text-muted)"}),

    ], className="sidebar", style={
        "width": "220px",
        "padding": "16px",
        "height": "100vh",
        "overflowY": "auto",
        "flexShrink": "0",
    })
