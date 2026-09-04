"""
layouts/cohort.py
─────────────────────────────────────────────────────────────────────────────
Replication Cohort tab (Option A) — single scrolling page.

Load an independent cohort (e.g. SSC) from its raw instrument + covariate
files, score it with the discovery definitions, and run the full √ΔR² analytic
suite ON that cohort:

  1. Hubness ranking (feature level) + ranked table
  2. PCA of the √ΔR² matrix (scree, PC1 outcome loadings, PC1 predictor scores)
  3. Suppressor / anxiety-adjustment (Δ√ΔR² heatmap + before/after scatter)
  4. Age-stratified hubs (grouped hubness bars + hub-rank heatmap)
  5. Domain-composite √ΔR² (+ its own PCA)

Because √ΔR² is scale-free, this within-cohort replication does not require the
cohort to share a measurement scale with discovery.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


COVARIATE_FIELDS_FOR_MAPPING = [
    ("sex",        "Sex"),
    ("age_months", "Age (months)"),
    ("nviq",       "Nonverbal IQ"),
]


def cohort_layout() -> html.Div:
    return html.Div([

        html.Div("Replication Cohort — Full √ΔR² Suite",
                 style={"fontSize": "15px", "fontWeight": "700",
                        "marginBottom": "2px"}),
        html.Div(
            "Load an independent cohort from its raw instrument and covariate "
            "files, scored with the same definitions as discovery, then run the "
            "full √ΔR² analytic suite on it: hubness, PCA, suppressor, "
            "age-stratified hubs, and domain composites. √ΔR² is scale-free, so "
            "replication does not require a shared measurement scale with "
            "discovery.",
            style={"fontSize": "11px", "color": "var(--text-muted)",
                   "marginBottom": "16px", "maxWidth": "900px"},
        ),

        dbc.Row([
            dbc.Col([
                html.Div("Step 1 — Upload cohort files", className="section-label"),
                html.Div(
                    "Drag in the cohort's raw instrument files (DCDQ, RBS-R, "
                    "SCQ, CBCL) plus one descriptive/covariate file with sex, "
                    "age, and nonverbal IQ.",
                    style={"fontSize": "10px", "color": "var(--text-muted)",
                           "marginBottom": "8px"},
                ),
                dcc.Upload(
                    id="upload-cohort",
                    children=html.Div([
                        html.Div("Cohort instrument + covariate files", style={
                            "fontSize": "11px", "fontWeight": "700",
                            "color": "var(--accent)"}),
                        html.Div("DCDQ · RBS-R · SCQ · CBCL · descriptive (.csv/.xlsx)",
                                 style={"fontSize": "10px",
                                        "color": "var(--text-muted)",
                                        "marginTop": "2px"}),
                    ]),
                    className="dash-upload", multiple=True, accept=".csv,.xlsx",
                    style={"marginBottom": "6px"},
                ),
                html.Div(id="status-cohort", className="status-muted",
                         style={"fontSize": "10px", "marginBottom": "10px",
                                "minHeight": "14px"}),
                html.Div(id="cohort-cov-mapping"),
            ], width=4),

            dbc.Col([
                html.Div("Step 2 — Covariates", className="section-label"),
                dbc.Checklist(
                    id="cohort-covariates",
                    options=[
                        {"label": " Age", "value": "age_months"},
                        {"label": " Sex", "value": "sex"},
                        {"label": " Nonverbal IQ", "value": "nviq"},
                    ],
                    value=["age_months", "sex", "nviq"],
                    inline=True,
                    style={"fontSize": "11px", "marginBottom": "12px"},
                ),
                html.Div("Suppressor (for anxiety-adjustment)",
                         className="section-label"),
                dbc.Select(
                    id="cohort-suppressor",
                    options=[{"label": "CBCL Anxious/Dep.",
                              "value": "cbcl_Anxious/Dep."}],
                    value="cbcl_Anxious/Dep.", size="sm",
                    style={"marginBottom": "12px", "maxWidth": "260px"},
                ),
                html.Div("FDR threshold", className="section-label"),
                dbc.Select(
                    id="cohort-fdr",
                    options=[
                        {"label": "q < .05", "value": "0.05"},
                        {"label": "q < .01", "value": "0.01"},
                        {"label": "q < .001", "value": "0.001"},
                    ],
                    value="0.05", size="sm",
                    style={"marginBottom": "12px", "maxWidth": "160px"},
                ),
            ], width=4),

            dbc.Col([
                html.Div("Step 3 — Run", className="section-label"),
                dbc.Button("Run full replication suite", id="cohort-run",
                           color="primary",
                           style={"width": "100%", "marginBottom": "6px"}),
                dbc.Button("Export all results (Excel)", id="cohort-export",
                           color="secondary", outline=True,
                           style={"width": "100%", "marginBottom": "8px"}),
                dcc.Download(id="cohort-download"),
                html.Div(
                    "Runs all five analyses on the loaded cohort. Export writes "
                    "every table (hubness, √ΔR² matrix, suppressor Δ, age bands, "
                    "domain composites) to one Excel workbook.",
                    style={"fontSize": "10px", "color": "var(--text-muted)"}),
            ], width=4),
        ], style={"marginBottom": "8px"}),

        html.Hr(style={"borderColor": "var(--border)"}),

        dcc.Loading(type="circle",
                    children=html.Div(id="cohort-results")),

    ], style={"padding": "16px"})


def covariate_mapping_panel(cov_report: dict, cov_columns: list,
                            unmatched_core: list) -> html.Div:
    """Covariate mapping UI shown after upload."""
    if not cov_columns:
        return html.Div(
            "No covariate/descriptive file detected yet. Upload one to enable "
            "covariate adjustment and age-stratified analysis.",
            style={"fontSize": "10px", "color": "var(--text-muted)",
                   "fontStyle": "italic", "marginBottom": "8px"})

    col_options = [{"label": "— none —", "value": "__none__"}] + [
        {"label": c, "value": c} for c in cov_columns]

    rows = []
    for field, label in COVARIATE_FIELDS_FOR_MAPPING:
        matched = cov_report.get(field)
        auto_ok = matched and matched != "__none__"
        rows.append(html.Div([
            html.Div([
                html.Span(label, style={"fontSize": "11px", "fontWeight": "600"}),
                html.Span("  ✓ auto" if auto_ok else "  ⚠ unmatched",
                          style={"fontSize": "10px",
                                 "color": ("var(--success)" if auto_ok
                                           else "var(--danger)"),
                                 "marginLeft": "4px"}),
            ]),
            dbc.Select(
                id={"type": "cohort-cov-map", "field": field},
                options=col_options,
                value=matched if auto_ok else "__none__",
                size="sm", style={"fontSize": "10px", "marginBottom": "6px"}),
        ]))

    header = (html.Div("⚠ Some covariates need mapping",
                       style={"fontSize": "10px", "color": "var(--danger)",
                              "marginBottom": "6px"})
              if unmatched_core else
              html.Div("✓ All core covariates auto-detected",
                       style={"fontSize": "10px", "color": "var(--success)",
                              "marginBottom": "6px"}))

    return html.Div([
        html.Div("Covariate mapping", className="section-label"),
        header, *rows,
    ])
