from dash import html, dcc
import dash_bootstrap_components as dbc

def dev_age_panel():
    return html.Div([
        html.Div([
            html.Span("Developmental Prediction — Age-Band Analysis",
                      style={"fontWeight":"700","fontSize":"13px"}),
            html.Div("√ΔR² per domain × T1 age band → CBCL Psychopathology "
                     "(12–18y). ANOVA · Tukey HSD · Segmented Regression · "
                     "Bayesian Models (cell + individual level).",
                     style={"fontSize":"11px","color":"var(--text-muted)",
                            "marginTop":"2px","marginBottom":"12px"}),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Button("Run analysis", id="daa-run",
                           color="primary", size="sm",
                           style={"width":"100%","marginBottom":"6px"}),
                dbc.Button("Export results", id="daa-save",
                           color="secondary", size="sm", outline=True,
                           style={"width":"100%"}),
                dcc.Download(id="daa-download"),
                html.Hr(style={"margin":"12px 0","borderColor":"var(--border)"}),
                html.Div("Show sections", className="ctrl-label"),
                dbc.Checklist(
                    id="daa-sections",
                    options=[
                        {"label":"Cell √ΔR² table",       "value":"cells"},
                        {"label":"ANOVA",                  "value":"anova"},
                        {"label":"Tukey HSD",              "value":"tukey"},
                        {"label":"Segmented regression",   "value":"seg"},
                        {"label":"GLM (Type III SS)",      "value":"glm"},
                        {"label":"GLM Logistic",           "value":"glm_logistic"},
                        {"label":"Mixed Models",           "value":"mixed"},
                        {"label":"Bayesian (cell level)",  "value":"bayes_cell"},
                        {"label":"Bayesian (individual)",  "value":"bayes_ind"},
                    ],
                    value=["cells","anova","tukey","seg","glm","glm_logistic","mixed","bayes_cell","bayes_ind"],
                    inputStyle={"marginRight":"6px"},
                    labelStyle={"fontSize":"11px","display":"block",
                                "marginBottom":"3px"},
                ),
            ], width=3, style={"borderRight":"1px solid var(--border)",
                               "paddingRight":"16px"}),
            dbc.Col([
                dcc.Loading(type="circle",
                            children=html.Div(id="daa-results")),
            ], width=9),
        ]),
        dcc.Store(id="dev-age-results-store", storage_type="memory"),
    ])
