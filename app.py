"""
app.py
─────────────────────────────────────────────────────────────────────────────
Developmental Dynamics of Phenotypic Architecture in ASD
Fernandez et al. — Cell Reports Medicine

This Dash application reproduces all analyses reported in the paper:

  Tab 1 — Correlations       → Figure 1: unadjusted Pearson r matrix
  Tab 2 — √ΔR² Analysis      → Figure 2: hubness, PCA; Figure 3A-B: split-half;
                                 Figure 5: anxiety-adjustment sensitivity
  Tab 3 — Age-Stratified Hubs → Figure 4: developmental hub reorganization
  Tab 4 — Dev. Coupling       → Figure 6: domain × age-band psychopathology coupling
  Tab 5 — Ridge Regression    → Figure 3C: out-of-sample generalization (RM0035)

Architecture:
  - Per-instrument data stores; single writer per store
  - get_merged_data() helper computes merged DataFrame on demand
  - All callbacks registered before app.run() via register_all()
  - Theme via CSS variables, no hardcoded colors

Data:
  Upload instrument files (DCDQ, RBS-R, SCQ, ADOS, CBCL, Covariates) via the
  sidebar. For Ridge Regression (Tab 5), upload the separate RM0035 sensory
  subsample using the upload control on that tab.
"""

from pathlib import Path

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from callbacks import register_all


# ─────────────────────────────────────────────────────────────────────────────
# Configuration

VERSION     = "v4.3.0"
APP_TITLE   = "ASD Phenotypic Architecture"
OUTPUT_DIR  = Path.home() / "Documents" / "asd_phenotypic_arch_outputs"


# ─────────────────────────────────────────────────────────────────────────────
# App initialization

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title=APP_TITLE,
)

server = app.server  # Flask server (exposed for any future endpoints)


# ─────────────────────────────────────────────────────────────────────────────
# Layout

def make_layout():
    """Build the top-level layout shell. Tab content rendered by callbacks."""
    from layouts.sidebar import sidebar_layout

    # ── Data stores — one per instrument ─────────────────────────────────────
    # Each store has exactly one writer (upload callback). Single-writer rule.
    source_stores = [
        dcc.Store(id="dcdq-store",             storage_type="memory"),
        dcc.Store(id="rbs-store",              storage_type="memory"),
        dcc.Store(id="scq-store",              storage_type="memory"),
        dcc.Store(id="ados-store",             storage_type="memory"),
        dcc.Store(id="cbcl-store",             storage_type="memory"),
        dcc.Store(id="cov-store",              storage_type="memory"),
        dcc.Store(id="sensory-store",          storage_type="memory"),
        dcc.Store(id="css-store",              storage_type="memory"),
        # Analysis result caches
        dcc.Store(id="corr-results-store",     storage_type="memory"),
        dcc.Store(id="p2-results-store",       storage_type="memory"),
        dcc.Store(id="p2-split-store",         storage_type="memory"),
        dcc.Store(id="p2-suppress-store",      storage_type="memory"),
        dcc.Store(id="p2-pca-store",           storage_type="memory"),
        dcc.Store(id="p2-hub-store",           storage_type="memory"),
        dcc.Store(id="dom-results-store",      storage_type="memory"),
        dcc.Store(id="dom-pca-store",          storage_type="memory"),
        dcc.Store(id="dom-split-store",        storage_type="memory"),
        dcc.Store(id="dev-age-results-store",  storage_type="memory"),
        # Ridge regression — holds RM0035 dataset (separate upload)
        dcc.Store(id="ridge-holdout-store",    storage_type="memory"),
        dcc.Store(id="ridge-results-store",    storage_type="memory"),
        # Replication cohort — holds a second scored cohort + its covariate cols
        dcc.Store(id="cohort-store",           storage_type="memory"),
        dcc.Store(id="cohort-cov-cols-store",  storage_type="memory"),
        dcc.Store(id="cohort-export-store",    storage_type="memory"),
    ]

    # Theme store — persists across sessions
    theme_store = dcc.Store(id="theme-store", storage_type="local", data="dark")

    # ── Header ────────────────────────────────────────────────────────────────
    header = html.Div([
        dbc.Tabs(
            id="main-tabs",
            active_tab="tab-corr",
            children=[
                dbc.Tab(label="Correlations",        tab_id="tab-corr"),
                dbc.Tab(label="√ΔR² Analysis",       tab_id="tab-p2"),
                dbc.Tab(label="Age-Stratified Hubs", tab_id="tab-domains"),
                dbc.Tab(label="Dev. Coupling",        tab_id="tab-devpred"),
                dbc.Tab(label="Ridge Regression",     tab_id="tab-ridge"),
                dbc.Tab(label="Replication Cohort",   tab_id="tab-cohort"),
            ],
            style={"flex": "1"},
        ),
        dbc.Button("☀ Light", id="btn-theme",
                   size="sm", color="secondary", outline=True,
                   style={"marginLeft": "12px"}),
        html.Span(VERSION, className="version-badge",
                  style={"marginLeft": "8px"}),
    ], className="header-bar")

    # ── Body ─────────────────────────────────────────────────────────────────
    body = html.Div([
        sidebar_layout(),
        html.Div(id="tab-content",
                 style={"flex": "1", "overflowY": "auto",
                        "padding": "0 16px"}),
    ], style={"display": "flex", "flex": "1", "overflow": "hidden"})

    return html.Div([
        *source_stores,
        theme_store,
        header,
        body,
    ], className="app-wrapper",
       style={"display": "flex", "flexDirection": "column",
              "height": "100vh", "overflow": "hidden"})


app.layout = make_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Register all callbacks

register_all(app)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point

if __name__ == "__main__":
    app.run(debug=True, port=8050)
