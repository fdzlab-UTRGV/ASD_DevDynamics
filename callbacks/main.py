"""
callbacks/main.py
─────────────────────────────────────────────────────────────────────────────
Tab content router. Renders the correct layout based on the active tab.

Tab → Figure mapping (Fernandez et al.):
  tab-corr     → Figure 1:   unadjusted Pearson r matrix + CBCL vs. ADOS distributions
  tab-p2       → Figure 2:   √ΔR² hubness + PCA
                  Figure 3A-B: split-half reproducibility + bootstrap stability
                  Figure 5:   anxiety-adjustment sensitivity analysis
  tab-domains  → Figure 4:   age-stratified hub reorganization
  tab-devpred  → Figure 6:   developmental domain–psychopathology coupling
  tab-ridge    → Figure 3C:  out-of-sample ridge regression (holdout upload)
"""

from dash import Input, Output, html

from layouts.correlations import corr_layout
from layouts.part2 import part2_layout
from layouts.domains import domain_layout
from layouts.dev_age_analysis_panel import dev_age_panel
from layouts.ridge import ridge_layout
from layouts.cohort import cohort_layout


def register(app):

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs",    "active_tab"),
    )
    def render_tab(active_tab):
        layouts = {
            "tab-corr":    corr_layout,
            "tab-p2":      part2_layout,
            "tab-domains": domain_layout,
            "tab-devpred": dev_age_panel,
            "tab-ridge":   ridge_layout,
            "tab-cohort":  cohort_layout,
        }
        layout_fn = layouts.get(active_tab)
        if layout_fn:
            return layout_fn()
        return html.Div("Select a tab.",
                        style={"padding": "16px",
                               "color": "var(--text-muted)"})
