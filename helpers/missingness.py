"""
helpers/missingness.py
─────────────────────────────────────────────────────────────────────────────
Shared missingness utilities used across PCA, MOFA, Mediation, and Sensory
regression tabs.

Public API:
    missingness_panel(merged, cols, title, view_level, imputed_n)
    drop_one_table(header_cols, primary_row, drop_rows)
"""

import pandas as pd
from dash import html
import dash_bootstrap_components as dbc


# ── Missingness panel ─────────────────────────────────────────────────────────

def missingness_panel(
    merged: pd.DataFrame | None,
    cols: list,
    title: str = "Missingness summary",
    view_level: bool = False,
    imputed_n: int | None = None,
) -> html.Div:
    """
    Universal missingness panel.

    Parameters
    ----------
    merged      : full merged DataFrame (pre complete-case filter)
    cols        : columns (or view keys when view_level=True) to summarize
    title       : section header text
    view_level  : if True, cols are view prefixes (e.g. "dcdq", "rbs") and
                  the panel summarizes available n per view rather than per column
    imputed_n   : if set, replaces "Imputed: N/A" with "Imputed: {n}" (Phase 9b)
    """
    if merged is None or not cols:
        return html.Div()

    n_total = len(merged)
    rows    = []

    if view_level:
        # Summarize at the view level — one row per instrument/view
        for view_key in cols:
            view_cols = [c for c in merged.columns
                         if c.startswith(f"{view_key}_")]
            if not view_cols:
                continue
            # A patient has data for this view if at least one domain is observed
            n_obs = int(merged[view_cols].notna().any(axis=1).sum())
            pct   = 100 * (1 - n_obs / n_total) if n_total else 0
            rows.append(html.Tr([
                html.Td(view_key.upper(),
                        style={"fontSize": "10px", "fontWeight": "600"}),
                html.Td(f"{n_obs:,}",
                        style={"fontSize": "10px"}),
                html.Td(f"{pct:.1f}%",
                        style={"fontSize": "10px",
                               "color": _miss_color(pct)}),
            ]))
        # Complete cases = all views have at least one observed domain
        all_view_cols = [c for vk in cols
                         for c in merged.columns
                         if c.startswith(f"{vk}_")]
        n_cc   = int(merged[all_view_cols].dropna(how="all").shape[0]) \
            if all_view_cols else 0
        pct_cc = 100 * n_cc / n_total if n_total else 0
        obs_cells = int(merged[all_view_cols].notna().sum().sum()) \
            if all_view_cols else 0
        total_cells = n_total * len(all_view_cols)
        pct_obs = 100 * obs_cells / total_cells if total_cells else 0

    else:
        # Column-level summary
        present_cols = [c for c in cols if c in merged.columns]
        for col in present_cols:
            n_obs = int(merged[col].notna().sum())
            pct   = 100 * (1 - n_obs / n_total) if n_total else 0
            rows.append(html.Tr([
                html.Td(col,
                        style={"fontSize": "10px"}),
                html.Td(f"{n_obs:,}",
                        style={"fontSize": "10px"}),
                html.Td(f"{pct:.1f}%",
                        style={"fontSize": "10px",
                               "color": _miss_color(pct)}),
            ]))
        n_cc   = int(merged[present_cols].dropna().shape[0]) \
            if present_cols else 0
        pct_cc = 100 * n_cc / n_total if n_total else 0
        obs_cells   = sum(int(merged[c].notna().sum())
                          for c in present_cols)
        total_cells = n_total * len(present_cols)
        pct_obs = 100 * obs_cells / total_cells if total_cells else 0

    imp_str = (f"Imputed: {imputed_n:,}" if imputed_n is not None
               else "Imputed: N/A")

    warn = []
    if pct_obs < 50:
        warn = [html.Div(
            f"⚠ Observed cell fraction is {pct_obs:.1f}% — "
            "interpret complete-case results with caution.",
            style={"fontSize": "9px", "color": "var(--danger)",
                   "marginTop": "4px"},
        )]

    return html.Div([
        html.Div(title,
                 style={"fontWeight": "700", "fontSize": "11px",
                        "color": "var(--text)", "marginBottom": "6px"}),
        html.Hr(style={"margin": "4px 0", "borderColor": "var(--border)"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th(h, style={
                    "fontSize": "9px", "color": "var(--text-muted)",
                    "padding": "2px 12px 2px 0", "fontWeight": "600",
                }) for h in ["Variable", "Available n", "Missing %"]
            ])),
            html.Tbody(rows),
        ], style={"width": "100%", "borderCollapse": "collapse"}),
        html.Hr(style={"margin": "6px 0", "borderColor": "var(--border)"}),
        html.Div(
            f"Complete cases: {n_cc:,} ({pct_cc:.1f}%)  |  "
            f"Observed cells: {pct_obs:.1f}%  |  {imp_str}",
            style={"fontSize": "10px", "color": "var(--text-muted)"},
        ),
        html.Div(
            "Analysis run on complete cases. "
            "Use drop-one sensitivity to see each variable's impact on n.",
            style={"fontSize": "9px", "color": "var(--text-muted)",
                   "marginTop": "4px"},
        ),
        *warn,
    ], style={
        "padding": "10px 14px",
        "border": "1px solid var(--border)",
        "borderRadius": "6px",
        "marginTop": "12px",
        "backgroundColor": "var(--bg-alt)",
    })


# ── Drop-one table ────────────────────────────────────────────────────────────

def drop_one_table(
    header_cols: list,
    primary_row: list,
    drop_rows: list,
    alert_text: str = (
        "Drop-one sensitivity: each row shows the result when that variable "
        "is excluded. Green n = sample size gain from relaxing overlap."
    ),
) -> html.Div:
    """
    Generic drop-one comparison table.

    Parameters
    ----------
    header_cols : list of column header strings
    primary_row : list of html.Td elements for the primary (no-drop) row
    drop_rows   : list of lists of html.Td elements, one per dropped variable
    alert_text  : explanatory note shown above the table
    """
    return html.Div([
        dbc.Alert(alert_text, color="info",
                  style={"fontSize": "11px"}),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th(h, style={"fontSize": "10px"})
                for h in header_cols
            ])),
            html.Tbody([
                html.Tr(primary_row),
                *[html.Tr(row) for row in drop_rows],
            ]),
        ], bordered=False, size="sm"),
    ])


# ── Internal ──────────────────────────────────────────────────────────────────

def _miss_color(pct: float) -> str:
    if pct > 50:
        return "var(--danger)"
    if pct > 20:
        return "var(--warning)"
    return "var(--text-muted)"
