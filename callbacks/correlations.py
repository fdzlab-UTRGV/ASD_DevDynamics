"""
callbacks/correlations.py
─────────────────────────────────────────────────────────────────────────────
Correlations tab callbacks.

Pattern:
- populate_corr_checks: on tab activation, populate predictor/outcome checklists
  from columns actually present in merged data
- run_corr: compute matrix, build heatmap + table, write result to corr-results-store
- export_corr: download CSV from corr-results-store
- save_corr: write Save Run bundle to disk
"""

import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc

from helpers.store import get_merged_data
from helpers.theme import get_plotly_layout, get_axis_style
from helpers.output import save_run
from modules.stats import pearson_pairwise_matrix
from modules import schema as S


def register(app):

    # ── Populate checklists when entering the tab ────────────────────────────
    @app.callback(
        Output("corr-predictors", "options"),
        Output("corr-outcomes",   "options"),
        Output("corr-predictors", "value"),
        Output("corr-outcomes",   "value"),
        Input("main-tabs",       "active_tab"),
        State("dcdq-store",      "data"),
        State("rbs-store",       "data"),
        State("scq-store",       "data"),
        State("ados-store",      "data"),
        State("cbcl-store",      "data"),
        State("sensory-store",   "data"),
    )
    def populate_corr_checks(tab, dcdq, rbs, scq, ados, cbcl, sensory):
        if tab != "tab-corr":
            return [], [], [], []
        merged = get_merged_data(dcdq=dcdq, rbs=rbs, scq=scq, ados=ados,
                                  cbcl=cbcl, sensory=sensory)
        if merged is None:
            return [], [], [], []

        all_cols = set(c for c in merged.columns if not c.startswith("_"))

        def make_opts(pairs):
            return [{"label": f"{s.upper()} - {d}", "value": f"{s}_{d}"}
                    for s, d in pairs if f"{s}_{d}" in all_cols]

        pred_opts = make_opts(S.CORR_PREDICTORS)
        out_opts  = make_opts(S.CORR_OUTCOMES)
        return (pred_opts, out_opts,
                [o["value"] for o in pred_opts],
                [o["value"] for o in out_opts])

    # ── Run correlations ─────────────────────────────────────────────────────
    @app.callback(
        Output("corr-content",        "children"),
        Output("corr-results-store",  "data"),
        Output("btn-corr-export",     "disabled"),
        Output("btn-corr-save",       "disabled"),
        Output("corr-rm0035-note",    "children"),
        Input("btn-corr",             "n_clicks"),
        State("corr-predictors",      "value"),
        State("corr-outcomes",        "value"),
        State("corr-rm0035",          "value"),
        State("dcdq-store",           "data"),
        State("rbs-store",            "data"),
        State("scq-store",            "data"),
        State("ados-store",           "data"),
        State("cbcl-store",           "data"),
        State("cov-store",            "data"),
        State("sensory-store",        "data"),
        State("css-store",            "data"),
        State("theme-store",          "data"),
        prevent_initial_call=True,
    )
    def run_corr(_, predictors, outcomes, rm0035,
                 dcdq, rbs, scq, ados, cbcl, cov, sensory, css, theme_mode):

        if not predictors or not outcomes:
            return (dbc.Alert("Select predictors and outcomes.", color="warning"),
                    None, True, True, "")

        merged = get_merged_data(dcdq=dcdq, rbs=rbs, scq=scq, ados=ados,
                                  cbcl=cbcl, cov=cov, sensory=sensory, css=css)
        if merged is None:
            return (dbc.Alert("No data loaded.", color="warning"),
                    None, True, True, "")

        # RM0035 filter
        rm0035_note = ""
        if rm0035 and "rm0035" in rm0035:
            sensory_cols = [c for c in merged.columns
                            if c.startswith(("sp_", "seq_", "isq_"))]
            if sensory_cols:
                mask = merged[sensory_cols].notna().any(axis=1)
                merged = merged[mask]
                rm0035_note = (f"RM0035 mode: {len(merged):,} patients with sensory data")
            else:
                rm0035_note = "⚠ No sensory data loaded — running on all patients"

        try:
            result = pearson_pairwise_matrix(merged, predictors, outcomes)
        except Exception as e:
            return (dbc.Alert(f"Correlation error: {e}", color="danger"),
                    None, True, True, rm0035_note)

        if result is None or result.empty:
            return (dbc.Alert("No overlapping data found.", color="warning"),
                    None, True, True, rm0035_note)

        # ── Build pivot tables (pred × out) ──────────────────────────────
        # Only pairs where pred != out appear in result; self-pairs are NaN
        pivot_r = result.pivot(index="predictor", columns="outcome", values="r")
        pivot_p = result.pivot(index="predictor", columns="outcome", values="p_fdr")

        # Preserve user-selected ordering; only keep cols/rows present in data
        row_order = [p for p in predictors if p in pivot_r.index]
        col_order = [o for o in outcomes  if o in pivot_r.columns]
        pivot_r = pivot_r.reindex(index=row_order, columns=col_order)
        pivot_p = pivot_p.reindex(index=row_order, columns=col_order)

        def _stars(q):
            if pd.isna(q): return ""
            if q < 0.001:  return "***"
            if q < 0.01:   return "**"
            if q < 0.05:   return "*"
            return ""

        # Annotation: "0.32**" for real pairs, "—" for self-correlations,
        # "n/a" for structural gaps (e.g. missing data)
        ann_text, z_vals = [], []
        for pred in row_order:
            ann_row, z_row = [], []
            for out in col_order:
                if pred == out:
                    ann_row.append("—")
                    z_row.append(None)          # no colour for self-cells
                else:
                    r_val = pivot_r.loc[pred, out]
                    p_val = pivot_p.loc[pred, out]
                    if pd.isna(r_val):
                        ann_row.append("n/a")
                        z_row.append(None)
                    else:
                        ann_row.append(f"{r_val:.2f}{_stars(p_val)}")
                        z_row.append(float(r_val))
            ann_text.append(ann_row)
            z_vals.append(z_row)

        dark = (theme_mode or "dark") == "dark"
        mid  = "#0d0f14" if dark else "#f1f5f9"

        fig = go.Figure(go.Heatmap(
            z=z_vals,
            x=col_order,
            y=row_order,
            colorscale=[[0, "#34d399"], [0.5, mid], [1, "#f87171"]],
            zmin=-0.7, zmax=0.7,
            text=ann_text,
            texttemplate="%{text}",
            textfont={"size": 10},
            showscale=True,
            colorbar={"title": "r", "thickness": 12,
                      "tickfont": {"size": 9}},
            hoverongaps=False,
            hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
        ))
        axis_style = get_axis_style(theme_mode or "dark")
        fig.update_layout(
            height=max(320, len(row_order) * 46 + 140),
            margin={"l": 160, "r": 60, "t": 30, "b": 130},
            xaxis={**axis_style, "tickangle": -45, "tickfont": {"size": 10}},
            yaxis={**axis_style, "tickfont": {"size": 10}},
            **get_plotly_layout(theme_mode or "dark"),
        )

        # FDR legend note
        fdr_note = html.Div(
            "* FDR q < .05   ** q < .01   *** q < .001  │  "
            "Benjamini–Hochberg correction across all displayed pairs. "
            "Self-correlations (—) excluded by design.",
            style={"fontSize": "10px", "color": "var(--text-muted)",
                   "marginTop": "6px"},
        )

        # Sorted table by |r|
        result_sorted = result.sort_values("r", key=abs, ascending=False)
        rows = []
        for _, row in result_sorted.iterrows():
            if pd.isna(row["r"]):
                continue
            p_fdr = row.get("p_fdr", 1.0)
            sig = ("***" if p_fdr < 0.001 else
                   "**"  if p_fdr < 0.01  else
                   "*"   if p_fdr < 0.05  else "n.s.")
            r_color = "var(--danger)" if row["r"] > 0 else "var(--success)"
            p_color = "var(--accent)" if p_fdr < 0.05 else "var(--text-muted)"
            rows.append(html.Tr([
                html.Td(row["predictor"], style={"fontSize": "10px"}),
                html.Td(row["outcome"],   style={"fontSize": "10px"}),
                html.Td(f"{row['r']:.3f}",
                        style={"color": r_color, "fontWeight": "600",
                               "fontSize": "10px"}),
                html.Td("<.001" if p_fdr < 0.001 else f"{p_fdr:.3f}",
                        style={"fontSize": "10px", "color": p_color}),
                html.Td(sig, style={"fontSize": "10px"}),
                html.Td(str(int(row["n"])), style={"fontSize": "10px"}),
            ]))

        table = dbc.Table([
            html.Thead(html.Tr([html.Th(h)
                                for h in ["Predictor", "Outcome", "r",
                                          "p(FDR)", "Sig.", "n"]])),
            html.Tbody(rows),
        ], bordered=False, size="sm", style={"fontSize": "10px"})

        content = html.Div([
            dcc.Graph(figure=fig, config={
                "displayModeBar": True,
                "toImageButtonOptions": {
                    "format": "png", "scale": 2,
                    "filename": "correlation_heatmap"
                },
            }),
            fdr_note,
            html.Div(table, style={"maxHeight": "400px",
                                    "overflowY": "auto",
                                    "marginTop": "12px"}),
        ])

        # Store full result for export/save
        results_payload = {
            "result": result.to_dict(orient="records"),
            "predictors": predictors,
            "outcomes": outcomes,
            "n_patients": len(merged),
            "rm0035_filter": bool(rm0035 and "rm0035" in rm0035),
        }

        return content, results_payload, False, False, rm0035_note

    # ── Export CSV ───────────────────────────────────────────────────────────
    @app.callback(
        Output("download-corr",      "data"),
        Input("btn-corr-export",     "n_clicks"),
        State("corr-results-store",  "data"),
        prevent_initial_call=True,
    )
    def export_corr(_, results):
        if not results or "result" not in results:
            return no_update
        df = pd.DataFrame(results["result"])
        return dcc.send_data_frame(df.to_csv, "correlations.csv", index=False)

    # ── Save run bundle ──────────────────────────────────────────────────────
    @app.callback(
        Output("corr-save-status",   "children"),
        Input("btn-corr-save",       "n_clicks"),
        State("corr-results-store",  "data"),
        prevent_initial_call=True,
    )
    def save_corr(_, results):
        if not results or "result" not in results:
            return "⚠ Run correlations first"
        try:
            df = pd.DataFrame(results["result"])
            run_dir = save_run(
                run_type="correlations",
                params={
                    "predictors":     results["predictors"],
                    "outcomes":       results["outcomes"],
                    "n_patients":     results["n_patients"],
                    "rm0035_filter":  results["rm0035_filter"],
                },
                dataframes={"correlations.csv": df},
            )
            return f"✓ Saved to {run_dir.name}"
        except Exception as e:
            return f"⚠ Save error: {e}"
