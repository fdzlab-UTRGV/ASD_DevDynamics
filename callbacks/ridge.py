"""
callbacks/ridge.py
─────────────────────────────────────────────────────────────────────────────
Ridge Regression tab callbacks — Figure 3C (Fernandez et al.).

Out-of-sample generalization:
  - Holdout files uploaded here → scored → ridge-holdout-store
  - On Run: train ridge on SPARK discovery sample → predict on holdout
  - Display bar chart of Pearson r per outcome, with significance markers
"""

from __future__ import annotations

import base64
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc

from helpers.store import df_to_store, df_from_store, get_merged_data
from helpers.theme import get_plotly_layout
from modules.ridge import run_oos_ridge
from modules.holdout_loader import load_holdout
import modules.schema as S


# Predictor and outcome column sets — identical to the main Part 2 analysis
# so the discovery model uses the same features as reported in the paper
RIDGE_PREDICTORS = [
    "dcdq_Gross Motor",
    "dcdq_Fine Motor",
    "dcdq_Coordination",
    "rbs_Sensory",
    "rbs_Obsessive",
    "rbs_Ritualistic",
    "rbs_Stereotyped",
    "scq_Social",
    "scq_Communication",
    "scq_Sensory",
]

RIDGE_OUTCOMES = [
    "cbcl_Anxious/Dep.",
    "cbcl_Internalizing",
    "cbcl_Externalizing",
    "cbcl_Social Prob.",
    "cbcl_Attention",
    "cbcl_Thought Prob.",
    "css_total",
    "css_sa",
    "css_rrb",
]

OUTCOME_LABELS = {
    "cbcl_Anxious/Dep.":   "CBCL Anxious/Dep.",
    "cbcl_Internalizing":  "CBCL Internalizing",
    "cbcl_Externalizing":  "CBCL Externalizing",
    "cbcl_Social Prob.":   "CBCL Social Prob.",
    "cbcl_Attention":      "CBCL Attention",
    "cbcl_Thought Prob.":  "CBCL Thought Prob.",
    "css_total":           "ADOS CSS Total",
    "css_sa":              "ADOS CSS Social Affect",
    "css_rrb":             "ADOS CSS RRB",
}

# Domain colors for bar chart
OUTCOME_COLORS = {
    "cbcl_Anxious/Dep.":  "#7c3aed",
    "cbcl_Internalizing": "#7c3aed",
    "cbcl_Externalizing": "#7c3aed",
    "cbcl_Social Prob.":  "#7c3aed",
    "cbcl_Attention":     "#7c3aed",
    "cbcl_Thought Prob.": "#7c3aed",
    "css_total":          "#6b7280",
    "css_sa":             "#6b7280",
    "css_rrb":            "#6b7280",
}


def _write_uploads_to_temp(contents_list, filenames_list):
    """Decode Dash multi-upload contents to temp file paths. Caller deletes."""
    if contents_list is None:
        return []
    if not isinstance(contents_list, list):
        contents_list = [contents_list]
    if filenames_list is None:
        filenames_list = [f"holdout_{i}.csv" for i in range(len(contents_list))]
    elif not isinstance(filenames_list, list):
        filenames_list = [filenames_list]
    while len(filenames_list) < len(contents_list):
        filenames_list.append(f"holdout_{len(filenames_list)}.csv")

    paths = []
    for contents, name in zip(contents_list, filenames_list):
        try:
            _, content_string = contents.split(",")
            data = base64.b64decode(content_string)
            suffix = Path(name).suffix or ".csv"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data)
            tmp.close()
            paths.append(tmp.name)
        except Exception as e:
            print(f"[ridge] holdout parse error for {name}: {e}")
    return paths


def register(app):

    # ── Upload holdout files → score → ridge-holdout-store ───────────────────
    @app.callback(
        Output("ridge-holdout-store",    "data"),
        Output("status-ridge-holdout",   "children"),
        Input("upload-ridge-holdout",    "contents"),
        Input("btn-clear",               "n_clicks"),
        State("upload-ridge-holdout",    "filename"),
        prevent_initial_call=True,
    )
    def upload_holdout(contents, n_clear, filenames):
        if ctx.triggered_id == "btn-clear":
            return None, html.Span("", className="status-muted")
        if contents is None:
            return no_update, no_update

        paths = _write_uploads_to_temp(contents, filenames)
        if not paths:
            return None, html.Span("⚠ Could not read files",
                                   className="status-err")
        try:
            merged, summary = load_holdout(paths)
            if merged is None or merged.empty:
                return None, html.Span(
                    "⚠ No recognized instruments. Expected DCDQ, RBS-R, SCQ, "
                    "or CBCL raw files.",
                    className="status-err")

            n = len(merged)
            inst_str = " · ".join(f"{k}: {v:,}" for k, v in summary.items())
            return df_to_store(merged), html.Span(
                f"✓ {n:,} patients · {inst_str}",
                className="status-ok")
        except Exception as e:
            import traceback; traceback.print_exc()
            return None, html.Span(f"⚠ {str(e)[:80]}", className="status-err")
        finally:
            for p in paths:
                try: Path(p).unlink()
                except: pass

    # ── Run ridge regression ─────────────────────────────────────────────────
    @app.callback(
        Output("ridge-results",       "children"),
        Output("ridge-results-store", "data"),
        Input("ridge-run",        "n_clicks"),
        State("dcdq-store",       "data"),
        State("rbs-store",        "data"),
        State("scq-store",        "data"),
        State("cbcl-store",       "data"),
        State("cov-store",        "data"),
        State("css-store",        "data"),
        State("ridge-holdout-store", "data"),
        State("ridge-n-perm",     "value"),
        State("ridge-alpha-thresh", "value"),
        prevent_initial_call=True,
    )
    def run_ridge(n_clicks, dcdq, rbs, scq, cbcl, cov, css, holdout_data,
                  n_perm, alpha_thresh):
        if not n_clicks:
            return no_update, no_update

        # ── Validate data ────────────────────────────────────────────────────
        discovery = get_merged_data(dcdq=dcdq, rbs=rbs, scq=scq,
                                    cbcl=cbcl, cov=cov, css=css)
        if discovery is None:
            return _msg_card("No SPARK data loaded. Upload instruments in the sidebar."), no_update

        holdout = df_from_store(holdout_data)
        if holdout is None:
            return _msg_card("No holdout file uploaded. Use the upload panel on the left."), no_update

        n_perm_int     = int(n_perm or 1000)
        alpha_f        = float(alpha_thresh or 0.05)

        # ── Run ─────────────────────────────────────────────────────────────
        result = run_oos_ridge(
            discovery=discovery,
            holdout=holdout,
            predictors=RIDGE_PREDICTORS,
            outcomes=RIDGE_OUTCOMES,
            n_perm=n_perm_int,
        )

        if "error" in result:
            return _msg_card(f"Analysis error: {result['error']}"), no_update

        rows = result["results"]
        if not rows:
            return _msg_card("Analysis returned no results. Check predictor/outcome overlap."), no_update

        # ── Build figures ────────────────────────────────────────────────────
        df_res = pd.DataFrame(rows)
        df_res["label"]     = df_res["outcome"].map(OUTCOME_LABELS).fillna(df_res["outcome"])
        df_res["sig"]       = df_res["p_perm"] < alpha_f
        df_res["bar_color"] = df_res["outcome"].map(OUTCOME_COLORS).fillna("#6b7280")
        df_res = df_res.sort_values("r", ascending=False)

        layout_kwargs = get_plotly_layout()

        # Bar chart: Pearson r per outcome
        fig = go.Figure()
        for _, row in df_res.iterrows():
            marker_color = row["bar_color"] if row["sig"] else "#d1d5db"
            fig.add_bar(
                x=[row["label"]],
                y=[row["r"]],
                marker_color=marker_color,
                marker_line_color="var(--border)" if not row["sig"] else marker_color,
                marker_line_width=1.5,
                text=[f'r = {row["r"]:.3f}<br>p = {row["p_perm"]:.3f}'],
                textposition="outside",
                showlegend=False,
                hovertemplate=(
                    f'<b>{row["label"]}</b><br>'
                    f'r = {row["r"]:.3f}<br>'
                    f'R² = {row["r2"]:.3f}<br>'
                    f'p (perm) = {row["p_perm"]:.4f}<br>'
                    f'n discovery = {row["n_discovery"]:,}<br>'
                    f'n holdout = {row["n_holdout"]}<extra></extra>'
                ),
            )

        fig.update_layout(
            **layout_kwargs,
            title=dict(
                text="Out-of-sample Pearson r (SPARK discovery → holdout cohort)",
                font=dict(size=12),
            ),
            xaxis=dict(title="Outcome", tickangle=-30),
            yaxis=dict(title="Pearson r (predicted vs. observed)", range=[0, 0.7]),
            height=420,
            margin=dict(t=60, b=100, l=60, r=20),
            bargap=0.3,
        )

        # Legend annotation
        fig.add_annotation(
            text=(f"■ Significant (p < {alpha_f}, perm. test) &nbsp;"
                  f"□ Non-significant"),
            xref="paper", yref="paper",
            x=0.01, y=1.08,
            showarrow=False,
            font=dict(size=10),
            align="left",
        )

        # Results table
        table_rows = []
        for _, row in df_res.iterrows():
            sig_marker = "***" if row["p_perm"] < 0.001 else (
                         "**"  if row["p_perm"] < 0.01  else (
                         "*"   if row["p_perm"] < alpha_f else "ns"))
            table_rows.append(html.Tr([
                html.Td(row["label"],
                        style={"fontWeight": "600" if row["sig"] else "normal"}),
                html.Td(f'{row["r"]:.3f}'),
                html.Td(f'{row["r2"]:.3f}'),
                html.Td(f'{row["p_perm"]:.4f}'),
                html.Td(sig_marker,
                        style={"color": "var(--accent)" if row["sig"] else "var(--text-muted)"}),
                html.Td(str(row["n_holdout"])),
                html.Td(f'{row["alpha"]:.3g}'),
            ]))

        table = dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Outcome"),
                    html.Th("r"),
                    html.Th("R²"),
                    html.Th("p (perm)"),
                    html.Th("Sig."),
                    html.Th("n holdout"),
                    html.Th("α (CV)"),
                ])),
                html.Tbody(table_rows),
            ],
            bordered=True, hover=True, striped=True, size="sm",
            style={"fontSize": "11px"},
        )

        # Summary stats (Ns vary per outcome — population-level analysis)
        n_sig = int(df_res["sig"].sum())
        n_tot = len(df_res)
        hold_lo, hold_hi = int(df_res["n_holdout"].min()), int(df_res["n_holdout"].max())
        disc_lo, disc_hi = int(df_res["n_discovery"].min()), int(df_res["n_discovery"].max())
        n_pred = int(df_res["n_predictors"].iloc[0]) if len(df_res) else 0

        hold_str = f"{hold_lo}" if hold_lo == hold_hi else f"{hold_lo}–{hold_hi}"
        disc_str = f"{disc_lo:,}" if disc_lo == disc_hi else f"{disc_lo:,}–{disc_hi:,}"

        summary = html.Div([
            html.Span(f"n holdout = {hold_str} · ", className="stat-chip"),
            html.Span(f"n discovery = {disc_str} · ", className="stat-chip"),
            html.Span(f"predictors = {n_pred} · ", className="stat-chip"),
            html.Span(f"{n_sig}/{n_tot} outcomes significant (p < {alpha_f})",
                      className="stat-chip"),
        ], style={"marginBottom": "12px"})

        return html.Div([
            summary,
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.Div("Results table", style={
                "fontSize": "12px", "fontWeight": "700",
                "marginTop": "20px", "marginBottom": "8px",
            }),
            table,
        ]), rows

    # ── Export results ───────────────────────────────────────────────────────
    @app.callback(
        Output("ridge-download", "data"),
        Input("ridge-export",    "n_clicks"),
        State("ridge-results-store", "data"),
        prevent_initial_call=True,
    )
    def export_ridge(n_clicks, results_data):
        if not results_data:
            return dcc.send_string(
                "No results to export. Run the ridge regression first.",
                filename="ridge_results.txt",
            )
        df = pd.DataFrame(results_data)
        df["label"] = df["outcome"].map(OUTCOME_LABELS).fillna(df["outcome"])
        cols = ["label", "outcome", "r", "r2", "p_perm",
                "alpha", "n_discovery", "n_holdout", "n_predictors"]
        cols = [c for c in cols if c in df.columns]
        buf = io.StringIO()
        df[cols].to_csv(buf, index=False, float_format="%.6f")
        return dcc.send_string(buf.getvalue(), filename="ridge_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers

def _msg_card(msg: str) -> html.Div:
    return html.Div(
        msg,
        style={
            "padding": "20px",
            "color": "var(--text-muted)",
            "fontStyle": "italic",
            "fontSize": "12px",
        },
    )
