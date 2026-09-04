"""
callbacks/cohort.py
─────────────────────────────────────────────────────────────────────────────
Replication Cohort tab — single scrolling page running the full √ΔR² suite on
an independently loaded cohort (Option A).

Analyses (all run on one "Run" click):
  1. Hubness ranking (feature level) + ranked table
  2. PCA of the √ΔR² matrix
  3. Suppressor / anxiety-adjustment (Δ√ΔR² heatmap + before/after scatter)
  4. Age-stratified hubs (grouped bars + hub-rank heatmap)
  5. Domain-composite √ΔR² (+ its own PCA)

Each analysis reuses the same underlying functions as the discovery tabs, so
results are directly comparable.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, no_update, ALL
import dash_bootstrap_components as dbc

from helpers.store import df_to_store, df_from_store
from helpers.theme import get_plotly_layout, get_axis_style
from modules.cohort_loader import load_cohort
from modules.mass_univariate import run_mass_univariate
from modules.hubness import compute_hubness
from modules.age_hub import run_age_stratified_hubness, AGE_BANDS
from modules.domains import compute_domain_composites, DOMAIN_ORDER
from layouts.cohort import covariate_mapping_panel


PREDICTORS = [
    "dcdq_Gross Motor", "dcdq_Fine Motor", "dcdq_Coordination",
    "rbs_Sensory", "rbs_Obsessive", "rbs_Ritualistic", "rbs_Stereotyped",
    "scq_Social", "scq_Communication", "scq_Sensory",
]
OUTCOMES = [
    "cbcl_Internalizing", "cbcl_Externalizing", "cbcl_Anxious/Dep.",
    "cbcl_Social Prob.", "cbcl_Attention", "cbcl_Thought Prob.",
]
POS_COLOR = "#f87171"   # outcome-positive / CBCL side
NEG_COLOR = "#34d399"   # outcome-negative / ADOS side


# ─────────────────────────────────────────────────────────────────────────────
# Upload helpers

def _write_temp(contents_list, filenames_list):
    if contents_list is None:
        return []
    if not isinstance(contents_list, list):
        contents_list = [contents_list]
    if filenames_list is None:
        filenames_list = [f"cohort_{i}.csv" for i in range(len(contents_list))]
    elif not isinstance(filenames_list, list):
        filenames_list = [filenames_list]
    while len(filenames_list) < len(contents_list):
        filenames_list.append(f"cohort_{len(filenames_list)}.csv")
    paths = []
    for contents, name in zip(contents_list, filenames_list):
        try:
            _, b64 = contents.split(",")
            data = base64.b64decode(b64)
            suffix = Path(name).suffix or ".csv"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data); tmp.close()
            paths.append(tmp.name)
        except Exception as e:
            print(f"[cohort] parse error for {name}: {e}")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Section builders (each returns a Dash component)

def _section_title(text, sub=None):
    kids = [html.Div(text, style={"fontSize": "13px", "fontWeight": "700",
                                  "marginTop": "18px", "marginBottom": "2px"})]
    if sub:
        kids.append(html.Div(sub, style={"fontSize": "10px",
                                         "color": "var(--text-muted)",
                                         "marginBottom": "8px"}))
    return html.Div(kids)


def _build_hubness(result, thresh, preds, outs, theme):
    payload = {"sqrt_dr2": result["sqrt_dr2"].to_dict(),
               "pval_fdr": result["pval_fdr"].to_dict(),
               "n_obs": result["n_obs"].to_dict(),
               "predictors": preds, "outcomes": outs}
    hub = compute_hubness(payload, fdr_thresh=thresh, sig_only=True)
    lay = get_plotly_layout(theme)

    hub_s = hub.sort_values("hubness_index", ascending=True)
    fig = go.Figure(go.Bar(
        x=hub_s["hubness_index"], y=hub_s["predictor"], orientation="h",
        marker_color=hub_s["domain_color"],
        text=[f"{v:.2f}" for v in hub_s["hubness_index"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Hubness = %{x:.3f}<extra></extra>"))
    fig.update_layout(**lay, height=320,
                      margin=dict(t=20, b=40, l=150, r=40),
                      xaxis=dict(title="Hubness Index (Σ|√ΔR²| over FDR-sig outcomes)"),
                      yaxis=dict(title=""))

    # ranked table
    header = html.Thead(html.Tr([html.Th(h) for h in
             ["Rank", "Predictor", "Domain", "Hubness", "Sig/Total",
              "Mean|√ΔR²|", "Max|√ΔR²|"]]))
    body_rows = []
    for _, r in hub.iterrows():
        body_rows.append(html.Tr([
            html.Td(str(int(r["rank"]))),
            html.Td(r["predictor"], style={"fontWeight": "600"}),
            html.Td(r["domain"]),
            html.Td(f"{r['hubness_index']:.3f}"),
            html.Td(f"{int(r['n_significant'])}/{int(r['n_total'])}"),
            html.Td(f"{r['mean_abs_effect']:.3f}" if pd.notna(r['mean_abs_effect']) else "—"),
            html.Td(f"{r['max_abs_effect']:.3f}" if pd.notna(r['max_abs_effect']) else "—"),
        ]))
    table = dbc.Table([header, html.Tbody(body_rows)],
                      bordered=True, hover=True, striped=True, size="sm",
                      style={"fontSize": "11px"})
    return html.Div([
        _section_title("1 · Hubness ranking",
                       "Σ|√ΔR²| across FDR-significant outcomes, per predictor."),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        table,
    ]), hub


def _build_pca(mat, theme, label="2 · PCA of √ΔR² matrix"):
    mat = mat.dropna(how="all").dropna(axis=1, how="all")
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        return html.Div([_section_title(label),
                         dbc.Alert("Not enough data for PCA.", color="warning",
                                   style={"fontSize": "11px"})])
    Z = mat.fillna(0).values.astype(float)
    Z = Z - Z.mean(axis=0)
    std = Z.std(axis=0); std[std == 0] = 1; Z = Z / std
    try:
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:
        return html.Div([_section_title(label),
                         dbc.Alert("SVD failed.", color="danger",
                                   style={"fontSize": "11px"})])
    var_exp = (s**2) / (s**2).sum()
    pc1_load = Vt[0]; pc1_score = U[:, 0] * s[0]
    lay = get_plotly_layout(theme); ax = get_axis_style(theme)

    fig_scree = go.Figure(go.Bar(
        x=[f"PC{i+1}" for i in range(min(8, len(s)))],
        y=[float(v*100) for v in var_exp[:8]], marker_color="#38bdf8"))
    fig_scree.update_layout(**lay, height=220,
        title=dict(text="Variance explained per PC", font=dict(size=12)),
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis=dict(title="% variance"))

    fig_load = go.Figure(go.Bar(
        x=mat.columns.tolist(), y=pc1_load.tolist(),
        marker_color=[POS_COLOR if v > 0 else NEG_COLOR for v in pc1_load]))
    fig_load.update_layout(**lay, height=280,
        title=dict(text=f"PC1 outcome loadings ({var_exp[0]*100:.1f}% variance)",
                   font=dict(size=12)),
        margin=dict(l=40, r=20, t=40, b=90),
        xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
        yaxis=dict(title="Loading"))

    fig_score = go.Figure(go.Bar(
        x=mat.index.tolist(), y=pc1_score.tolist(),
        marker_color=[POS_COLOR if v > 0 else NEG_COLOR for v in pc1_score]))
    fig_score.update_layout(**lay, height=300,
        title=dict(text="PC1 predictor scores", font=dict(size=12)),
        margin=dict(l=40, r=20, t=40, b=110),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(title="Score"))

    return html.Div([
        _section_title(label,
                       f"PC1 explains {var_exp[0]*100:.1f}% of √ΔR² variance."),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_scree,
                              config={"displayModeBar": False}), width=4),
            dbc.Col(dcc.Graph(figure=fig_load,
                              config={"displayModeBar": False}), width=8),
        ]),
        dcc.Graph(figure=fig_score, config={"displayModeBar": False}),
    ])


def _build_suppressor(merged, preds, outs, cov_present, suppressor, theme, return_data=False):
    if suppressor not in merged.columns:
        blk = html.Div([_section_title("3 · Suppressor / anxiety-adjustment"),
                        dbc.Alert(f"Suppressor '{suppressor}' not in cohort.",
                                  color="warning", style={"fontSize": "11px"})])
        return (blk, None) if return_data else blk
    sup_preds = [p for p in preds if p != suppressor]
    sup_outs  = [o for o in outs if o != suppressor]
    base = run_mass_univariate(merged, sup_preds, sup_outs, cov_present)
    aug  = run_mass_univariate(merged, sup_preds, sup_outs,
                               cov_present + [suppressor])
    if "error" in base or "error" in aug:
        blk = html.Div([_section_title("3 · Suppressor / anxiety-adjustment"),
                        dbc.Alert(base.get("error") or aug.get("error"),
                                  color="danger", style={"fontSize": "11px"})])
        return (blk, None) if return_data else blk

    b = base["sqrt_dr2"]; a = aug["sqrt_dr2"]
    delta = (a.abs() - b.abs())  # + = revealed, - = concealed
    lay = get_plotly_layout(theme)

    # scatter before vs after (|√ΔR²|)
    bx = b.abs().values.flatten(); ay = a.abs().values.flatten()
    m = ~(np.isnan(bx) | np.isnan(ay))
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=bx[m], y=ay[m], mode="markers",
        marker=dict(color=[POS_COLOR if (ay[i]-bx[i]) < 0 else NEG_COLOR
                           for i in range(len(bx)) if m[i]], size=7,
                    opacity=0.7),
        hovertemplate="base=%{x:.3f}<br>+supp=%{y:.3f}<extra></extra>"))
    lim = float(np.nanmax([bx[m].max() if m.any() else 0.5,
                           ay[m].max() if m.any() else 0.5]))
    fig_sc.add_trace(go.Scatter(x=[-0.05, lim], y=[-0.05, lim], mode="lines",
                    line=dict(dash="dash", color="#888"), showlegend=False))
    fig_sc.update_layout(**lay, height=360,
        title=dict(text="|√ΔR²| before vs after adding suppressor",
                   font=dict(size=12)),
        margin=dict(l=50, r=20, t=40, b=50),
        xaxis=dict(title="|√ΔR²| base model"),
        yaxis=dict(title="|√ΔR²| + suppressor"), showlegend=False)

    # delta heatmap
    fig_hm = go.Figure(go.Heatmap(
        z=delta.values.astype(float), x=list(delta.columns), y=list(delta.index),
        colorscale="RdBu", zmid=0,
        colorbar=dict(title="Δ|√ΔR²|"),
        text=[[f"{v:+.3f}" if pd.notna(v) else "" for v in row]
              for row in delta.values],
        texttemplate="%{text}", textfont=dict(size=9),
        hovertemplate="%{y} → %{x}<br>Δ = %{z:+.3f}<extra></extra>"))
    fig_hm.update_layout(**lay, height=420,
        title=dict(text=f"Δ|√ΔR²| when '{suppressor}' added as covariate",
                   font=dict(size=12)),
        margin=dict(l=140, r=20, t=40, b=90),
        xaxis=dict(tickangle=-30))

    concealed = int((delta.values < -0.01).sum())
    revealed  = int((delta.values > 0.01).sum())
    meand = float(np.nanmean(delta.values))
    summ = html.Div([
        html.Span(f"{concealed} pairs concealed (Δ < −0.01) · ",
                  className="stat-chip"),
        html.Span(f"{revealed} revealed (Δ > +0.01) · ", className="stat-chip"),
        html.Span(f"mean Δ = {meand:+.4f}", className="stat-chip"),
    ], style={"marginBottom": "8px"})

    blk = html.Div([
        _section_title("3 · Suppressor / anxiety-adjustment",
                       "Effect of adding the suppressor as a covariate on every "
                       "predictor→outcome |√ΔR²|. Negative = concealed."),
        summ,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_sc,
                              config={"displayModeBar": False}), width=5),
            dbc.Col(dcc.Graph(figure=fig_hm,
                              config={"displayModeBar": False}), width=7),
        ]),
    ])
    return (blk, delta) if return_data else blk


def _build_age_hubs(merged, preds, outs, cov_present, thresh, theme, return_data=False):
    age_cov = [c for c in cov_present if c not in ("age_months",)]
    res = run_age_stratified_hubness(merged, preds, outs, age_cov,
                                     age_col="age_months", fdr_thresh=thresh)
    if "error" in res:
        blk = html.Div([_section_title("4 · Age-stratified hubs"),
                        dbc.Alert(res["error"], color="warning",
                                  style={"fontSize": "11px"})])
        return (blk, None) if return_data else blk
    hub_matrix = res["hub_matrix"]      # predictors × bands (hubness)
    rank_matrix = res["rank_matrix"]    # predictors × bands (rank)
    band_ns = res.get("band_ns", {})
    lay = get_plotly_layout(theme)
    lay = {**lay, "legend": {"orientation": "h", "y": 1.05,
                             "font": {"size": 9}}}

    band_cols = [c for c in hub_matrix.columns]
    palette = ["#c08457", "#8d5a7c", "#3f7d8c", "#4a90c2", "#5aa17f"]

    fig_bar = go.Figure()
    for i, band in enumerate(band_cols):
        n_lbl = f" (n={band_ns.get(band, 0):,})" if band in band_ns else ""
        fig_bar.add_bar(name=f"{band}{n_lbl}", x=list(hub_matrix.index),
                        y=hub_matrix[band].values,
                        marker_color=palette[i % len(palette)])
    fig_bar.update_layout(**lay, barmode="group", height=380,
        title=dict(text="Hubness Index by age band", font=dict(size=12)),
        margin=dict(l=50, r=20, t=40, b=110),
        xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
        yaxis=dict(title="Hubness Index (Σ|√ΔR²|)"))

    fig_rank = go.Figure(go.Heatmap(
        z=rank_matrix.values.astype(float), x=list(rank_matrix.columns),
        y=list(rank_matrix.index), colorscale="RdBu", reversescale=False,
        colorbar=dict(title="Rank<br>(1=top)"),
        text=[[f"{int(v)}" if pd.notna(v) else "" for v in row]
              for row in rank_matrix.values],
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="%{y}<br>%{x}<br>rank %{z}<extra></extra>"))
    fig_rank.update_layout(**lay, height=380,
        title=dict(text="Hub rank by age band (1 = top hub)", font=dict(size=12)),
        margin=dict(l=140, r=20, t=40, b=60))

    stab = res.get("stability", {})
    mean_r = stab.get("mean_r")
    sub = ("Hubness recomputed within each age band. "
           + (f"Mean pairwise rank stability ρ = {mean_r:.3f}."
              if mean_r is not None else ""))

    blk = html.Div([
        _section_title("4 · Age-stratified hubs", sub),
        dcc.Graph(figure=fig_bar, config={"displayModeBar": False}),
        dcc.Graph(figure=fig_rank, config={"displayModeBar": False}),
    ])
    if return_data:
        return blk, {"hub": hub_matrix.to_json(), "rank": rank_matrix.to_json()}
    return blk


def _build_domain_composite(merged, outs, cov_present, thresh, theme, return_data=False):
    comp = compute_domain_composites(merged)
    if comp.empty:
        blk = html.Div([_section_title("5 · Domain-composite √ΔR²"),
                        dbc.Alert("Could not compute domain composites.",
                                  color="warning", style={"fontSize": "11px"})])
        return (blk, None) if return_data else blk
    dom_cols = [c for c in comp.columns]
    dmerged = merged.join(comp, how="left")
    result = run_mass_univariate(dmerged, dom_cols, outs, cov_present)
    if "error" in result:
        blk = html.Div([_section_title("5 · Domain-composite √ΔR²"),
                        dbc.Alert(result["error"], color="danger",
                                  style={"fontSize": "11px"})])
        return (blk, None) if return_data else blk
    mat = result["sqrt_dr2"]
    lay = get_plotly_layout(theme)

    fig_hm = go.Figure(go.Heatmap(
        z=mat.values.astype(float), x=list(mat.columns), y=list(mat.index),
        colorscale="RdBu_r", zmid=0, colorbar=dict(title="√ΔR²"),
        text=[[f"{v:+.3f}" if pd.notna(v) else "" for v in row]
              for row in mat.values],
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="%{y} → %{x}<br>√ΔR² = %{z:+.3f}<extra></extra>"))
    fig_hm.update_layout(**lay, height=300,
        title=dict(text="Domain-composite √ΔR² matrix", font=dict(size=12)),
        margin=dict(l=110, r=20, t=40, b=90),
        xaxis=dict(tickangle=-30))

    pca_block = _build_pca(mat, theme, label="   Domain-composite PCA")

    blk = html.Div([
        _section_title("5 · Domain-composite √ΔR²",
                       "Each predictor row is a domain composite (z-score mean "
                       "across constituent features)."),
        dcc.Graph(figure=fig_hm, config={"displayModeBar": False}),
        pca_block,
    ])
    return (blk, mat) if return_data else blk


# ─────────────────────────────────────────────────────────────────────────────
# Registration

def register(app):

    # ── Upload + score cohort ────────────────────────────────────────────────
    @app.callback(
        Output("cohort-store",          "data"),
        Output("cohort-cov-cols-store", "data"),
        Output("status-cohort",         "children"),
        Output("cohort-cov-mapping",    "children"),
        Input("upload-cohort",          "contents"),
        Input("btn-clear",              "n_clicks"),
        State("upload-cohort",          "filename"),
        prevent_initial_call=True,
    )
    def upload_cohort(contents, n_clear, filenames):
        if ctx.triggered_id == "btn-clear":
            return None, None, html.Span("", className="status-muted"), None
        if contents is None:
            return no_update, no_update, no_update, no_update
        paths = _write_temp(contents, filenames)
        if not paths:
            return None, None, html.Span("⚠ Could not read files",
                                         className="status-err"), None
        try:
            res = load_cohort(paths)
            if "error" in res:
                return None, None, html.Span(f"⚠ {res['error']}",
                                             className="status-err"), None
            merged = res["merged"]
            summary = res["instrument_summary"]
            overlap = res.get("overlap_report", {})
            n = len(merged)
            inst_str = " · ".join(f"{k}: {v:,}" for k, v in summary.items())
            status_kids = [html.Span(f"✓ {n:,} patients · {inst_str}",
                                     className="status-ok")]
            # Flag instruments with low outcome overlap (join problem)
            low = []
            label_map = {"dcdq": "DCDQ", "rbs": "RBS-R", "scq": "SCQ",
                         "cbcl": "CBCL"}
            for name, info in overlap.items():
                if name == "cbcl":
                    continue
                if info["overlap_with_outcomes"] == 0 and info["n"] > 0:
                    low.append(f"{label_map.get(name, name)} (0% overlap with CBCL)")
                elif info["pct"] < 25 and info["n"] > 0:
                    low.append(f"{label_map.get(name, name)} ({info['pct']:.0f}% overlap)")
            if low:
                status_kids.append(html.Div(
                    "⚠ Low person_id overlap with outcomes: " + ", ".join(low)
                    + ". These predictors will show 0/0. Check that person_id "
                    "formats match across files.",
                    style={"fontSize": "10px", "color": "var(--danger)",
                           "marginTop": "4px"}))
            status = html.Div(status_kids)
            panel = covariate_mapping_panel(res["covariate_report"],
                                            res["covariate_columns"],
                                            res["unmatched_core"])
            return (df_to_store(merged),
                    {"columns": res["covariate_columns"]}, status, panel)
        except Exception as e:
            import traceback; traceback.print_exc()
            return None, None, html.Span(f"⚠ {str(e)[:80]}",
                                         className="status-err"), None
        finally:
            for p in paths:
                try: Path(p).unlink()
                except: pass

    # -- Export all results to Excel --
    @app.callback(
        Output("cohort-download",     "data"),
        Input("cohort-export",        "n_clicks"),
        State("cohort-export-store",  "data"),
        prevent_initial_call=True,
    )
    def export_cohort(n_clicks, export_data):
        if not n_clicks or not export_data:
            return no_update
        import io
        buf = io.BytesIO()

        def _read(key):
            try:
                return pd.read_json(io.StringIO(export_data[key]))
            except Exception:
                return None

        sheets = {
            "hubness":          _read("hubness"),
            "sqrt_dr2":         _read("sqrt_dr2"),
            "pval_fdr":         _read("pval_fdr"),
            "n_obs":            _read("n_obs"),
            "suppressor_delta": _read("suppressor_delta"),
            "age_hubness":      _read("age_hubness"),
            "age_rank":         _read("age_rank"),
            "domain_sqrt_dr2":  _read("domain_sqrt_dr2"),
        }
        try:
            with pd.ExcelWriter(buf, engine="openpyxl") as xl:
                wrote_any = False
                for name, df in sheets.items():
                    if df is not None and not df.empty:
                        df.to_excel(xl, sheet_name=name[:31])
                        wrote_any = True
                if not wrote_any:
                    pd.DataFrame({"note": ["No results to export."]}).to_excel(
                        xl, sheet_name="empty", index=False)
        except Exception as e:
            print(f"[cohort export] {e}")
            return no_update
        buf.seek(0)
        return dcc.send_bytes(buf.getvalue(),
                              "replication_cohort_results.xlsx")

    # -- Run all five analyses --
    @app.callback(
        Output("cohort-results",      "children"),
        Output("cohort-export-store", "data"),
        Input("cohort-run",           "n_clicks"),
        State("cohort-store",      "data"),
        State("cohort-covariates", "value"),
        State("cohort-suppressor", "value"),
        State("cohort-fdr",        "value"),
        State("theme-store",       "data"),
        prevent_initial_call=True,
    )
    def run_all(n_clicks, cohort_data, covariates, suppressor, fdr_thresh, theme):
        if not n_clicks:
            return no_update, no_update
        merged = df_from_store(cohort_data)
        if merged is None or merged.empty:
            return _msg("No cohort loaded. Upload the cohort's files first."), None

        preds = [p for p in PREDICTORS if p in merged.columns]
        outs  = [o for o in OUTCOMES   if o in merged.columns]
        if not preds or not outs:
            return (_msg("Cohort is missing predictor or outcome columns after "
                        "scoring. Check the uploaded instrument files."), None)

        cov_present = [c for c in (covariates or []) if c in merged.columns]
        missing_cov = [c for c in (covariates or []) if c not in merged.columns]
        thresh = float(fdr_thresh or 0.05)
        t = theme or "dark"

        # Base √ΔR² (shared by hubness + PCA)
        result = run_mass_univariate(merged, preds, outs, cov_present)
        if "error" in result:
            return _msg(f"Analysis error: {result['error']}"), None

        # Export payload — collected as we go, serialized to JSON-able dict
        export = {}
        export["sqrt_dr2"] = result["sqrt_dr2"].to_json()
        export["pval_fdr"] = result["pval_fdr"].to_json()
        export["n_obs"]    = result["n_obs"].to_json()

        sections = []

        # header chips
        n_range = f"{int(result['n_obs'].min().min()):,}–{int(result['n_obs'].max().max()):,}"
        chips = [
            html.Span(f"predictors = {len(preds)} · ", className="stat-chip"),
            html.Span(f"outcomes = {len(outs)} · ", className="stat-chip"),
            html.Span(f"pairwise N = {n_range} · ", className="stat-chip"),
            html.Span(f"covariates = {', '.join(cov_present) or 'none'}",
                      className="stat-chip"),
        ]
        sections.append(html.Div(chips, style={"marginBottom": "6px"}))
        if missing_cov:
            sections.append(dbc.Alert(
                f"Covariate(s) not found and skipped: {', '.join(missing_cov)}. "
                "Map them in the panel above if the cohort has them.",
                color="warning", style={"fontSize": "11px", "padding": "8px"}))

        # 1 · Hubness
        try:
            hub_block, hub_df = _build_hubness(result, thresh, preds, outs, t)
            sections.append(hub_block)
            export["hubness"] = hub_df.to_json()
        except Exception as e:
            sections.append(_err_section("1 · Hubness ranking", e))

        # 2 · PCA
        try:
            sections.append(_build_pca(result["sqrt_dr2"], t))
        except Exception as e:
            sections.append(_err_section("2 · PCA of √ΔR² matrix", e))

        # 3 · Suppressor
        try:
            sup_block, sup_delta = _build_suppressor(
                merged, preds, outs, cov_present,
                suppressor or "cbcl_Anxious/Dep.", t, return_data=True)
            sections.append(sup_block)
            if sup_delta is not None:
                export["suppressor_delta"] = sup_delta.to_json()
        except Exception as e:
            sections.append(_err_section("3 · Suppressor / anxiety-adjustment", e))

        # 4 · Age-stratified hubs
        if "age_months" in merged.columns or "age_years" in merged.columns:
            try:
                age_block, age_data = _build_age_hubs(
                    merged, preds, outs, cov_present, thresh, t, return_data=True)
                sections.append(age_block)
                if age_data is not None:
                    export["age_hubness"] = age_data.get("hub", "")
                    export["age_rank"] = age_data.get("rank", "")
            except Exception as e:
                sections.append(_err_section("4 · Age-stratified hubs", e))
        else:
            sections.append(html.Div([
                _section_title("4 · Age-stratified hubs"),
                dbc.Alert("No age column in cohort — map an age covariate above "
                          "to enable this analysis.", color="warning",
                          style={"fontSize": "11px"})]))

        # 5 · Domain composites
        try:
            dom_block, dom_mat = _build_domain_composite(
                merged, outs, cov_present, thresh, t, return_data=True)
            sections.append(dom_block)
            if dom_mat is not None:
                export["domain_sqrt_dr2"] = dom_mat.to_json()
        except Exception as e:
            sections.append(_err_section("5 · Domain-composite √ΔR²", e))

        return html.Div(sections), export


def _msg(text):
    return html.Div(text, style={"padding": "20px", "fontStyle": "italic",
                                 "fontSize": "12px", "color": "var(--text-muted)"})


def _err_section(title, e):
    return html.Div([_section_title(title),
                     dbc.Alert(f"Section failed: {str(e)[:120]}",
                               color="danger", style={"fontSize": "11px"})])
